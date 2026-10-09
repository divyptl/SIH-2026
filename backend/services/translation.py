"""Indic ⇄ English translation layer in front of the analysis models.

Every specialist model and the OpenRouter baseline are prompted in English, so
queries in the 22 scheduled Indian languages are translated to English before
they reach the controller, and the English answer is translated back for
display. Translation is done locally with AI4Bharat's IndicTrans2 distilled
checkpoints (one model per direction), loaded lazily on first use.

The heavy dependencies (torch, transformers, IndicTransToolkit) are imported
inside the loader, so the API still starts and serves English queries when they
are missing — only non-English requests fail, with an actionable error.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from config import Settings, get_settings

logger = logging.getLogger("satquery.translation")

ENGLISH = "en"
ENGLISH_FLORES = "eng_Latn"


@dataclass(frozen=True)
class Language:
    """One supported language.

    ``scripts`` maps an ISO 15924 script to the IndicTrans2 (FLORES-200) tag for
    this language written in that script. The first entry is the script the
    answer is translated into.
    """

    code: str
    name: str
    native_name: str
    scripts: dict[str, str]
    rtl: bool = False

    @property
    def flores(self) -> str:
        return next(iter(self.scripts.values()))


LANGUAGES: dict[str, Language] = {
    lang.code: lang
    for lang in (
        Language("en", "English", "English", {"Latn": ENGLISH_FLORES}),
        Language("as", "Assamese", "অসমীয়া", {"Beng": "asm_Beng"}),
        Language("bn", "Bengali", "বাংলা", {"Beng": "ben_Beng"}),
        Language("brx", "Bodo", "बड़ो", {"Deva": "brx_Deva"}),
        Language("doi", "Dogri", "डोगरी", {"Deva": "doi_Deva"}),
        Language("gu", "Gujarati", "ગુજરાતી", {"Gujr": "guj_Gujr"}),
        Language("hi", "Hindi", "हिन्दी", {"Deva": "hin_Deva"}),
        Language("kn", "Kannada", "ಕನ್ನಡ", {"Knda": "kan_Knda"}),
        Language("ks", "Kashmiri", "کٲشُر", {"Arab": "kas_Arab", "Deva": "kas_Deva"}, rtl=True),
        Language("gom", "Konkani", "कोंकणी", {"Deva": "gom_Deva"}),
        Language("mai", "Maithili", "मैथिली", {"Deva": "mai_Deva"}),
        Language("ml", "Malayalam", "മലയാളം", {"Mlym": "mal_Mlym"}),
        Language("mni", "Manipuri", "ꯃꯤꯇꯩꯂꯣꯟ", {"Mtei": "mni_Mtei", "Beng": "mni_Beng"}),
        Language("mr", "Marathi", "मराठी", {"Deva": "mar_Deva"}),
        Language("ne", "Nepali", "नेपाली", {"Deva": "npi_Deva"}),
        Language("or", "Odia", "ଓଡ଼ିଆ", {"Orya": "ory_Orya"}),
        Language("pa", "Punjabi", "ਪੰਜਾਬੀ", {"Guru": "pan_Guru"}),
        Language("sa", "Sanskrit", "संस्कृतम्", {"Deva": "san_Deva"}),
        Language("sat", "Santali", "ᱥᱟᱱᱛᱟᱲᱤ", {"Olck": "sat_Olck"}),
        Language("sd", "Sindhi", "سنڌي", {"Arab": "snd_Arab", "Deva": "snd_Deva"}, rtl=True),
        Language("ta", "Tamil", "தமிழ்", {"Taml": "tam_Taml"}),
        Language("te", "Telugu", "తెలుగు", {"Telu": "tel_Telu"}),
        Language("ur", "Urdu", "اردو", {"Arab": "urd_Arab"}, rtl=True),
    )
}

# Unicode blocks used to tell which script a query is written in.
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("Deva", 0x0900, 0x097F),
    ("Deva", 0xA8E0, 0xA8FF),
    ("Beng", 0x0980, 0x09FF),
    ("Guru", 0x0A00, 0x0A7F),
    ("Gujr", 0x0A80, 0x0AFF),
    ("Orya", 0x0B00, 0x0B7F),
    ("Taml", 0x0B80, 0x0BFF),
    ("Telu", 0x0C00, 0x0C7F),
    ("Knda", 0x0C80, 0x0CFF),
    ("Mlym", 0x0D00, 0x0D7F),
    ("Arab", 0x0600, 0x06FF),
    ("Arab", 0x0750, 0x077F),
    ("Arab", 0xFB50, 0xFDFF),
    ("Arab", 0xFE70, 0xFEFF),
    ("Olck", 0x1C50, 0x1C7F),
    ("Mtei", 0xABC0, 0xABFF),
    ("Mtei", 0xAAE0, 0xAAFF),
)

# Which language to assume for a script when the client gives no usable hint.
# Several languages share Devanagari, Bengali and Arabic script; the most widely
# spoken one is the default, and the client's language hint disambiguates.
_SCRIPT_DEFAULT_LANGUAGE: dict[str, str] = {
    "Deva": "hi",
    "Beng": "bn",
    "Guru": "pa",
    "Gujr": "gu",
    "Orya": "or",
    "Taml": "ta",
    "Telu": "te",
    "Knda": "kn",
    "Mlym": "ml",
    "Arab": "ur",
    "Olck": "sat",
    "Mtei": "mni",
}


class TranslationError(RuntimeError):
    """Translation could not be performed."""


def _script_of(char: str) -> str | None:
    point = ord(char)
    if char.isascii():
        return "Latn" if char.isalpha() else None
    for script, start, end in _SCRIPT_RANGES:
        if start <= point <= end:
            return script
    return None


def dominant_script(text: str) -> str | None:
    """The script most letters of ``text`` are written in, or None if no letters."""
    counts: dict[str, int] = {}
    for char in text:
        script = _script_of(char)
        if script:
            counts[script] = counts.get(script, 0) + 1
    if not counts:
        return None
    # Indic text often carries Latin acronyms ("SAR", "NDVI"); any real amount
    # of Indic script means the query is not English.
    indic = {script: n for script, n in counts.items() if script != "Latn"}
    if indic and sum(indic.values()) >= 0.2 * sum(counts.values()):
        return max(indic, key=lambda script: indic[script])
    return "Latn"


@dataclass(frozen=True)
class ResolvedLanguage:
    language: Language
    flores: str
    detected: bool
    """False when the client's hint was used, True when inferred from the script."""


def resolve_source_language(text: str, hint: str | None) -> ResolvedLanguage:
    """Work out which language (and script) the query is written in.

    The script is read from the text itself. The client's hint picks between
    languages that share a script (Hindi vs Marathi, Bengali vs Assamese, ...);
    if the hint does not match the script — for example an English query typed
    while the UI is set to Tamil — the script wins.
    """
    script = dominant_script(text)
    if script is None or script == "Latn":
        return ResolvedLanguage(LANGUAGES[ENGLISH], ENGLISH_FLORES, detected=hint != ENGLISH)

    hinted = LANGUAGES.get(hint or "")
    if hinted and script in hinted.scripts:
        return ResolvedLanguage(hinted, hinted.scripts[script], detected=False)

    language = LANGUAGES[_SCRIPT_DEFAULT_LANGUAGE[script]]
    return ResolvedLanguage(language, language.scripts[script], detected=True)


# Sentence boundaries: Latin/Indic full stops, danda (।) and double danda (॥),
# Urdu full stop (۔), question/exclamation marks, followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?।॥۔؟])\s+")

# IndicTransToolkit's detokeniser splits Meetei Mayek clusters around the apun
# iyek (virama, U+ABED), e.g. "ꯇ ꯭ ꯌꯨꯟ" instead of "ꯇ꯭ꯌꯨꯟ".
_MEETEI_VIRAMA = "꯭"
_MEETEI_VIRAMA_SPACES = re.compile(rf"\s*{_MEETEI_VIRAMA}\s*")


def _split_sentences(paragraph: str) -> list[str]:
    return [part for part in _SENTENCE_END.split(paragraph.strip()) if part]


def dependencies_available() -> tuple[bool, str | None]:
    """Whether the optional translation dependencies can be imported."""
    missing = [
        module
        for module in ("torch", "transformers", "IndicTransToolkit")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return False, f"Missing Python packages: {', '.join(missing)}. Run `uv sync` in backend/."
    return True, None


class _Direction:
    """One IndicTrans2 checkpoint plus its tokenizer, loaded on demand."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._lock = threading.Lock()
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, device: str) -> tuple[Any, Any]:
        with self._lock:
            if self._model is None:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                started = time.perf_counter()
                dtype = torch.float16 if device.startswith("cuda") else torch.float32
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_id, trust_remote_code=True, dtype=dtype
                    )
                except OSError as exc:
                    raise TranslationError(
                        f"Could not load '{self.model_id}'. IndicTrans2 checkpoints are gated: "
                        "accept the licence on its Hugging Face page, then set HF_TOKEN in "
                        f"backend/.env (or run `hf auth login`). Underlying error: {exc}"
                    ) from exc
                # IndicTrans2's remote modeling code only understands legacy tuple
                # caches (it indexes past_key_values[0][0].shape and ships its own
                # tuple-based _reorder_cache). Newer transformers otherwise hands
                # it an empty EncoderDecoderCache, which crashes on the first
                # decoder step; this keeps generate() on the legacy-cache path.
                type(model)._supports_default_dynamic_cache = classmethod(lambda cls: False)
                self._model = model.to(device).eval()
                self._tokenizer = tokenizer
                logger.info(
                    "Loaded %s on %s in %.1fs",
                    self.model_id,
                    device,
                    time.perf_counter() - started,
                )
            return self._model, self._tokenizer


class IndicTranslator:
    """Batch translator between English and the 22 scheduled Indian languages."""

    engine = "IndicTrans2"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._to_english = _Direction(settings.indic_en_model)
        self._from_english = _Direction(settings.en_indic_model)
        self._processor: Any = None
        self._processor_lock = threading.Lock()
        self._device: str | None = None

    # -- status ---------------------------------------------------------------

    def status(self) -> tuple[str, str | None]:
        """``(state, detail)`` where state is disabled / unavailable / idle / ready."""
        if not self._settings.translation_enabled:
            return "disabled", "TRANSLATION_ENABLED is false."
        ok, reason = dependencies_available()
        if not ok:
            return "unavailable", reason
        if self._to_english.loaded and self._from_english.loaded:
            return "ready", None
        return "idle", "Models load on the first non-English request."

    # -- internals ------------------------------------------------------------

    def _resolve_device(self) -> str:
        if self._device is None:
            import torch

            wanted = self._settings.translation_device
            if wanted == "auto":
                wanted = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = wanted
        return self._device

    def _get_processor(self) -> Any:
        with self._processor_lock:
            if self._processor is None:
                from IndicTransToolkit.processor import IndicProcessor

                self._processor = IndicProcessor(inference=True)
            return self._processor

    def _check_enabled(self) -> None:
        state, detail = self.status()
        if state in {"disabled", "unavailable"}:
            raise TranslationError(f"Translation is {state}: {detail}")

    def _translate_sentences(
        self, direction: _Direction, sentences: list[str], src: str, tgt: str
    ) -> list[str]:
        import torch

        device = self._resolve_device()
        model, tokenizer = direction.load(device)
        processor = self._get_processor()
        batch_size = self._settings.translation_batch_size

        results: list[str] = []
        for start in range(0, len(sentences), batch_size):
            chunk = sentences[start : start + batch_size]
            prepared = processor.preprocess_batch(chunk, src_lang=src, tgt_lang=tgt)
            inputs = tokenizer(
                prepared,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=self._settings.translation_beams,
                    num_return_sequences=1,
                )
            decoded = tokenizer.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            restored = processor.postprocess_batch(decoded, lang=tgt)
            if tgt.endswith("_Mtei"):
                restored = [_MEETEI_VIRAMA_SPACES.sub(_MEETEI_VIRAMA, text) for text in restored]
            results.extend(restored)
        return results

    def _translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        """Translate whole texts, preserving paragraph breaks.

        IndicTrans2 is trained on sentences, so each text is split into
        paragraphs and sentences, everything is translated in one batched pass,
        and the pieces are stitched back together.
        """
        self._check_enabled()
        direction = self._from_english if src == ENGLISH_FLORES else self._to_english

        # layout[i] = list of paragraphs, each a list of indices into `flat`.
        flat: list[str] = []
        layout: list[list[list[int]]] = []
        for text in texts:
            paragraphs: list[list[int]] = []
            for paragraph in text.split("\n"):
                indices = []
                for sentence in _split_sentences(paragraph):
                    indices.append(len(flat))
                    flat.append(sentence)
                paragraphs.append(indices)
            layout.append(paragraphs)

        if not flat:
            return list(texts)

        try:
            translated = self._translate_sentences(direction, flat, src, tgt)
        except TranslationError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any model failure uniformly
            logger.exception("Translation %s -> %s failed", src, tgt)
            raise TranslationError(f"Translation {src} → {tgt} failed: {exc}") from exc

        return [
            "\n".join(" ".join(translated[i] for i in paragraph) for paragraph in paragraphs)
            for paragraphs in layout
        ]

    # -- public API -------------------------------------------------------------

    def to_english(self, texts: list[str], source: str) -> list[str]:
        """Translate ``texts`` from the FLORES tag ``source`` into English."""
        return self._translate(texts, source, ENGLISH_FLORES)

    def from_english(self, texts: list[str], target: str) -> list[str]:
        """Translate English ``texts`` into the FLORES tag ``target``."""
        return self._translate(texts, ENGLISH_FLORES, target)

    def warm_up(self) -> None:
        """Load both checkpoints ahead of the first request."""
        self._check_enabled()
        device = self._resolve_device()
        self._to_english.load(device)
        self._from_english.load(device)
        self._get_processor()


@lru_cache(maxsize=1)
def get_translator() -> IndicTranslator:
    return IndicTranslator(get_settings())
