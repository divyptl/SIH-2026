"""Fill the frontend's UI translation files using the same IndicTrans2 models.

``frontend/src/locales/en.json`` is the source of truth. By default only keys
missing from a locale are translated, so hand-reviewed strings are never
overwritten; pass ``--force`` to retranslate everything for the given languages.

    cd backend
    uv run python -m scripts.translate_locales            # every language, missing keys only
    uv run python -m scripts.translate_locales hi ta      # just Hindi and Tamil
    uv run python -m scripts.translate_locales sat --force

Interpolation placeholders (``{{size}}``) and inline tags (``<code>…</code>``)
are kept out of the model: the text around them is translated piecewise.

Every output is checked before it is written. A string is rejected — left out of
the locale, so the UI shows English for it and the next run retries it — when
the model leaks letters from another script, or mangles a technical term that
must stay verbatim (GeoTIFF, .tif/.tiff, SAR, VQA). The distilled model does
both for low-resource languages such as Santali and Manipuri.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.translation import ENGLISH, LANGUAGES, get_translator  # noqa: E402

LOCALES_DIR = Path(__file__).resolve().parents[2] / "frontend" / "src" / "locales"
_PROTECTED = re.compile(r"(\{\{\s*\w+\s*\}\}|<[^>]+>)")
# The model ends almost every output as a sentence; UI labels ("Language",
# "Reset") must not gain a full stop, danda (।॥), Urdu stop (۔), Ol Chiki
# mucaad (᱾) or Meetei cheikhei (꯫).
_TRAILING_STOP = re.compile("[.।॥۔᱾꯫]+$")
_ENDS_WITH_PUNCTUATION = re.compile(r"[.!?…:]$")
# Terms that must survive translation character for character.
_VERBATIM = re.compile(r"GeoTIFF|\.tif/\.tiff|SAR|VQA")
# Unicode character-name prefix of each script's letters.
_SCRIPT_NAMES = {
    "Deva": "DEVANAGARI",
    "Beng": "BENGALI",
    "Guru": "GURMUKHI",
    "Gujr": "GUJARATI",
    "Orya": "ORIYA",
    "Taml": "TAMIL",
    "Telu": "TELUGU",
    "Knda": "KANNADA",
    "Mlym": "MALAYALAM",
    "Arab": "ARABIC",
    "Olck": "OL CHIKI",
    "Mtei": "MEETEI",
}


def _flatten(tree: dict, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _unflatten(flat: dict[str, str]) -> dict:
    tree: dict = {}
    for path, value in flat.items():
        node = tree
        *parents, leaf = path.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return tree


def _order_like(source: dict, target: dict) -> dict:
    """Keep the locale's keys in the same order as en.json for readable diffs."""
    ordered = {}
    for key, value in source.items():
        if key in target:
            ordered[key] = _order_like(value, target[key]) if isinstance(value, dict) else target[key]
    return ordered


def _tidy(fragment: str, output: str) -> str:
    """Match the source's end punctuation: labels get none, "Label:" keeps its colon."""
    if fragment.endswith(":"):
        # Devanagari output often writes the colon as a visarga (ः).
        return _TRAILING_STOP.sub("", output).rstrip(" :ः") + ":"
    if _ENDS_WITH_PUNCTUATION.search(fragment):
        return output
    return _TRAILING_STOP.sub("", output).rstrip()


def _problem(english: str, translated: str, script: str) -> str | None:
    """Why ``translated`` is unusable, or None if it looks sound."""
    for term in _VERBATIM.findall(english):
        if term not in translated:
            return f"mangled '{term}'"
    expected = _SCRIPT_NAMES[script]
    foreign = {
        unicodedata.name(char, "?").split()[0]
        for char in translated
        if ord(char) > 0x2FF
        and unicodedata.category(char)[0] in "LM"
        and not unicodedata.name(char, "").startswith(expected)
    }
    if foreign:
        return f"contains {', '.join(sorted(foreign))} script"
    return None


def translate_locale(code: str, force: bool) -> tuple[int, dict[str, str]]:
    """Returns (strings written, {rejected key: reason})."""
    language = LANGUAGES[code]
    script = language.flores.split("_")[1]
    source = json.loads((LOCALES_DIR / "en.json").read_text(encoding="utf-8"))
    path = LOCALES_DIR / f"{code}.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    source_flat = _flatten(source)
    target_flat = {} if force else _flatten(existing)
    todo = [key for key in source_flat if key not in target_flat]
    if not todo:
        return 0, {}

    # Split every string into protected tokens and translatable fragments.
    pieces: dict[str, list[str]] = {key: _PROTECTED.split(source_flat[key]) for key in todo}
    fragments = sorted(
        {
            part.strip()
            for parts in pieces.values()
            for part in parts
            if part.strip()
            and not _PROTECTED.fullmatch(part)
            and not _VERBATIM.fullmatch(part.strip())
            and re.search(r"[A-Za-z]", part)
        }
    )
    outputs = get_translator().from_english(fragments, language.flores)
    translated = {
        fragment: _tidy(fragment, output) for fragment, output in zip(fragments, outputs)
    }

    rejected: dict[str, str] = {}
    for key, parts in pieces.items():
        out = []
        for part in parts:
            core = part.strip()
            if core in translated:
                lead = part[: len(part) - len(part.lstrip())]
                trail = part[len(part.rstrip()) :]
                out.append(f"{lead}{translated[core]}{trail}")
            else:
                out.append(part)
        text = "".join(out)
        problem = _problem(source_flat[key], text, script)
        if problem:
            rejected[key] = problem
        else:
            target_flat[key] = text

    merged = _order_like(source, _unflatten(target_flat))
    # LF endings to match the rest of the frontend (and Prettier) on Windows too.
    path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return len(todo) - len(rejected), rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("languages", nargs="*", help="Language codes (default: all).")
    parser.add_argument("--force", action="store_true", help="Retranslate existing keys too.")
    args = parser.parse_args()

    codes = args.languages or [code for code in LANGUAGES if code != ENGLISH]
    unknown = [code for code in codes if code not in LANGUAGES or code == ENGLISH]
    if unknown:
        parser.error(f"Unknown language code(s): {', '.join(unknown)}")

    for code in codes:
        written, rejected = translate_locale(code, args.force)
        print(f"{code}: {written} string(s) translated, {len(rejected)} rejected")
        for key, reason in rejected.items():
            print(f"    {key}: {reason} (falls back to English)")


if __name__ == "__main__":
    main()
