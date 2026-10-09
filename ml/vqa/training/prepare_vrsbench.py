"""
Prepare VRSBench's single-image VQA data for the VQA specialist.

VRSBench_train.json holds three kinds of turns per image — captions
(`[caption]`), referring expressions (`[refer]`) and questions (`[vqa]`). Only
the `[vqa]` turns are VQA data; the other two belong to captioning and
grounding. VRSBench_EVAL_vqa.json is the official VQA evaluation set (37,409
questions, each labelled with a question type).

Writes to --output (default data/vrsbench_vqa):
    train.jsonl   VQA pairs from 95% of the training images
    dev.jsonl     VQA pairs from the other 5% — used to pick the checkpoint,
                  so the evaluation set plays no part in any choice
    test.jsonl    the official evaluation set, with its question types
    answers.json  answer vocabulary: the most frequent training answers
    questions.json question-word vocabulary from the training questions

Each JSONL line: {"image", "question", "answer", "type", "modality"}; image
paths are relative to --output. Images are the VRSBench tiles already
extracted for grounding (data/vrsbench/extracted_images); nothing is copied.

    python -m ml.vqa.training.prepare_vrsbench
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from ml.vqa.model import SimpleTokenizer

HF_REPO = "xiang709/VRSBench"

# "<image>\n[vqa] What is ...?. A short answer to the question is"
_PROMPT_PREFIX = re.compile(r"^\s*(<image>\s*)?\[vqa\]\s*", re.IGNORECASE)
_PROMPT_SUFFIX = re.compile(r"\.?\s*A short answer to the question is\s*$", re.IGNORECASE)


def normalize_answer(text: str) -> str:
    """Lower-case, collapse whitespace, drop a trailing full stop."""
    return " ".join(str(text).strip().lower().split()).rstrip(".").strip()


# Instruction templates wrapped around ~70% of the training questions (the
# evaluation questions are bare). Removed so training and evaluation see the
# same kind of question.
_TEMPLATE_PREFIXES = (
    "What is the answer to the following question",
    "Based on the image, respond to this question with a short answer:",
    "Given the image, answer the following question with no more than three words.",
    "Use the provided image to answer the question:",
    "The question",
    "Question:",
    "Q:",
)
_TEMPLATE_SUFFIXES = (
    "can be answered using the image. A short answer is",
    "Provide your answer as short as possible",
    "Short answer:",
    "A:",
)


def clean_question(prompt: str) -> str:
    question = _PROMPT_SUFFIX.sub("", _PROMPT_PREFIX.sub("", prompt)).strip()
    changed = True
    while changed:
        changed = False
        question = question.rstrip(".").strip()
        for prefix in _TEMPLATE_PREFIXES:
            if question.startswith(prefix):
                question, changed = question[len(prefix):].strip(), True
        for suffix in _TEMPLATE_SUFFIXES:
            if question.endswith(suffix):
                question, changed = question[: -len(suffix)].strip(), True
    return question


def question_words(text: str) -> list[str]:
    """Split exactly as the model's tokenizer will at inference time."""
    return SimpleTokenizer().tokenize(text)


def _index_images(root: Path) -> dict[str, Path]:
    return {p.name: p for p in root.rglob("*.png")}


def _is_dev(image_name: str, fraction: float) -> bool:
    return int(hashlib.md5(image_name.encode()).hexdigest(), 16) % 10_000 < fraction * 10_000


def prepare(
    output: str = "data/vrsbench_vqa",
    train_images: str = "data/vrsbench/extracted_images/train",
    eval_images: str = "data/vrsbench/extracted_images/validation",
    cache_dir: str = "data/vrsbench",
    num_answers: int = 1000,
    min_word_count: int = 3,
    max_words: int = 2000,
    dev_fraction: float = 0.05,
) -> None:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    train_index = _index_images(Path(train_images))
    eval_index = _index_images(Path(eval_images))
    if not train_index or not eval_index:
        raise FileNotFoundError(
            f"No VRSBench images under {train_images} / {eval_images}. Extract them first "
            "(running any ml.grounding command downloads and extracts them)."
        )

    def rel(path: Path) -> str:
        return Path(os.path.relpath(path, out)).as_posix()

    # ── Training questions ──
    train_json = hf_hub_download(HF_REPO, "VRSBench_train.json", repo_type="dataset",
                                 cache_dir=cache_dir)
    with open(train_json, encoding="utf-8") as f:
        items = json.load(f)

    splits: dict[str, list[dict]] = {"train": [], "dev": []}
    missing = 0
    for item in items:
        image = item.get("image", "")
        path = train_index.get(image)
        convo = item.get("conversations", [])
        for human, gpt in zip(convo[::2], convo[1::2]):
            if "[vqa]" not in human.get("value", ""):
                continue
            if path is None:
                missing += 1
                continue
            record = {
                "image": rel(path),
                "question": clean_question(human["value"]),
                "answer": normalize_answer(gpt["value"]),
                "type": None,     # the training file has no question types
                "modality": "optical",
            }
            splits["dev" if _is_dev(image, dev_fraction) else "train"].append(record)

    # ── Official evaluation set ──
    eval_json = hf_hub_download(HF_REPO, "VRSBench_EVAL_vqa.json", repo_type="dataset",
                                cache_dir=cache_dir)
    with open(eval_json, encoding="utf-8") as f:
        eval_items = json.load(f)
    test = []
    for entry in eval_items:
        path = eval_index.get(entry["image_id"])
        if path is None:
            missing += 1
            continue
        test.append({
            "image": rel(path),
            "question": entry["question"].strip(),
            "answer": normalize_answer(entry["ground_truth"]),
            "type": entry.get("type"),
            "modality": "optical",
        })

    # ── Vocabularies, from the training split only ──
    answer_counts = Counter(r["answer"] for r in splits["train"])
    answers = [a for a, _ in answer_counts.most_common(num_answers)]
    word_counts = Counter(w for r in splits["train"] for w in question_words(r["question"]))
    words = [w for w, c in word_counts.most_common(max_words) if c >= min_word_count]

    for name, records in (("train", splits["train"]), ("dev", splits["dev"]), ("test", test)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    (out / "answers.json").write_text(json.dumps(answers, indent=0), encoding="utf-8")
    (out / "questions.json").write_text(json.dumps(words, indent=0), encoding="utf-8")

    def coverage(records: list[dict]) -> float:
        known = set(answers)
        return sum(r["answer"] in known for r in records) / max(len(records), 1)

    print(f"  train {len(splits['train']):,}  dev {len(splits['dev']):,}  test {len(test):,}"
          + (f"  ({missing} pairs skipped: image not found)" if missing else ""))
    print(f"  answer vocabulary: {len(answers)} answers; covers "
          f"train {coverage(splits['train']):.1%}, dev {coverage(splits['dev']):.1%}, "
          f"test {coverage(test):.1%}")
    print(f"  question vocabulary: {len(words)} words (seen >= {min_word_count} times)")
    print(f"  written to {out.resolve()}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Prepare VRSBench single-image VQA data")
    p.add_argument("--output", default="data/vrsbench_vqa")
    p.add_argument("--train-images", default="data/vrsbench/extracted_images/train")
    p.add_argument("--eval-images", default="data/vrsbench/extracted_images/validation")
    p.add_argument("--num-answers", type=int, default=1000)
    args = p.parse_args(argv)
    prepare(args.output, args.train_images, args.eval_images, num_answers=args.num_answers)


if __name__ == "__main__":
    main()
