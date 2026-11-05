from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from services.images import PreparedImage
from services.openrouter_client import OpenRouterClient, parse_json_object
from services.specialists import format_km2

BOX_COLOUR = (255, 214, 0)
ANNOTATED_EDGE_PX = 1024
RELATIVE_TOLERANCE = 0.1
ABSOLUTE_TOLERANCE = 0.6
MAX_SUMMARY_CHARS = 1200
MAX_REGION_CHARS = 400

_NUMBER_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)?")
_WORD_RE = re.compile(r"[a-z]+")
_PERCENT_RE = re.compile(r"([0-9]+(?:[.,][0-9]+)?)\s*(%|percent\b|per cent\b)")
_DIRECTIONS = ("north", "south", "east", "west")

_SIDE_WORDS = {
    "left": "left", "leftmost": "left",
    "right": "right", "rightmost": "right",
    "top": "top", "upper": "top", "topmost": "top",
    "bottom": "bottom", "lower": "bottom", "bottommost": "bottom",
}
_COMPASS_WORDS = {
    "north", "south", "east", "west", "northern", "southern", "eastern", "western",
    "northeast", "northwest", "southeast", "southwest",
    "northeastern", "northwestern", "southeastern", "southwestern",
}
_JARGON = (
    "bounding box", "detection", "threshold", "confidence score", "probability",
    "pixel", "model output", "candidate", "re-ranker", "reranker", "iou", "logit",
)
_NUMBER_WORDS = {
    "zero": 0, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20,
    "dozen": 12, "hundred": 100,
}

SYSTEM_PROMPT = """\
You explain satellite change-detection results in plain language for people with \
no technical background: farmers, local officials, relief workers, journalists.

A fine-tuned change-detection model has already compared two images of the same \
place taken at different times. It found the changed regions, drawn as numbered \
yellow boxes on both images, and measured them. Your only job is to put the \
model's findings into simple words.

Rules -- follow all of them:
1. Describe only the numbered regions listed under FACTS. Never mention, number or \
describe any other area, even if you think something changed there.
2. Every number you write must come from FACTS, optionally rounded. Do not \
calculate new numbers, counts, dates, distances or percentages. Words like "about \
a third" or "most of the image" are fine.
3. For locations use only the direction given for each region in FACTS (for \
example "in the south"). Do not add other directions, and do not name the place: \
you are not told where it is.
4. Look closely inside each numbered box, first in the earlier image and then in \
the later one, and say in everyday words what the ground looks like in each: for \
example dry sand, sandbanks, river water, muddy floodwater, green crops, bare \
fields, forest, houses. Be specific to what you see in that box; do not just \
repeat the model's description. If what you see clearly does not match the \
model's description, say what you see and that the change there is uncertain. Do \
not guess causes you cannot see.
5. Region sentences must not repeat the region's size or location: the app shows \
those next to your sentence.
6. Plain language: short sentences, no jargon. Never use the words mask, pixel, \
resolution, confidence score, bi-temporal, model label, segmentation or GSD.
7. In the summary, answer the user's question first if the facts allow, then say \
what changed overall, where, and how much, in words a non-expert would use.

Respond with a single JSON object and nothing else:
{"summary": "<2 to 4 short sentences>",
 "regions": [{"number": <a region number from FACTS>,
              "before": "<what the ground in the box looks like in the earlier image, a few words>",
              "after": "<what it looks like in the later image, a few words>",
              "description": "<one short sentence on what changed there, in plain words>"}]}"""

GROUNDING_SYSTEM_PROMPT = """You explain the result of an object-finding model on a satellite or aerial image, in \
plain language, for people with no technical background: farmers, local officials, \
relief workers, journalists.

A fine-tuned model was asked to find what the user described. It marked the area it \
thinks matches best as yellow box 1, and possibly other, less likely matches as boxes \
2, 3 and so on. Your only job is to tell the user, in simple words, what it found.

Rules -- follow all of them:
1. Talk only about the numbered boxes listed under FACTS. Never mention, count or \
describe any other object or area as something the model found.
2. Look inside each numbered box and say in everyday words what it shows, for example \
"a large white ship", "a football field", "a row of parked cars". If box 1 does not \
look like what the user asked for, say plainly that the match may be wrong. Never say \
something is in a box if you cannot see it there.
3. Every number you write must come from FACTS. Do not count objects, and do not add \
sizes, distances or percentages of your own. Words like "small" or "most of the \
picture" are fine.
4. For locations use only the position given for each box in FACTS (left, right, top, \
bottom, centre). Never use compass directions (north, south, east, west): which way \
the image faces is unknown. Do not name the place.
5. Say how sure the model is using the words given in FACTS ("confident", "fairly \
confident", "not very confident").
6. Plain language, short sentences. Never use these words: bounding box, detection, \
threshold, score, confidence score, probability, pixel, model output, candidate, \
re-ranker, IoU, logit.

Respond with a single JSON object and nothing else:
{"summary": "<2 or 3 short sentences: what was found for the user's request, where, and how sure the model is>",
 "boxes": [{"number": <a box number from FACTS>,
            "description": "<one short sentence on what this box shows, in plain words>"}]}"""


class NarrationRejected(ValueError):
    pass


@dataclass
class Narration:
    summary: str
    regions: dict[int, str] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)


def _decode(image: PreparedImage) -> Image.Image:
    raw = base64.b64decode(image.data_uri.split(",", 1)[1])
    return Image.open(io.BytesIO(raw)).convert("RGB")


def encode_numpy_image(img_array: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", img_array)
    if not success:
        raise ValueError("Could not encode numpy array to PNG.")
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def annotate(image: PreparedImage, regions: list[dict[str, Any]]) -> str:
    canvas = _decode(image)
    scale = min(1.0, ANNOTATED_EDGE_PX / max(canvas.size))
    if scale < 1.0:
        canvas = canvas.resize((round(canvas.width * scale), round(canvas.height * scale)))
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)
    line = max(2, round(max(width, height) / 300))
    font = _font(max(14, round(max(width, height) / 40)))
    pad = line + 1

    for region in regions:
        x1, y1, x2, y2 = region["box"]
        box = (x1 * width, y1 * height, x2 * width, y2 * height)
        draw.rectangle(box, outline=BOX_COLOUR, width=line)
        tag = str(region["number"])
        left, top, right, bottom = draw.textbbox((0, 0), tag, font=font)
        draw.rectangle(
            (box[0], box[1], box[0] + right - left + 2 * pad, box[1] + bottom - top + 2 * pad),
            fill=BOX_COLOUR,
        )
        draw.text((box[0] + pad - left, box[1] + pad - top), tag, fill=(0, 0, 0), font=font)

    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _direction(sector: str) -> str:
    first = sector.removeprefix("the ").split()[0]
    return "centre" if first == "central" else first


def facts_text(query: str, facts: dict[str, Any]) -> str:
    regions = facts["regions"]
    changed = f"{facts['changed_share']:.1%} of the image"
    if facts["changed_km2"] is not None:
        changed += f" (about {format_km2(facts['changed_km2'])})"

    lines = [
        f"User's question: {query}\n\nThe first image is the earlier one (before), the second the later one (after).\n\nFACTS",
        f'- The model\'s overall description of the change: "{facts["model_answer"]}".',
        f"- Changed ground in total: {changed}.",
    ]
    if facts["total_regions"] > len(regions):
        lines.append(
            f"- The model found {facts['total_regions']} separate changed patches; the {len(regions)} "
            "largest are numbered on the images. Only these numbered ones may be described."
        )
    for r in regions:
        size = f"{r['share']:.1%} of the image" + (
            f" (about {format_km2(r['area_km2'])})" if r["area_km2"] is not None else ""
        )
        lines.append(
            f'- Region {r["number"]}: the model describes it as "{r["label"]}"; '
            f"size {size}; location: {_direction(r['sector'])}."
        )
    return "\n".join(lines)


def _allowed_numbers(facts: dict[str, Any]) -> list[float]:
    allowed = [facts["changed_share"] * 100, float(facts["total_regions"]), float(len(facts["regions"]))]
    if facts["changed_km2"] is not None:
        allowed.append(facts["changed_km2"])
    for r in facts["regions"]:
        allowed.extend([float(r["number"]), r["share"] * 100])
        if r["area_km2"] is not None:
            allowed.append(r["area_km2"])
    return allowed


def _unsupported_numbers(text: str, allowed: list[float]) -> list[str]:
    return [
        token for token in _NUMBER_RE.findall(text)
        if not any(
            abs(float(token.replace(",", ".")) - fact) <= max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * fact)
            for fact in allowed
        )
    ]


def _directions(text: str) -> set[str]:
    found: set[str] = set()
    for word in _WORD_RE.findall(text.lower()):
        rest = word.removesuffix("ern")
        named = set()
        for d in _DIRECTIONS:
            if rest.startswith(d):
                named.add(d)
                rest = rest[len(d):]
        if not rest:
            found |= named
    return found


def _sector_directions(sector: str) -> set[str]:
    return {d for d in _DIRECTIONS if d in sector}


def _validate_summary(parsed: dict[str, Any]) -> str:
    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise NarrationRejected("no summary was returned")
    summary = summary.strip()
    if len(summary) > MAX_SUMMARY_CHARS:
        raise NarrationRejected("the summary was too long")
    return summary


def check(parsed: dict[str, Any], facts: dict[str, Any]) -> Narration:
    summary = _validate_summary(parsed)
    by_number = {r["number"]: r for r in facts["regions"]}
    allowed = _allowed_numbers(facts)
    every_direction = set().union(*(_sector_directions(r["sector"]) for r in facts["regions"]))

    if bad := _unsupported_numbers(summary, allowed):
        raise NarrationRejected(f"the summary contains figures the model did not measure: {', '.join(bad)}")
    if extra := _directions(summary) - every_direction:
        raise NarrationRejected(f"the summary names directions with no changed region: {', '.join(sorted(extra))}")

    raw_regions = parsed.get("regions") or []
    if not isinstance(raw_regions, list):
        raise NarrationRejected("the region descriptions were malformed")

    regions: dict[int, str] = {}
    for item in raw_regions:
        if not isinstance(item, dict):
            raise NarrationRejected("the region descriptions were malformed")
        try:
            number = int(item.get("number"))
        except (TypeError, ValueError):
            raise NarrationRejected("a region description had no valid region number") from None
        if number not in by_number:
            raise NarrationRejected(f"it described region {number}, which the model did not find")

        parts = [item.get(k) for k in ("description", "before", "after")]
        if not all(isinstance(p, str) and p.strip() for p in parts):
            continue
        sentence, before, after = (p.strip().rstrip(".") for p in parts)
        text = f"{sentence}. Before: {before}. After: {after}."
        if len(text) > MAX_REGION_CHARS:
            raise NarrationRejected(f"the description of region {number} was too long")
        if bad := _unsupported_numbers(text, allowed):
            raise NarrationRejected(
                f"region {number}'s description contains figures the model did not measure: {', '.join(bad)}"
            )
        if extra := _directions(text) - _sector_directions(by_number[number]["sector"]):
            raise NarrationRejected(
                f"region {number} was placed in the {', '.join(sorted(extra))}, "
                f"but the model located it in {by_number[number]['sector']}"
            )
        regions[number] = text
    return Narration(summary=summary, regions=regions)


async def narrate(
    client: OpenRouterClient,
    *,
    model: str,
    query: str,
    images: list[PreparedImage],
    facts: dict[str, Any],
) -> Narration:
    regions = facts["regions"]
    is_grounding = facts.get("kind") == "grounding"
    text, usage = await client.complete(
        model=model,
        system_prompt=GROUNDING_SYSTEM_PROMPT if is_grounding else SYSTEM_PROMPT,
        user_text=grounding_facts_text(query, facts) if is_grounding else facts_text(query, facts),
        image_data_uris=[annotate(images[0], regions)] if is_grounding else [annotate(images[0], regions), annotate(images[1], regions)],
        json_object=True,
        temperature=0.2,
        max_tokens=700 if is_grounding else 900,
    )
    narration = (check_grounding if is_grounding else check)(parse_json_object(text), facts)
    narration.usage = usage
    return narration


def grounding_facts_text(query: str, facts: dict[str, Any]) -> str:
    lines = [
        f"User's request: {query}\n\nFACTS",
        f"- The model marked {len(facts['regions'])} numbered box(es) on the image.",
    ]
    for r in facts["regions"]:
        role = "the best match" if r["number"] == 1 else "a less likely match"
        lines.append(
            f"- Box {r['number']}: {role}; position: {r['position']} of the image; "
            f"covers {r['size']}; the model is {r['certainty']} about it."
        )
    return "\n".join(lines)


def _sides(text: str) -> set[str]:
    return {_SIDE_WORDS[w] for w in _WORD_RE.findall(text.lower()) if w in _SIDE_WORDS}


def _position_sides(position: str) -> set[str]:
    return {p for p in position.split("-") if p in {"left", "right", "top", "bottom"}}


def _grounding_numbers(facts: dict[str, Any]) -> tuple[list[float], list[float]]:
    counts = [float(len(facts["regions"]))] + [float(r["number"]) for r in facts["regions"]]
    percentages = [val for r in facts["regions"] for val in (r["share"] * 100, r["confidence"] * 100)]
    return counts, percentages


def _check_grounding_text(
    text: str, allowed: tuple[list[float], list[float]], sides: set[str], where: str,
) -> None:
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)
    counts, percentages = allowed

    bad = _unsupported_numbers(" ".join(m.group(1) for m in _PERCENT_RE.finditer(lowered)), percentages)
    bad += _unsupported_numbers(_PERCENT_RE.sub(" ", lowered), counts)
    bad += [
        w for w in words
        if (val := _NUMBER_WORDS.get(w)) is not None
        and not any(abs(val - f) <= ABSOLUTE_TOLERANCE for f in counts)
    ]
    if bad:
        raise NarrationRejected(f"{where} contains figures the model did not produce: {', '.join(bad)}")
    if compass := sorted(set(words) & _COMPASS_WORDS):
        raise NarrationRejected(
            f"{where} uses compass directions ({', '.join(compass)}), but the image's orientation is unknown"
        )
    if extra := _sides(text) - sides:
        raise NarrationRejected(
            f"{where} places something on the {', '.join(sorted(extra))} of the image, where the model marked no box"
        )
    if jargon := [term for term in _JARGON if re.search(rf"\b{re.escape(term)}\b", lowered)]:
        raise NarrationRejected(f"{where} uses technical terms: {', '.join(jargon)}")


def check_grounding(parsed: dict[str, Any], facts: dict[str, Any]) -> Narration:
    summary = _validate_summary(parsed)
    by_number = {r["number"]: r for r in facts["regions"]}
    allowed = _grounding_numbers(facts)
    all_sides = set().union(*(_position_sides(r["position"]) for r in facts["regions"]))
    _check_grounding_text(summary, allowed, all_sides, "the summary")

    raw_boxes = parsed.get("boxes") or []
    if not isinstance(raw_boxes, list):
        raise NarrationRejected("the box descriptions were malformed")

    boxes: dict[int, str] = {}
    for item in raw_boxes:
        if not isinstance(item, dict):
            raise NarrationRejected("the box descriptions were malformed")
        try:
            number = int(item.get("number"))
        except (TypeError, ValueError):
            raise NarrationRejected("a box description had no valid box number") from None
        if number not in by_number:
            raise NarrationRejected(f"it described box {number}, which the model did not mark")

        desc = item.get("description")
        if not isinstance(desc, str) or not desc.strip():
            continue
        text = desc.strip().rstrip(".")
        text = f"{text[:1].upper()}{text[1:]}."
        if len(text) > MAX_REGION_CHARS:
            raise NarrationRejected(f"the description of box {number} was too long")
        _check_grounding_text(
            text, allowed, _position_sides(by_number[number]["position"]), f"box {number}'s description"
        )
        boxes[number] = text
    return Narration(summary=summary, regions=boxes)