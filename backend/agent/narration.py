"""Plain-language narration of a Change-VQA result.

The fine-tuned change model produces every fact: which regions changed, how
much, where, and what the change is. Its own wording ("bare ground to water",
"33.5% of the scene, in the south sector") is exact but unreadable for the
people this app is for, so a VLM rewords it.

The VLM is shown both images with the model's regions drawn on them as numbered
boxes, and the model's measurements as the only facts it may use. Its reply is
then checked, and discarded in favour of the model's own text, if it
- describes a region number the model did not report,
- writes a number that is not one of the model's (after rounding), or
- places something in a direction the model did not report.
It can still choose words the model would not ("sandbank", "river"): that is
the point of the step, and it only ever applies them to the model's regions.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from services.images import PreparedImage
from services.openrouter_client import OpenRouterClient, parse_json_object
from services.specialists import format_km2

# Box colour on the annotated images: stands out on water, vegetation and sand.
BOX_COLOUR = (255, 214, 0)
# Longest edge of the annotated images sent to the VLM.
ANNOTATED_EDGE_PX = 1024

# A written number matches a fact when it is within 10% of it, or 0.6 absolute,
# which accepts rounding ("70%" for 69.91%) and nothing else.
RELATIVE_TOLERANCE = 0.1
ABSOLUTE_TOLERANCE = 0.6

MAX_SUMMARY_CHARS = 1200
MAX_REGION_CHARS = 400

_NUMBER_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)?")
_WORD_RE = re.compile(r"[a-z]+")
_DIRECTIONS = ("north", "south", "east", "west")

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


class NarrationRejected(ValueError):
    """The VLM's wording went beyond the model's facts, so it is not shown."""


@dataclass
class Narration:
    summary: str
    # Region number -> the VLM's sentence about it.
    regions: dict[int, str] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)


def _decode(image: PreparedImage) -> Image.Image:
    raw = base64.b64decode(image.data_uri.split(",", 1)[1])
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has a single fixed-size default font
        return ImageFont.load_default()


def annotate(image: PreparedImage, regions: list[dict[str, Any]]) -> str:
    """The image with the model's regions drawn as numbered boxes, as a JPEG data URI."""
    canvas = _decode(image)
    scale = min(1.0, ANNOTATED_EDGE_PX / max(canvas.size))
    if scale < 1.0:
        canvas = canvas.resize((round(canvas.width * scale), round(canvas.height * scale)))
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)
    line = max(2, round(max(width, height) / 300))
    font = _font(max(14, round(max(width, height) / 40)))

    for region in regions:
        x1, y1, x2, y2 = region["box"]
        box = (x1 * width, y1 * height, x2 * width, y2 * height)
        draw.rectangle(box, outline=BOX_COLOUR, width=line)
        tag = str(region["number"])
        left, top, right, bottom = draw.textbbox((0, 0), tag, font=font)
        pad = line + 1
        tag_box = (box[0], box[1], box[0] + right - left + 2 * pad, box[1] + bottom - top + 2 * pad)
        draw.rectangle(tag_box, fill=BOX_COLOUR)
        draw.text((box[0] + pad - left, box[1] + pad - top), tag, fill=(0, 0, 0), font=font)

    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _direction(sector: str) -> str:
    """'the north-west quadrant' -> 'north-west'; 'the central sector' -> 'centre'."""
    words = sector.removeprefix("the ").split()
    return "centre" if words[0] == "central" else words[0]


def facts_text(query: str, facts: dict[str, Any]) -> str:
    """The model's findings as the prompt's FACTS block."""
    regions = facts["regions"]
    lines = [
        f"User's question: {query}",
        "",
        "The first image is the earlier one (before), the second the later one (after).",
        "",
        "FACTS",
        f"- The model's overall description of the change: \"{facts['model_answer']}\".",
    ]
    changed = f"{facts['changed_share']:.1%} of the image"
    if facts["changed_km2"] is not None:
        changed += f" (about {format_km2(facts['changed_km2'])})"
    lines.append(f"- Changed ground in total: {changed}.")
    total = facts["total_regions"]
    if total > len(regions):
        lines.append(
            f"- The model found {total} separate changed patches; the {len(regions)} "
            "largest are numbered on the images. Only these numbered ones may be described."
        )
    for region in regions:
        size = f"{region['share']:.1%} of the image"
        if region["area_km2"] is not None:
            size += f" (about {format_km2(region['area_km2'])})"
        lines.append(
            f"- Region {region['number']}: the model describes it as \"{region['label']}\"; "
            f"size {size}; location: {_direction(region['sector'])}."
        )
    return "\n".join(lines)


def _allowed_numbers(facts: dict[str, Any]) -> list[float]:
    """Every number the narration may contain, in the units a reader would see."""
    allowed = [
        facts["changed_share"] * 100,
        float(facts["total_regions"]),
        float(len(facts["regions"])),
    ]
    if facts["changed_km2"] is not None:
        allowed.append(facts["changed_km2"])
    for region in facts["regions"]:
        allowed += [float(region["number"]), region["share"] * 100]
        if region["area_km2"] is not None:
            allowed.append(region["area_km2"])
    return allowed


def _unsupported_numbers(text: str, allowed: list[float]) -> list[str]:
    bad = []
    for token in _NUMBER_RE.findall(text):
        value = float(token.replace(",", "."))
        if not any(
            abs(value - fact) <= max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * fact)
            for fact in allowed
        ):
            bad.append(token)
    return bad


def _directions(text: str) -> set[str]:
    """Compass directions named in ``text``: 'north-eastern' -> {north, east}."""
    found: set[str] = set()
    for word in _WORD_RE.findall(text.lower()):
        rest = word.removesuffix("ern")
        named = set()
        for direction in _DIRECTIONS:
            if rest.startswith(direction):
                named.add(direction)
                rest = rest[len(direction):]
        # Only whole direction words count ("least" is not "east").
        if not rest:
            found |= named
    return found


def _sector_directions(sector: str) -> set[str]:
    return {d for d in _DIRECTIONS if d in sector}


def check(parsed: dict[str, Any], facts: dict[str, Any]) -> Narration:
    """Accept the VLM's reply only if it stays within the model's facts."""
    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise NarrationRejected("no summary was returned")
    summary = summary.strip()
    if len(summary) > MAX_SUMMARY_CHARS:
        raise NarrationRejected("the summary was too long")

    by_number = {region["number"]: region for region in facts["regions"]}
    allowed = _allowed_numbers(facts)
    every_direction = set().union(*(_sector_directions(r["sector"]) for r in facts["regions"]))

    bad = _unsupported_numbers(summary, allowed)
    if bad:
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
        parts = [item.get(key) for key in ("description", "before", "after")]
        if not all(isinstance(part, str) and part.strip() for part in parts):
            continue
        sentence, before, after = (part.strip().rstrip(".") for part in parts)
        text = f"{sentence}. Before: {before}. After: {after}."
        if len(text) > MAX_REGION_CHARS:
            raise NarrationRejected(f"the description of region {number} was too long")
        bad = _unsupported_numbers(text, allowed)
        if bad:
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
    """Have the VLM put the model's findings into plain words, then check them.

    Raises NarrationRejected when the reply strays from the facts, and
    OpenRouterError when the call itself fails.
    """
    regions = facts["regions"]
    text, usage = await client.complete(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_text=facts_text(query, facts),
        image_data_uris=[annotate(images[0], regions), annotate(images[1], regions)],
        json_object=True,
        temperature=0.2,
        max_tokens=900,
    )
    narration = check(parse_json_object(text), facts)
    narration.usage = usage
    return narration
