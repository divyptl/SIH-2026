"""
Question-answer generation for Change-VQA training tiles.

Every answer is computed from the tile's change labels, and a question is only
asked when the source dataset actually labels what it asks about: a building
change dataset never answers vegetation questions, and a binary change dataset
never names what an area turned into. (The earlier LEVIR-CD script derived
vegetation answers from building masks, which taught the model label noise.)

Each question type has several paraphrases, one picked per tile, so the model
sees the varied wording real users type instead of one fixed template.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

# Unified land-cover classes every semantic source is mapped onto.
LAND_COVER = ["water", "vegetation", "cropland", "built-up", "bare ground", "other"]

# What a source's labels can support. Binary sources set ``focus`` to the kind
# of change they annotate; semantic sources also provide from/to class maps.
FOCUS_BUILDING = "building"      # binary building change (LEVIR-CD, S2Looking, WHU-CD)
FOCUS_DAMAGE = "damage"          # building damage after a disaster (xView2)
FOCUS_ANY = "any"                # binary change of unspecified type (SYSU-CD, OSCD)
FOCUS_SEMANTIC = "semantic"      # from/to land-cover maps (SECOND, Dynamic World)

# Net share of the tile a class must gain or lose to count as a trend.
TREND_THRESHOLD = 0.005
# Share of the tile a transition must cover to count as "any new water" etc.
PRESENCE_THRESHOLD = 0.002

ANSWERS: list[str] = [
    "yes", "no",
    "increased", "decreased", "unchanged",
    "no change", "minor change", "moderate change", "large scale change",
    "north-west", "north-east", "south-west", "south-east", "center",
    "0", "1", "2", "3", "4", "5 or more",
    "new buildings", "buildings removed", "building damage", "land-cover change",
    *LAND_COVER,
    *(f"{a} to {b}" for a in LAND_COVER for b in LAND_COVER if a != b),
]

PARAPHRASES: dict[str, list[str]] = {
    "any_change": [
        "Has anything changed between the two images?",
        "Is there any change between these two dates?",
        "Did the area change between T1 and T2?",
        "Are there any changes between the images?",
    ],
    "scale": [
        "How significant is the overall change between the two images?",
        "How much of the area has changed?",
        "How large is the change between the two dates?",
    ],
    "where": [
        "Where did the primary change take place?",
        "Where is most of the change located?",
        "In which part of the image did the change happen?",
    ],
    "count": [
        "How many distinct change regions can be identified?",
        "How many changed areas are there?",
        "How many separate regions changed?",
    ],
    "what_changed": [
        "What has changed in these 2 images?",
        "What changed between the two dates?",
        "What is the main change between T1 and T2?",
        "What kind of change occurred?",
    ],
    "became": [
        "What did most of the changed area become?",
        "What land cover replaced the changed area?",
        "What is the changed area now?",
    ],
    "was": [
        "What was most of the changed area before?",
        "What land cover was lost in the changed area?",
        "What was the changed area originally?",
    ],
    "trend_built-up": [
        "Has the built-up area increased, decreased, or remained unchanged?",
        "How has the urban area changed?",
        "Did the built-up area grow or shrink?",
    ],
    "trend_water": [
        "Has the water area increased, decreased, or remained unchanged?",
        "Has the water body expanded or receded?",
        "How has the water extent changed?",
    ],
    "trend_vegetation": [
        "Has the vegetation cover increased, decreased, or remained unchanged?",
        "Has the vegetation increased or decreased?",
        "How has the vegetation cover changed?",
    ],
    "trend_cropland": [
        "Has the cropland area increased, decreased, or remained unchanged?",
        "How has the agricultural land changed?",
    ],
    "trend_bare ground": [
        "Has the bare ground increased, decreased, or remained unchanged?",
        "How has the bare soil area changed?",
    ],
    "new_buildings": [
        "Have new buildings been constructed?",
        "Are there signs of recent construction activity?",
        "Did any new structures appear between the two dates?",
    ],
    "demolished": [
        "Were any buildings demolished?",
        "Have any buildings been removed?",
    ],
    "damage": [
        "Are there damaged buildings?",
        "Were any buildings damaged between the two images?",
        "Is there building damage after the event?",
    ],
    "flooding": [
        "Is there any new water or flooding?",
        "Is there any flooding visible?",
        "Did water spread into new areas?",
    ],
    "vegetation_loss": [
        "Has vegetation been cleared?",
        "Did deforestation or vegetation loss happen in this region?",
        "Was any vegetation removed?",
    ],
}


@dataclass
class ChangeLabels:
    """What a source labels for one tile.

    Attributes:
        change: (H, W) bool, pixels that changed.
        focus: One of the FOCUS_* constants.
        before / after: (H, W) int indices into LAND_COVER for changed pixels
            (semantic sources only); unchanged pixels are ignored.
        gained / lost: Per-class (H, W) bool maps for binary sources that know
            the direction, e.g. S2Looking's new vs demolished buildings.
        labelled_classes: LAND_COVER classes the source annotates.
    """

    change: np.ndarray
    focus: str
    before: np.ndarray | None = None
    after: np.ndarray | None = None
    gained: dict[str, np.ndarray] = field(default_factory=dict)
    lost: dict[str, np.ndarray] = field(default_factory=dict)
    labelled_classes: tuple[str, ...] = ()


def _pick(kind: str, key: str) -> str:
    options = PARAPHRASES[kind]
    return options[zlib.crc32(f"{key}:{kind}".encode()) % len(options)]


def _scale(ratio: float) -> str:
    if ratio < 0.001:
        return "no change"
    if ratio < 0.05:
        return "minor change"
    if ratio < 0.20:
        return "moderate change"
    return "large scale change"


def _where(change: np.ndarray) -> str:
    h, w = change.shape
    mid_h, mid_w = h // 2, w // 2
    quadrants = {
        "north-west": change[:mid_h, :mid_w].sum(),
        "north-east": change[:mid_h, mid_w:].sum(),
        "south-west": change[mid_h:, :mid_w].sum(),
        "south-east": change[mid_h:, mid_w:].sum(),
    }
    centre = change[h // 4 : h - h // 4, w // 4 : w - w // 4].sum()
    best = max(quadrants, key=quadrants.__getitem__)
    return "center" if centre > quadrants[best] else best


def _count(change: np.ndarray, min_pixels: int = 20) -> str:
    """Connected regions (8-connectivity) of at least ``min_pixels``."""
    try:
        import cv2

        n, _, stats, _ = cv2.connectedComponentsWithStats(change.astype(np.uint8), connectivity=8)
        regions = int((stats[1:, cv2.CC_STAT_AREA] >= min_pixels).sum()) if n > 1 else 0
    except ImportError:  # pragma: no cover - opencv is a core dependency
        regions = int(change.any())
    return str(regions) if regions < 5 else "5 or more"


def _class_net(labels: ChangeLabels, name: str, pixels: int) -> tuple[float, float]:
    """(gained, lost) share of the tile for a class."""
    if labels.before is not None and labels.after is not None:
        index = LAND_COVER.index(name)
        changed = labels.change
        gained = ((labels.after == index) & (labels.before != index) & changed).sum()
        lost = ((labels.before == index) & (labels.after != index) & changed).sum()
        return gained / pixels, lost / pixels
    gained = labels.gained[name].sum() if name in labels.gained else 0
    lost = labels.lost[name].sum() if name in labels.lost else 0
    return gained / pixels, lost / pixels


def _trend(gained: float, lost: float) -> str:
    net = gained - lost
    if net > TREND_THRESHOLD:
        return "increased"
    if net < -TREND_THRESHOLD:
        return "decreased"
    return "unchanged"


def _dominant(values: np.ndarray, mask: np.ndarray) -> int | None:
    picked = values[mask]
    if picked.size == 0:
        return None
    return int(np.bincount(picked, minlength=len(LAND_COVER)).argmax())


def generate_questions(labels: ChangeLabels, key: str) -> list[dict[str, str]]:
    """Question-answer pairs a tile's labels can answer.

    Args:
        labels: The tile's change labels.
        key: Stable tile identifier; selects the paraphrase of each question.
    """
    change = labels.change.astype(bool)
    pixels = change.size
    ratio = float(change.sum()) / pixels
    has_change = ratio >= 0.001
    qa: list[dict[str, str]] = []

    def ask(kind: str, answer: str) -> None:
        assert answer in ANSWERS, answer
        qa.append({"question": _pick(kind, key), "answer": answer, "type": kind})

    ask("any_change", "yes" if has_change else "no")
    ask("scale", _scale(ratio))
    ask("count", _count(change) if has_change else "0")
    if has_change:
        ask("where", _where(change))

    classes = labels.labelled_classes
    shares = {name: _class_net(labels, name, pixels) for name in classes}

    # "What changed?" -- the most informative answer the source supports.
    if not has_change:
        ask("what_changed", "no change")
    elif labels.focus == FOCUS_SEMANTIC and labels.before is not None:
        before = _dominant(labels.before, change)
        after = _dominant(labels.after, change)
        if before is not None and after is not None and before != after:
            ask("what_changed", f"{LAND_COVER[before]} to {LAND_COVER[after]}")
            ask("was", LAND_COVER[before])
            ask("became", LAND_COVER[after])
    elif labels.focus == FOCUS_DAMAGE:
        ask("what_changed", "building damage")
    elif labels.focus == FOCUS_BUILDING:
        gained, lost = shares.get("built-up", (ratio, 0.0))
        ask("what_changed", "new buildings" if gained >= lost else "buildings removed")
    else:
        ask("what_changed", "land-cover change")

    for name in classes:
        if f"trend_{name}" in PARAPHRASES:
            ask(f"trend_{name}", _trend(*shares[name]))

    if "built-up" in classes:
        gained, lost = shares["built-up"]
        ask("new_buildings", "yes" if gained >= PRESENCE_THRESHOLD else "no")
        if labels.focus != FOCUS_BUILDING or "built-up" in labels.lost:
            ask("demolished", "yes" if lost >= PRESENCE_THRESHOLD else "no")
    if "water" in classes:
        ask("flooding", "yes" if shares["water"][0] >= PRESENCE_THRESHOLD else "no")
    if "vegetation" in classes:
        ask("vegetation_loss", "yes" if shares["vegetation"][1] >= PRESENCE_THRESHOLD else "no")
    if labels.focus == FOCUS_DAMAGE:
        ask("damage", "yes" if has_change else "no")

    return qa
