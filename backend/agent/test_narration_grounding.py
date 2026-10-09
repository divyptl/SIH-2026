"""Checks on the plain-language narration of grounding results.

Run from backend/:
    python -m unittest agent.test_narration_grounding -v
"""

from __future__ import annotations

import unittest

from agent.narration import NarrationRejected, check_grounding, grounding_facts_text
from services.specialists import box_position, certainty_phrase, position_phrase

FACTS = {
    "kind": "grounding",
    "query": "the large ship",
    "regions": [
        {"number": 1, "evidence_index": 0, "box": [0.05, 0.05, 0.25, 0.2],
         "position": "top-left", "place": "in the top-left corner", "share": 0.03,
         "size": "a small area", "confidence": 0.72, "certainty": "confident",
         "measure": "Best match, in the top-left corner, covering a small area."},
        {"number": 2, "evidence_index": 1, "box": [0.75, 0.4, 0.95, 0.6],
         "position": "right", "place": "on the right side", "share": 0.04,
         "size": "a small area", "confidence": 0.15, "certainty": "not very confident",
         "measure": "Other possible match, on the right side, covering a small area."},
    ],
}

GOOD = {
    "summary": "The model found a large white ship in the top-left corner, marked as box 1. "
               "It is confident about this. Box 2 on the right might also be a boat.",
    "boxes": [
        {"number": 1, "description": "A long white ship moored beside a pier"},
        {"number": 2, "description": "A smaller boat on the water, on the right"},
    ],
}


def reply(**changes):
    parsed = {"summary": GOOD["summary"], "boxes": [dict(b) for b in GOOD["boxes"]]}
    parsed.update(changes)
    return parsed


class GroundingNarrationChecks(unittest.TestCase):
    def test_faithful_reply_is_accepted(self):
        narration = check_grounding(reply(), FACTS)
        self.assertIn("box 1", narration.summary)
        self.assertEqual(set(narration.regions), {1, 2})

    def test_box_the_model_did_not_mark(self):
        with self.assertRaisesRegex(NarrationRejected, "box 3"):
            check_grounding(reply(boxes=[{"number": 3, "description": "A tanker"}]), FACTS)

    def test_invented_count_in_words(self):
        with self.assertRaisesRegex(NarrationRejected, "three"):
            check_grounding(reply(summary="There are three ships in box 1."), FACTS)

    def test_model_percentage_is_allowed(self):
        narration = check_grounding(
            reply(summary="Box 1, in the top-left corner, covers about 3% of the image."), FACTS
        )
        self.assertIn("3%", narration.summary)

    def test_count_cannot_borrow_a_percentage(self):
        # Box 1 covers 3% of the image; that must not license "3 ships".
        with self.assertRaisesRegex(NarrationRejected, "3"):
            check_grounding(reply(summary="Box 1 shows 3 ships."), FACTS)

    def test_invented_number_in_digits(self):
        with self.assertRaisesRegex(NarrationRejected, "40"):
            check_grounding(reply(summary="Box 1 shows a ship about 40 metres long."), FACTS)

    def test_compass_direction(self):
        with self.assertRaisesRegex(NarrationRejected, "compass"):
            check_grounding(reply(summary="Box 1 shows a ship in the north of the harbour."), FACTS)

    def test_side_with_no_box(self):
        with self.assertRaisesRegex(NarrationRejected, "bottom"):
            check_grounding(reply(summary="A ship is visible near the bottom, in box 1."), FACTS)

    def test_box_placed_on_the_wrong_side(self):
        boxes = [{"number": 1, "description": "A ship on the right side"}]
        with self.assertRaisesRegex(NarrationRejected, "right"):
            check_grounding(reply(boxes=boxes), FACTS)

    def test_jargon(self):
        with self.assertRaisesRegex(NarrationRejected, "detection"):
            check_grounding(reply(summary="The detection in box 1 is a ship."), FACTS)

    def test_missing_summary(self):
        with self.assertRaises(NarrationRejected):
            check_grounding({"boxes": GOOD["boxes"]}, FACTS)

    def test_facts_text_lists_every_box(self):
        text = grounding_facts_text("the large ship", FACTS)
        self.assertIn("Box 1: the best match; position: top-left", text)
        self.assertIn("Box 2: a less likely match; position: right", text)


class GroundingWording(unittest.TestCase):
    def test_positions(self):
        self.assertEqual(box_position(0.0, 0.0, 0.2, 0.2), "top-left")
        self.assertEqual(box_position(0.4, 0.4, 0.6, 0.6), "centre")
        self.assertEqual(box_position(0.8, 0.4, 0.95, 0.6), "right")
        self.assertEqual(box_position(0.4, 0.8, 0.6, 0.95), "bottom")
        self.assertEqual(position_phrase("bottom-right"), "in the bottom-right corner")
        self.assertEqual(position_phrase("left"), "on the left side")

    def test_certainty(self):
        self.assertEqual(certainty_phrase(0.8), "confident")
        self.assertEqual(certainty_phrase(0.5), "fairly confident")
        self.assertEqual(certainty_phrase(0.1), "not very confident")


if __name__ == "__main__":
    unittest.main()
