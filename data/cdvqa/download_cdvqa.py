"""
Download LEVIR-CD+ from Hugging Face and format it as a CDVQA dataset
with rich, diverse question-answer annotations derived from the change masks.

Usage:
    python data/cdvqa/download_cdvqa.py                    # default 500 samples/split
    python data/cdvqa/download_cdvqa.py --max-samples 100  # quick test
    python data/cdvqa/download_cdvqa.py --check            # verify integrity
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Force Hugging Face to use D: drive for all caches (datasets and hub)
hf_cache_dir = Path(__file__).resolve().parent.parent / 'hf_cache'
os.environ['HF_HOME'] = str(hf_cache_dir)
os.environ['HF_DATASETS_CACHE'] = str(hf_cache_dir)
os.environ['HF_HUB_CACHE'] = str(hf_cache_dir)

try:
    import numpy as np
    from datasets import load_dataset
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("Missing required libraries. Please install them first:")
    print("pip install datasets numpy pillow tqdm")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Annotation generation helpers
# ---------------------------------------------------------------------------

def _connected_components(binary_mask: np.ndarray) -> int:
    """Count connected components in a binary mask (simple flood-fill, no scipy needed)."""
    visited = np.zeros_like(binary_mask, dtype=bool)
    count = 0
    rows, cols = binary_mask.shape

    for r in range(rows):
        for c in range(cols):
            if binary_mask[r, c] and not visited[r, c]:
                count += 1
                # BFS flood-fill
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        continue
                    if visited[cr, cc] or not binary_mask[cr, cc]:
                        continue
                    visited[cr, cc] = True
                    stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])
    return count


def _dominant_quadrant(binary_mask: np.ndarray) -> str:
    """Return the quadrant where the most change pixels are concentrated."""
    h, w = binary_mask.shape
    mid_h, mid_w = h // 2, w // 2

    quadrants = {
        "north-west": binary_mask[:mid_h, :mid_w].sum(),
        "north-east": binary_mask[:mid_h, mid_w:].sum(),
        "south-west": binary_mask[mid_h:, :mid_w].sum(),
        "south-east": binary_mask[mid_h:, mid_w:].sum(),
    }

    # Check if change is spread across center
    center_h4, center_w4 = h // 4, w // 4
    center_sum = binary_mask[center_h4:h-center_h4, center_w4:w-center_w4].sum()
    quadrants["center"] = center_sum

    return max(quadrants, key=quadrants.get)


def _change_scale(change_ratio: float) -> str:
    """Classify the scale of change based on changed pixel ratio."""
    if change_ratio > 0.15:
        return "large scale change"
    elif change_ratio > 0.03:
        return "moderate change"
    elif change_ratio > 0.0:
        return "minor change"
    else:
        return "no significant change detected"


def _count_label(n: int) -> str:
    """Map a count to the canonical answer format."""
    if n >= 5:
        return "5 or more"
    return str(max(n, 1))


def generate_questions(mask_arr: np.ndarray) -> list[dict]:
    """
    Generate a diverse set of question-answer pairs from a binary change mask.
    Uses all relevant categories from CANONICAL_ANSWERS.
    """
    # Binarize (handle both 0/1 and 0/255 masks)
    threshold = 128 if mask_arr.max() > 1 else 0.5
    binary = (mask_arr > threshold).astype(np.uint8)
    total_pixels = binary.size
    changed_pixels = int(binary.sum())
    change_ratio = changed_pixels / total_pixels if total_pixels > 0 else 0.0
    has_change = changed_pixels > 0

    questions = []
    qid = 0

    # --- Q1: Yes/No existence of change ---
    questions.append({
        "id": qid,
        "question": "Did any new structures appear between the two dates?",
        "answer": "yes" if has_change else "no",
        "change_type": "urban",
    })
    qid += 1

    # --- Q2: Increase / Decrease / Unchanged ---
    if has_change:
        inc_dec = "increased"
    else:
        inc_dec = "unchanged"
    questions.append({
        "id": qid,
        "question": "Has the built-up area increased, decreased, or remained unchanged?",
        "answer": inc_dec,
        "change_type": "urban",
    })
    qid += 1

    # --- Q3: Scale of change ---
    scale = _change_scale(change_ratio)
    questions.append({
        "id": qid,
        "question": "How significant is the overall change between the two images?",
        "answer": scale,
        "change_type": "general",
    })
    qid += 1

    # --- Q4: Spatial location ---
    if has_change:
        quadrant = _dominant_quadrant(binary)
    else:
        quadrant = "center"  # fallback
    questions.append({
        "id": qid,
        "question": "Where did the primary land-cover change take place?",
        "answer": quadrant,
        "change_type": "spatial",
    })
    qid += 1

    # --- Q5: Count of distinct change regions ---
    if has_change:
        n_regions = _connected_components(binary)
    else:
        n_regions = 0
    questions.append({
        "id": qid,
        "question": "How many distinct change regions can be identified?",
        "answer": _count_label(n_regions) if has_change else "no significant change detected",
        "change_type": "counting",
    })
    qid += 1

    # --- Q6: Type of change (LEVIR-CD is about buildings) ---
    if has_change:
        if change_ratio > 0.10:
            change_type_answer = "new built-up area constructed"
        elif change_ratio > 0.04:
            change_type_answer = "building expansion"
        else:
            change_type_answer = "agricultural land converted to urban"
    else:
        change_type_answer = "unchanged"
    questions.append({
        "id": qid,
        "question": "What major environmental change occurred between T1 and T2?",
        "answer": change_type_answer,
        "change_type": "general",
    })
    qid += 1

    # --- Q7: Signs of construction ---
    questions.append({
        "id": qid,
        "question": "Are there signs of recent construction activity?",
        "answer": "yes" if has_change else "no",
        "change_type": "urban",
    })
    qid += 1

    # --- Q8: Vegetation question (LEVIR-CD is building-focused, so vegetation
    #         loss correlates with construction) ---
    if has_change and change_ratio > 0.05:
        veg_answer = "vegetation cleared / deforestation"
    elif has_change:
        veg_answer = "decreased"
    else:
        veg_answer = "unchanged"
    questions.append({
        "id": qid,
        "question": "Has the vegetation cover increased or decreased?",
        "answer": veg_answer,
        "change_type": "vegetation",
    })
    qid += 1

    return questions


# ---------------------------------------------------------------------------
# Dataset check
# ---------------------------------------------------------------------------

def check_dataset(root_dir):
    print(f"Checking dataset integrity in {root_dir}...")
    splits = ['train', 'val', 'test']
    missing = False
    for split in splits:
        anno_file = root_dir / f"{split}_annotations.json"
        if not anno_file.exists():
            print(f"[MISSING] {split}_annotations.json not found at {anno_file}")
            missing = True
            continue

        with open(anno_file, 'r') as f:
            data = json.load(f)
            samples = data.get("samples", [])
            total_qs = sum(len(s.get("questions", [])) for s in samples)
            print(f"[OK] {split}_annotations.json -- {len(samples)} image pairs, {total_qs} Q&A pairs")

    if missing:
        print("\nDataset check failed. Run without --check to download.")
        sys.exit(1)
    else:
        print("\nDataset check passed!")


# ---------------------------------------------------------------------------
# Main download + format pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download LEVIR-CD+ and format as CDVQA dataset")
    parser.add_argument("--check", action="store_true", help="Check dataset integrity")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max image pairs per split (default: 5000)")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent

    if args.check:
        check_dataset(root_dir)
        return

    print(f"Downloading LEVIR-CD (Cropped 256x256) from Hugging Face into {root_dir}...")
    print(f"Will process up to {args.max_samples} samples per split.")
    print("Using streaming mode to avoid disk-space issues.\n")

    DATASET_ID = 'ericyu/LEVIRCD_Cropped_256'
    local_cache = root_dir.parent / 'hf_cache'
    local_cache.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'test']:
        print(f"Processing '{split}' split...")
        split_dir = root_dir / split

        t1_dir = split_dir / 'images_t1'
        t2_dir = split_dir / 'images_t2'
        mask_dir = split_dir / 'masks'

        for d in [t1_dir, t2_dir, mask_dir]:
            d.mkdir(parents=True, exist_ok=True)

        annotations = []

        try:
            stream = load_dataset(
                DATASET_ID, split=split, streaming=True, cache_dir=str(local_cache),
            )
        except Exception as e:
            print(f"  Warning: Could not load split '{split}': {e}")
            continue

        for i, item in enumerate(tqdm(stream, total=args.max_samples, desc=f"  {split}")):
            if i >= args.max_samples:
                break

            # Dynamically resolve column names
            keys = list(item.keys())
            img_A_key = 'image_A' if 'image_A' in keys else keys[0]
            img_B_key = 'image_B' if 'image_B' in keys else keys[1]
            label_key = 'label' if 'label' in keys else keys[2]

            img_A = item[img_A_key]
            img_B = item[img_B_key]
            mask = item[label_key]

            if not isinstance(img_A, Image.Image):
                print(f"  Unexpected image type at index {i}: {type(img_A)}")
                continue

            base_name = f"sample_{i:04d}.png"
            img_A.save(t1_dir / base_name)
            img_B.save(t2_dir / base_name)
            mask.save(mask_dir / base_name)

            # Generate rich Q&A from the mask
            mask_arr = np.array(mask.convert("L"))
            qa_pairs = generate_questions(mask_arr)

            annotations.append({
                "id": f"{split}_{i:04d}",
                "image_t1": f"{split}/images_t1/{base_name}",
                "image_t2": f"{split}/images_t2/{base_name}",
                "mask": f"{split}/masks/{base_name}",
                "questions": qa_pairs,
            })

        # Save annotations
        anno_path = root_dir / f"{split}_annotations.json"
        with open(anno_path, "w") as f:
            json.dump({"samples": annotations}, f, indent=2)

        total_qs = sum(len(s["questions"]) for s in annotations)
        print(f"  Saved {len(annotations)} image pairs with {total_qs} Q&A pairs to {anno_path.name}")

    print(f"\nDone! CDVQA dataset prepared in {root_dir}")


if __name__ == "__main__":
    main()

