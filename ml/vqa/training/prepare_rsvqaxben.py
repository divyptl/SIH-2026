"""
Prepares the RSVQAxBEN dataset for SatQuery VQA training.

Extracts Images.zip and merges the relational JSON files (questions, answers, images)
into the flattened train.jsonl and val.jsonl manifests expected by our VQAModel.
"""

import json
import zipfile
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def extract_zip(zip_path: Path, extract_to: Path):
    if not zip_path.exists():
        print(f"Warning: {zip_path} not found. Skipping extraction.")
        return
    print(f"Extracting {zip_path.name} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_rsvqa_split(split: str, source_dir: Path, out_path: Path, img_dir_name: str):
    q_file = source_dir / f"RSVQAxBEN_split_{split}_questions.json"
    a_file = source_dir / f"RSVQAxBEN_split_{split}_answers.json"
    i_file = source_dir / f"RSVQAxBEN_split_{split}_images.json"
    
    if not (q_file.exists() and a_file.exists() and i_file.exists()):
        print(f"Warning: Missing files for {split} split. Skipping.")
        return

    print(f"\nProcessing {split} split...")
    questions = load_json(q_file)
    answers = load_json(a_file)
    images = load_json(i_file)
    
    # RSVQA answers list contains all answers. Map question_id -> list of answers.
    ans_by_qid = defaultdict(list)
    for ans in answers["answers"]:
        ans_by_qid[ans["question_id"]].append(ans["answer"])
        
    # Map image_id -> filename (assuming BigEarthNet naming or generic IDs)
    img_by_id = {}
    for img in images["images"]:
        # If filename is provided, use it, else assume ID + .tif (common in RSVQA)
        filename = img.get("filename", f"{img['id']}.tif")
        img_by_id[img["id"]] = filename

    out_file = out_path / f"{split}.jsonl"
    print(f"Writing to {out_file.name}...")
    
    with open(out_file, "w", encoding="utf-8") as f:
        for q in tqdm(questions["questions"]):
            qid = q["id"]
            img_id = q["img_id"]
            q_text = q["question"]
            
            if qid not in ans_by_qid or img_id not in img_by_id:
                continue
                
            # Take the first available answer (or use majority vote if preferred)
            ans_text = ans_by_qid[qid][0]
            img_filename = img_by_id[img_id]
            
            record = {
                "image": f"{img_dir_name}/{img_filename}",
                "question": q_text,
                "answer": ans_text,
                "modality": "multispectral"  # BigEarthNet is typically multispectral/optical
            }
            f.write(json.dumps(record) + "\n")


def prepare_rsvqa(source_dir: str, output_dir: str = "data/rsvqaxben_processed"):
    source = Path(source_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    images_zip = source / "Images.zip"
    images_dir = out_path / "Images"
    
    # 1. Extract Images.zip
    extract_zip(images_zip, images_dir)
    
    # 2. Process Splits
    for split in ["train", "val", "test"]:
        process_rsvqa_split(split, source, out_path, img_dir_name="Images")

    print("\n✅ Dataset preparation complete!")
    print(f"Manifests and images are ready in: {out_path.absolute()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare RSVQA dataset.")
    parser.add_argument("--source", type=str, required=True, help="Directory containing the zip and jsons")
    parser.add_argument("--output", type=str, default="data/rsvqaxben_processed", help="Where to save the processed files")
    args = parser.parse_args()
    
    prepare_rsvqa(args.source, args.output)
