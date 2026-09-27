"""
Prepares the local VRSBench dataset for SatQuery VQA training.

Extracts Images_train.zip and Images_val.zip, and converts the LLaVA-style
conversational JSON files (VRSBench_train.json) into the flattened 
train.jsonl and val.jsonl manifests expected by our custom PyTorch VQAModel.
"""

import json
import zipfile
from pathlib import Path
from tqdm import tqdm


def extract_zip(zip_path: Path, extract_to: Path):
    if not zip_path.exists():
        print(f"Warning: {zip_path} not found. Skipping extraction.")
        return
    print(f"Extracting {zip_path.name} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def process_vrsbench_json(json_path: Path, output_jsonl: Path, image_dir_name: str):
    if not json_path.exists():
        print(f"Warning: {json_path} not found. Skipping conversion.")
        return
        
    print(f"Converting {json_path.name} to {output_jsonl.name}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in tqdm(data, desc=json_path.name):
            image_name = item.get("image")
            if not image_name:
                continue

            # In VRSBench, the dialogues are stored in "conversations"
            conversations = item.get("conversations", [])
            
            # Group pairs of (human, gpt)
            for i in range(0, len(conversations) - 1, 2):
                human = conversations[i]
                gpt = conversations[i+1]
                
                if human["from"] == "human" and gpt["from"] == "gpt":
                    q = human["value"]
                    # Clean up <image>\n artifacts from LLaVA format
                    q = q.replace("<image>\n", "").strip()
                    # Optional: clean up [caption] or [vqa] tags
                    q = q.replace("[caption] ", "").replace("[vqa] ", "").strip()
                    
                    a = gpt["value"].strip()
                    
                    record = {
                        "image": f"{image_dir_name}/{image_name}",
                        "question": q,
                        "answer": a,
                        "modality": "optical"
                    }
                    f.write(json.dumps(record) + "\n")


def prepare_local_vrsbench(source_dir: str, output_dir: str = "data/vrsbench"):
    source = Path(source_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Paths to the raw files
    train_zip = source / "Images_train.zip"
    val_zip = source / "Images_val.zip"
    train_json = source / "VRSBench_train.json"
    val_json = source / "VRSBench_EVAL_referring.json"  # or whichever eval json you want to use
    
    # Extraction directories
    images_train_dir = out_path / "Images_train"
    images_val_dir = out_path / "Images_val"
    
    # 1. Extract ZIP files
    extract_zip(train_zip, images_train_dir)
    extract_zip(val_zip, images_val_dir)
    
    # 2. Convert JSONs to JSONL
    train_out = out_path / "train.jsonl"
    val_out = out_path / "val.jsonl"
    
    process_vrsbench_json(train_json, train_out, image_dir_name="Images_train")
    process_vrsbench_json(val_json, val_out, image_dir_name="Images_val")

    print("\n✅ Dataset preparation complete!")
    print(f"Manifests and images are ready in: {out_path.absolute()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare local VRSBench files.")
    parser.add_argument("--source", type=str, required=True, help="Directory containing the zips and jsons")
    parser.add_argument("--output", type=str, default="data/vrsbench", help="Where to save the processed files")
    args = parser.parse_args()
    
    prepare_local_vrsbench(args.source, args.output)
