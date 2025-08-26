#!/usr/bin/env python3
# Build MiniCPM-style JSONs for AI-4-Everyone/TableQA-v2

"""
Build MiniCPM-style JSON files (train/val/test) from a HF dataset with images.
Defaults to AI-4-Everyone/TableQA-v2 and clamps images to a given long edge.

Usage examples:
  python build_minicpm_dataset.py
"""

from pathlib import Path
import json
from datasets import load_dataset
from PIL import Image, ImageOps, Image

DATASET_REPO = "AI-4-Everyone/TableQA-v2"
OUT_DIR = Path("minicpm_ds")              # where JSONs live
IMG_DIR = OUT_DIR / "clamped_images"      # where images are saved
LONG_EDGE = 1024                          # clamp longest side
WRITE_ABSOLUTE_PATHS = False              # set True if you want /abs/paths in JSON

def to_pil(img):
    return img if isinstance(img, Image.Image) else Image.fromarray(img)

def clamp_long_edge(img, longest=LONG_EDGE):
    img = to_pil(img)
    if img.mode != "RGB":  # avoid JPEG RGBA errors
        img = img.convert("RGB")
    return ImageOps.contain(img, (longest, longest), Image.Resampling.BICUBIC)

def process_split(split, split_name, out_json):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for i, sample in enumerate(split):
        idx = sample["table_id"]
        img = clamp_long_edge(sample["image"], LONG_EDGE)
        question = sample["question"]
        answer = sample["answer"]

        img_path = IMG_DIR / f"{idx}.jpg"
        img.save(img_path, format="JPEG", quality=95, optimize=True)

        path_for_json = img_path.resolve() if WRITE_ABSOLUTE_PATHS else img_path
        records.append({
            "id": str(i),
            "image": str(path_for_json),
            "conversations": [
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer},
            ],
        })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[{split_name}] wrote {len(records)} items -> {out_json}")

if __name__ == "__main__":
    print(f"Loading dataset: {DATASET_REPO}")
    ds = load_dataset(DATASET_REPO)

    train = ds.get("train")
    val   = ds.get("validation")
    test  = ds.get("test")

    process_split(train, "train", OUT_DIR / "minicpm_train.json")
    process_split(val, "validation", OUT_DIR / "minicpm_val.json")
    process_split(test, "test", OUT_DIR / "minicpm_test.json")

    print("Done.")
