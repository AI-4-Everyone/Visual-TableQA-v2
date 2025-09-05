#!/usr/bin/env python3
# Build MiniCPM-style JSONs for AI-4-Everyone/Visual-TableQA

"""
Build InternVL-style JSON files (train/val/test) from a HF dataset with images.
Defaults to AI-4-Everyone/Visual-TableQA and clamps images to a given long edge.

Usage examples:
  python build_internvl_dataset.py
"""

import os, torch
from datasets import load_dataset
from PIL import Image, ImageOps
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Dict, List, Any
import json
import warnings
warnings.filterwarnings("ignore")

DATASET_REPO = "AI-4-Everyone/Visual-TableQA"
output_dir = "internvl_data"              # where JSONs live
IMG_DIR = OUT_DIR / "clamped_images"      # where images are saved
LONG_EDGE = 1024                          # clamp longest side

system_message = """You are a Vision Language Model specialized in interpreting visual data from charts and diagrams images.
Analyze the image and answer the questions with step-by-step reasoning—stay concise, but include any reasoning that’s relevant."""

def to_pil(img):
    return img if isinstance(img, Image.Image) else Image.fromarray(img)

def clamp_long_edge(img, longest=LONG_EDGE):
    img = to_pil(img)
    if img.mode != "RGB":  # avoid JPEG RGBA errors
        img = img.convert("RGB")
    return ImageOps.contain(img, (longest, longest), Image.Resampling.BICUBIC)

def prepare_data_for_official_script(split, split_name, output_dir="internvl_data"):
    """
    Convert dataset to the format expected by InternVL2 official scripts

    Expected format:
    - images/: directory containing all images
    - annotations.json: JSONL file with question-answer pairs
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    annotations = []

    for i, sample in enumerate(split):
        # Save image
        idx = sample["table_id"]
        img = clamp_long_edge(sample["image"])  # PIL.Image object
        width, height = img.size
        img_path = f"/notebooks/{output_dir}/images/{idx}.jpg"
        img.save(img_path)

        # Prepare annotation in InternVL2 format
        question = f"<image>\n{system_message}\n{sample['question']}"
        annotation = {
            "id": i,
            "image": f"images/{idx}.jpg",
            "width": width,
            "height": height,
            "conversations": [
                {
                    "from": "human",
                    "value": question,
                },
                {
                    "from": "gpt",
                    "value": sample["answer"]
                }
            ]
        }
        annotations.append(annotation)

    # Save annotations as JSONL
    with open(f"{output_dir}/annotations_{split_name}.jsonl", "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")

    print(f"Data prepared in {output_dir}")
    print(f"Total samples: {len(annotations)}")

if __name__ == "__main__":
    print(f"Loading dataset: {DATASET_REPO}")
    ds = load_dataset(DATASET_REPO)

    train = ds.get("train")
    evald   = ds.get("validation")
    test  = ds.get("test")

    prepare_data_for_official_script(train, "train")
    prepare_data_for_official_script(evald, "val")
    prepare_data_for_official_script(test, "test")

    print("Done.")
