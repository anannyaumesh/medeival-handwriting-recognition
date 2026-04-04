#!/usr/bin/env python3
"""
CATMuS Kraken inference.
"""

import unicodedata
import json
import zipfile
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm

from kraken.lib import models
from kraken.binarization import nlbin
from kraken.containers import Segmentation, BaselineLine
from kraken.rpred import rpred


# ============================================================
# CONFIG
# ============================================================

CATMUS_MODEL = "/home/guest/.local/share/htrmopo/fef4968a-652b-5c8b-a6f0-1a1162abe8ec/catmus-medieval.mlmodel"

TASK_DIRS = {
    "task_1": Path("/mnt/hdd2/rainfall_project/code/gsmap/proj/test/cmmhwr_test/task_1"),
    "task_2": Path("/mnt/hdd2/rainfall_project/code/gsmap/proj/test/cmmhwr_test/task_2"),
    "task_3": Path("/mnt/hdd2/rainfall_project/code/gsmap/proj/test/cmmhwr_test/task_3"),
}

OUTPUT_DIR = Path("/mnt/hdd2/rainfall_project/code/gsmap/proj/test/submissions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def find_image(xml_path):
    base = xml_path.with_suffix("")
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def fix_spacing(text):
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def preprocess_page(image):
    """
    Preprocess a full page image for Kraken recognition.
    """
    # Step 1: Convert to grayscale for binarization
    gray = image.convert("L")
    
    # Step 2: Binarize with nlbin (Kraken's built-in binarizer)
    # This normalizes contrast, removes background noise
    binarized = nlbin(gray)
    
    return binarized


def build_segmentation(xml_path, image):
    """Build segmentation from ALTO XML with proper baselines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
    
    w, h = image.size
    lines = []
    line_ids = []
    
    for text_line in root.findall(".//alto:TextLine", ns):
        line_id = text_line.attrib.get("ID", f"line_{len(lines)}")
        hpos = int(float(text_line.attrib.get("HPOS", 0)))
        vpos = int(float(text_line.attrib.get("VPOS", 0)))
        width = int(float(text_line.attrib.get("WIDTH", 0)))
        height = int(float(text_line.attrib.get("HEIGHT", 0)))
        
        x0, y0 = hpos, vpos
        x1, y1 = hpos + width, vpos + height
        
        # Clamp to image bounds
        x0 = max(1, min(x0, w - 2))
        y0 = max(1, min(y0, h - 2))
        x1 = max(x0 + 2, min(x1, w - 1))
        y1 = max(y0 + 2, min(y1, h - 1))
        
        # Baseline at ~85% of line height
        baseline_y = int(y0 + (y1 - y0) * 0.85)
        baseline = [[x0 + 1, baseline_y], [x1 - 1, baseline_y]]
        boundary = [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
        ]
        
        lines.append(
            BaselineLine(
                id=line_id,
                baseline=baseline,
                boundary=boundary,
                tags={"type": "default"},
            )
        )
        line_ids.append(line_id)
    
    seg = Segmentation(
        type="baselines",
        imagename=str(xml_path),
        text_direction="horizontal-lr",
        script_detection=False,
        lines=lines,
        regions={},
    )
    
    return seg, line_ids


# ============================================================
# RUN ON ONE TASK
# ============================================================

def run_task(nn, task_name, task_dir):
    print(f"\n{'='*60}")
    print(f"Running CATMuS (preprocessed) on {task_name}")
    print(f"  Directory: {task_dir}")
    print(f"{'='*60}")
    
    xml_files = sorted(task_dir.glob("*.xml"))
    print(f"  Found {len(xml_files)} XML files")
    
    predictions = {}
    total_lines = 0
    errors = 0
    
    for xml_path in tqdm(xml_files, desc=f"  {task_name}"):
        img_path = find_image(xml_path)
        if img_path is None:
            errors += 1
            continue
        
        try:
            # Load and PREPROCESS the image
            raw_image = Image.open(img_path).convert("RGB")
            image = preprocess_page(raw_image)
            
        except Exception as e:
            print(f"  Warning: Failed to preprocess {img_path}: {e}")
            # Fallback: try without binarization
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                errors += 1
                continue
        
        try:
            seg, line_ids = build_segmentation(xml_path, image)
            preds = list(rpred(nn, image, seg))
            
            for pred_obj, line_id in zip(preds, line_ids):
                text = unicodedata.normalize('NFD', pred_obj.prediction)
                text = fix_spacing(text)
                predictions[line_id] = text
                total_lines += 1
                
        except Exception as e:
            print(f"  Warning: Failed on {xml_path.name}: {e}")
            errors += 1
            continue
    
    print(f"  Total lines: {total_lines}")
    print(f"  Errors: {errors}")
    
    # Save
    pred_file = OUTPUT_DIR / f"catmus_preproc_{task_name}_predictions.json"
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {pred_file}")
    
    zip_file = OUTPUT_DIR / f"catmus_preproc_{task_name}.zip"
    with zipfile.ZipFile(zip_file, 'w') as zf:
        zf.write(pred_file, arcname='predictions.json')
    print(f"  Zip: {zip_file}")
    
    # Samples
    print(f"\n  Sample predictions:")
    for line_id, text in list(predictions.items())[:5]:
        print(f"    {line_id}: {text[:70]}")
    
    return predictions


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading CATMuS Medieval model...")
    nn = models.load_any(CATMUS_MODEL)
    print("  Model loaded ✅")
    
    all_predictions = {}
    
    for task_name, task_dir in TASK_DIRS.items():
        if not task_dir.exists():
            print(f"\n⚠️  {task_name} not found: {task_dir}")
            continue
        
        preds = run_task(nn, task_name, task_dir)
        all_predictions[task_name] = preds
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for task_name, preds in all_predictions.items():
        non_empty = sum(1 for v in preds.values() if v)
        print(f"  {task_name}: {len(preds)} total, {non_empty} non-empty")
        print(f"    File: {OUTPUT_DIR}/catmus_preproc_{task_name}_predictions.json")


if __name__ == "__main__":
    main()