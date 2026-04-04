"""
TTA + Beam Search + Spacing Fix (Combined)
===============================================================

Combines all three improvements:
  1. Test-Time Augmentation (TTA) - 5 models × N augmented copies
  2. Optimized beam search parameters
  3. Spacing post-processing fixes
"""

import json
import unicodedata
import argparse
import time
import zipfile
import numpy as np
from pathlib import Path
from collections import Counter
from PIL import Image, ImageEnhance, ImageFilter
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm
from xml.etree import ElementTree as ET


# ============================================================
# LINE EXTRACTION
# ============================================================

def extract_line_crops_from_xml(image_path, xml_path):
    full_page = Image.open(image_path).convert("RGB")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    line_crops = []
    
    for text_line in root.findall('.//{*}TextLine'):
        line_id = text_line.get('ID')
        if not line_id:
            continue
        hpos = int(float(text_line.get('HPOS', 0)))
        vpos = int(float(text_line.get('VPOS', 0)))
        width = int(float(text_line.get('WIDTH', 0)))
        height = int(float(text_line.get('HEIGHT', 0)))
        box = (hpos, vpos, hpos + width, vpos + height)
        try:
            line_crop = full_page.crop(box)
            target_height = 384
            aspect_ratio = line_crop.width / line_crop.height
            target_width = int(target_height * aspect_ratio)
            if target_width > 384 * 10:
                target_width = 384 * 10
            line_crop_resized = line_crop.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )
            line_crops.append((line_id, line_crop_resized))
        except Exception as e:
            print(f"Warning: Failed to crop line {line_id}: {e}")
    return line_crops


# ============================================================
# TTA AUGMENTATIONS
# ============================================================

def tta_augment(image, aug_idx):
    if aug_idx == 0:
        return image  # Original
    
    img = image.copy()
    np.random.seed(aug_idx * 1000)
    
    if aug_idx == 1:
        angle = np.random.uniform(-1.0, 1.0)
        img = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)
    elif aug_idx == 2:
        img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.90, 1.10))
    elif aug_idx == 3:
        img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.90, 1.10))
    elif aug_idx == 4:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    elif aug_idx == 5:
        img = img.filter(ImageFilter.SHARPEN)
    elif aug_idx == 6:
        angle = np.random.uniform(-0.8, 0.8)
        img = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)
        img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.92, 1.08))
    else:
        angle = np.random.uniform(-1.0, 1.0)
        img = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)
        img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.92, 1.08))
    return img


# ============================================================
# SPACING POST-PROCESSING
# ============================================================

def fix_spacing(text):
    """Apply spacing fixes that improved CER from 0.09614 → 0.09611."""
    # Remove double spaces
    while "  " in text:
        text = text.replace("  ", " ")
    # Strip leading/trailing spaces
    text = text.strip(" ")
    return text


# ============================================================
# MAJORITY VOTE
# ============================================================

def majority_vote(predictions_list):
    """Vote across all predictions for each line."""
    final = {}
    stats = {"unanimous": 0, "voted": 0, "total": 0}
    
    for line_id, preds in predictions_list.items():
        stats["total"] += 1
        vote_counts = Counter(preds)
        winner = vote_counts.most_common(1)[0][0]
        
        if len(vote_counts) == 1:
            stats["unanimous"] += 1
        else:
            stats["voted"] += 1
        
        final[line_id] = winner
    
    return final, stats


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='TTA + Beam + Spacing')
    parser.add_argument('--test_dir', type=Path, required=True)
    parser.add_argument('--model_paths', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n_augmentations', type=int, default=3)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--length_penalty', type=float, default=2.0)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Models: {len(args.model_paths)}")
    print(f"TTA augmentations: {args.n_augmentations}")
    print(f"Beam search: num_beams={args.num_beams}, length_penalty={args.length_penalty}")
    print(f"Total virtual models: {len(args.model_paths) * args.n_augmentations}")
    
    # Extract line crops ONCE
    print(f"\nExtracting line crops from {args.test_dir}...")
    xml_files = sorted(args.test_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} test pages")
    
    flat_crops = []
    for xml_path in tqdm(xml_files, desc="Extracting"):
        image_path = xml_path.with_suffix('.jpg')
        if not image_path.exists():
            continue
        flat_crops.extend(extract_line_crops_from_xml(image_path, xml_path))
    print(f"Total lines: {len(flat_crops)}")
    
    # Run inference on each model × each augmentation
    combined = {}  # line_id → list of ALL predictions
    
    for model_idx, model_path in enumerate(args.model_paths):
        print(f"\n{'='*60}")
        print(f"Model {model_idx + 1}/{len(args.model_paths)}: {model_path}")
        print(f"{'='*60}")
        
        processor = TrOCRProcessor.from_pretrained(str(model_path))
        model = VisionEncoderDecoderModel.from_pretrained(str(model_path))
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            for aug_idx in range(args.n_augmentations):
                aug_name = "original" if aug_idx == 0 else f"aug_{aug_idx}"
                
                for i in tqdm(range(0, len(flat_crops), args.batch_size),
                             desc=f"  {aug_name}"):
                    batch = flat_crops[i:i + args.batch_size]
                    images = [tta_augment(crop[1], aug_idx) for crop in batch]
                    line_ids = [crop[0] for crop in batch]
                    
                    pixel_values = processor(images, return_tensors="pt").pixel_values.to(device)
                    
                    generated_ids = model.generate(
                        pixel_values,
                        max_length=args.max_length,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,
                        early_stopping=True,
                    )
                    
                    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    
                    for line_id, pred in zip(line_ids, preds):
                        text = unicodedata.normalize('NFD', pred)
                        text = fix_spacing(text)
                        if line_id not in combined:
                            combined[line_id] = []
                        combined[line_id].append(text)
        
        del model
        torch.cuda.empty_cache()
    
    # Vote
    print(f"\n{'='*60}")
    print(f"COMBINING PREDICTIONS")
    print(f"{'='*60}")
    
    final_predictions, stats = majority_vote(combined)
    
    total_votes = len(args.model_paths) * args.n_augmentations
    print(f"  Total lines: {stats['total']}")
    print(f"  Votes per line: {total_votes}")
    print(f"  Unanimous: {stats['unanimous']} ({stats['unanimous']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Contested: {stats['voted']} ({stats['voted']/max(stats['total'],1)*100:.1f}%)")
    
    # Final spacing fix on voted results
    for line_id in final_predictions:
        final_predictions[line_id] = fix_spacing(final_predictions[line_id])
    
    # Save predictions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_predictions, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(final_predictions)} predictions to: {args.output}")
    
    # Create submission zip
    zip_path = args.output.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(args.output, arcname='predictions.json')
    print(f"Submission zip: {zip_path}")
    
    # Also save no-TTA version (original augmentation only) for comparison
    no_tta = {}
    for line_id, preds in combined.items():
        # Every Nth prediction is original (aug_idx=0), where N = n_augmentations
        originals = preds[::args.n_augmentations]
        vote_counts = Counter(originals)
        no_tta[line_id] = fix_spacing(vote_counts.most_common(1)[0][0])
    
    no_tta_path = args.output.with_stem(args.output.stem + "_no_tta")
    with open(no_tta_path, 'w', encoding='utf-8') as f:
        json.dump(no_tta, f, indent=2, ensure_ascii=False)
    
    no_tta_zip = no_tta_path.with_suffix('.zip')
    with zipfile.ZipFile(no_tta_zip, 'w') as zf:
        zf.write(no_tta_path, arcname='predictions.json')
    print(f"No-TTA baseline zip: {no_tta_zip}")
    
    # Sample predictions
    print(f"\n Sample predictions:")
    for line_id, pred in list(final_predictions.items())[:5]:
        print(f"  {line_id}: {pred[:80]}")


if __name__ == '__main__':
    main()
