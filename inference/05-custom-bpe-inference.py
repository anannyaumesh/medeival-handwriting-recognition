"""
Inference for ViT + Custom BPE model on test sets.
Uses the custom BPE tokenizer for decoding instead of TrOCR's RoBERTa tokenizer.
"""

import json
import unicodedata
import argparse
import zipfile
from pathlib import Path
from PIL import Image
import torch
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    PreTrainedTokenizerFast,
)
from tqdm import tqdm
from xml.etree import ElementTree as ET


MODEL_PATH = Path("/mnt/hdd2/rainfall_project/code/gsmap/proj/models/trocr_custom_bpe/final_model")
TOKENIZER_PATH = "/mnt/hdd2/rainfall_project/code/gsmap/proj/models/trocr_custom_bpe/tokenizer"
IMAGE_PROCESSOR_PATH = "/mnt/hdd2/rainfall_project/code/gsmap/proj/models/tridis_HTR"


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


def fix_spacing(text):
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load custom BPE tokenizer from CORRECT location
    print(f"Loading custom BPE tokenizer from {TOKENIZER_PATH}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
    print(f"  Vocab size: {len(tokenizer)}")

    # Load image processor (from TrIDIs v1 — same ViT preprocessing)
    print(f"Loading image processor from {IMAGE_PROCESSOR_PATH}...")
    processor = TrOCRProcessor.from_pretrained(IMAGE_PROCESSOR_PATH)

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_PATH))
    model.to(device)
    model.eval()
    print(f"  Model loaded ✅")

    # Extract line crops
    print(f"\nExtracting line crops from {args.test_dir}...")
    xml_files = sorted(args.test_dir.glob("*.xml"))
    print(f"  Found {len(xml_files)} XML files")

    flat_crops = []
    for xml_path in tqdm(xml_files, desc="Extracting"):
        image_path = xml_path.with_suffix('.jpg')
        if not image_path.exists():
            continue
        flat_crops.extend(extract_line_crops_from_xml(image_path, xml_path))
    print(f"  Total lines: {len(flat_crops)}")

    # Run inference
    predictions = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(flat_crops), args.batch_size), desc="Inference"):
            batch = flat_crops[i:i + args.batch_size]
            images = [crop[1] for crop in batch]
            line_ids = [crop[0] for crop in batch]

            pixel_values = processor(images, return_tensors="pt").pixel_values.to(device)

            generated_ids = model.generate(
                pixel_values,
                max_length=args.max_length,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                early_stopping=True,
            )

            # Decode with custom BPE tokenizer (NOT the TrOCR processor)
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for line_id, pred in zip(line_ids, preds):
                text = unicodedata.normalize('NFD', pred)
                text = fix_spacing(text)
                predictions[line_id] = text

    print(f"\n  Total predictions: {len(predictions)}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {args.output}")

    zip_path = args.output.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(args.output, arcname='predictions.json')
    print(f"  Zip: {zip_path}")

    # Samples
    print(f"\n  Sample predictions:")
    for line_id, text in list(predictions.items())[:5]:
        print(f"    {line_id}: {text[:70]}")


if __name__ == '__main__':
    main()