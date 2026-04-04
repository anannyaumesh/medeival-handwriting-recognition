"""
ICDAR 2026 CMMHWR - Augmentation Training Script
Based on: arxiv 2508.11499 (Meoded et al. 2025)
Key finding: Elastic Distortion + Random Rotation are the two best single augmentations
             Top-5 ensemble (Elastic, Rotation, Underline, Gaussian Blur, Baseline) = best overall
"""

import os
import json
import unicodedata
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import load_from_disk
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from scipy.ndimage import map_coordinates, gaussian_filter
from torchmetrics.text import CharErrorRate, WordErrorRate
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# ============================================================
# CELL 1: CONFIGURATION
# ============================================================

DATASET_PATH = Path("/mnt/hdd2/rainfall_project/code/gsmap/proj/processed_data/hf_dataset")
BASE_OUTPUT_DIR = Path("/mnt/hdd2/rainfall_project/code/gsmap/proj/models")
PREDICTIONS_DIR = Path("/mnt/hdd2/rainfall_project/code/gsmap/proj/predictions")
MODEL_NAME = "/mnt/hdd2/rainfall_project/code/gsmap/proj/models/tridis_HTR"

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SAMPLES = 148225       # Scale up from 30K
VAL_SAMPLES   = 3000
NUM_EPOCHS    = 8
BATCH_SIZE    = 20
SEED          = 42

# The 5 augmentation strategies from the paper
AUGMENTATION_STRATEGIES = [
    "baseline",        # No augmentation as one ensemble member
    "elastic",         # Best single augmentation in paper
    "rotation",        # Tied best with elastic
    "gaussian_blur",   # In paper's top-5 ensemble
    "underline",       # In paper's top-5 ensemble (simulates ink strikethrough)
    "e9_weighted",
]

RUN_STRATEGY = "baseline"   

# ============================================================
# CELL 2: AUGMENTATION FUNCTIONS
# ============================================================

def elastic_distortion(image: Image.Image, alpha: float = 12.0, sigma: float = 4.0) -> Image.Image:
    """
    Elastic distortion - best single augmentation per arxiv 2508.11499.
    Simulates natural ink/parchment warping in medieval manuscripts.
    alpha: controls distortion intensity (12-15 works well for line images)
    sigma: controls smoothness of distortion (3-5 recommended)
    """
    img_array = np.array(image.convert("RGB")).astype(np.float32)
    h, w = img_array.shape[:2]

    # Generate random displacement fields
    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha

    # Create coordinate grids
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices_x = np.clip(x + dx, 0, w - 1).reshape(-1)
    indices_y = np.clip(y + dy, 0, h - 1).reshape(-1)

    # Apply distortion to each channel
    distorted = np.zeros_like(img_array)
    for c in range(3):
        distorted[:, :, c] = map_coordinates(
            img_array[:, :, c],
            [indices_y, indices_x],
            order=1,
            mode='reflect'
        ).reshape(h, w)

    return Image.fromarray(distorted.astype(np.uint8))


def random_rotation(image: Image.Image, max_angle: float = 2.5) -> Image.Image:
    """
    Random rotation ±max_angle degrees.
    Tied with elastic as best single augmentation in paper.
    2-3 degrees is optimal — larger angles distort medieval script too much.
    """
    angle = np.random.uniform(-max_angle, max_angle)
    return image.rotate(angle, fillcolor=(255, 255, 255), expand=False)


def gaussian_blur(image: Image.Image, radius_range=(0.5, 1.5)) -> Image.Image:
    """
    Subtle Gaussian blur to simulate ink bleed / scanning softness.
    In top-5 ensemble per paper.
    """
    radius = np.random.uniform(*radius_range)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def underline_augmentation(image: Image.Image, p: float = 0.3) -> Image.Image:
    """
    Randomly adds underline strokes to simulate manuscript annotation marks.
    Novel augmentation designed specifically for historical manuscripts in paper.
    p: probability of adding underline to any given line image
    """
    if np.random.random() > p:
        return image
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # Underline at 75-90% of image height (near baseline)
    y_pos = int(h * np.random.uniform(0.75, 0.90))
    x_start = np.random.randint(0, w // 6)
    x_end = np.random.randint(5 * w // 6, w)
    line_width = np.random.randint(1, 3)
    draw.line([(x_start, y_pos), (x_end, y_pos)], fill=(0, 0, 0), width=line_width)
    return img


def brightness_contrast(image: Image.Image) -> Image.Image:
    """
    Random brightness/contrast variation.
    Simulates uneven illumination, ink fading, and scanning artifacts.
    """
    from PIL import ImageEnhance
    img = image
    # Brightness: 0.75-1.25
    img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.75, 1.25))
    # Contrast: 0.8-1.3
    img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.80, 1.30))
    return img


def apply_augmentation(image: Image.Image, strategy: str) -> Image.Image:
    """Apply the chosen augmentation strategy to a PIL image."""
    if strategy == "baseline":
        return image
    elif strategy == "elastic":
        image = elastic_distortion(image, alpha=12.0, sigma=4.0)
        image = brightness_contrast(image)
        return image
    elif strategy == "rotation":
        image = random_rotation(image, max_angle=2.5)
        image = brightness_contrast(image)
        return image
    elif strategy == "gaussian_blur":
        image = gaussian_blur(image, radius_range=(0.5, 1.5))
        image = brightness_contrast(image)
        return image
    elif strategy == "underline":
        image = underline_augmentation(image, p=0.4)
        image = random_rotation(image, max_angle=1.0)  # also slight rotation
        image = brightness_contrast(image)
        return image
    else:
        return image


# ============================================================
# CELL 3: DATASET CLASS (uses working TrOCRDataset pattern)
# ============================================================

class MedievalHTRDataset(TorchDataset):
    """
    Dataset class that applies augmentation on-the-fly.
    Uses the working TrOCRDataset pattern (NOT dataset.map which corrupts medieval chars).
    """
    def __init__(self, hf_dataset, processor, max_target_length=128,
                 is_training=True, augmentation_strategy="baseline"):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_target_length = max_target_length
        self.is_training = is_training
        self.augmentation_strategy = augmentation_strategy

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']

        # Apply augmentation only during training
        if self.is_training and self.augmentation_strategy != "baseline":
            try:
                image = apply_augmentation(image, self.augmentation_strategy)
            except Exception:
                pass  # If augmentation fails, use original

        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Normalize text to NFD
        text = unicodedata.normalize('NFD', item['text'])

        # Tokenize
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        # Replace padding token id with -100 (ignored in loss)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ============================================================
# CELL 4: STRATIFIED SAMPLING
# ============================================================

def stratified_sample(dataset_split, n_samples: int, seed: int = 42) -> list:
    """
    Sample n_samples indices with stratification by language.
    Ensures underrepresented languages (Venetian, Gallician, Navarrese)
    get proportional representation rather than being drowned out.
    """
    np.random.seed(seed)

    # Group indices by language
    language_indices = {}
    for idx in range(len(dataset_split)):
        lang = dataset_split[idx]['language']
        if lang not in language_indices:
            language_indices[lang] = []
        language_indices[lang].append(idx)

    total = len(dataset_split)
    sampled = []

    for lang, indices in language_indices.items():
        # Proportional allocation
        lang_proportion = len(indices) / total
        lang_n = max(1, int(n_samples * lang_proportion))
        lang_n = min(lang_n, len(indices))  # Can't sample more than available

        chosen = np.random.choice(indices, size=lang_n, replace=False)
        sampled.extend(chosen.tolist())
        print(f"  {lang:<15} {lang_n:>6} samples ({lang_proportion*100:.1f}% of data)")

    # If we're short due to rounding, fill with random
    if len(sampled) < n_samples:
        remaining_pool = list(set(range(total)) - set(sampled))
        extra = np.random.choice(remaining_pool, size=min(n_samples - len(sampled), len(remaining_pool)), replace=False)
        sampled.extend(extra.tolist())

    np.random.shuffle(sampled)
    return sampled[:n_samples]


# ============================================================
# CELL 5: TRAINING FUNCTION
# ============================================================

def train_model(strategy: str):
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL: {strategy.upper()} augmentation")
    print(f"{'='*70}")

    output_dir = BASE_OUTPUT_DIR / f"trocr_{strategy}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    print("\nLoading dataset...")
    dataset = load_from_disk(str(DATASET_PATH))
    print(f"Full dataset — Train: {len(dataset['train']):,}, Val: {len(dataset['validation']):,}")

    # --- Stratified sampling ---
    print(f"\nStratified sampling {TRAIN_SAMPLES:,} training examples...")
    train_indices = stratified_sample(dataset['train'], TRAIN_SAMPLES, seed=SEED)
    train_subset = dataset['train'].select(train_indices)

    print(f"\nStratified sampling {VAL_SAMPLES:,} val examples...")
    val_indices = stratified_sample(dataset['validation'], VAL_SAMPLES, seed=SEED + 1)
    val_subset = dataset['validation'].select(val_indices)

    print(f"\nFinal — Train: {len(train_subset):,}, Val: {len(val_subset):,}")

    # --- Load model & processor ---
    print(f"\nLoading TRIDIS model...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # Configure generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = 128
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4

    # --- Create PyTorch datasets ---
    train_dataset = MedievalHTRDataset(
        train_subset, processor, is_training=True, augmentation_strategy=strategy
    )
    val_dataset = MedievalHTRDataset(
        val_subset, processor, is_training=False, augmentation_strategy="baseline"
    )

    # --- Training arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        predict_with_generate=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        num_train_epochs=NUM_EPOCHS,

        # Evaluation — skip trainer.evaluate() to avoid multi-GPU OOM
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,

        # Optimizer
        learning_rate=5e-05,
        weight_decay=0.01,
        warmup_steps=500,
        lr_scheduler_type="cosine",  # Cosine decay — slightly better than linear

        # Performance
        fp16=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,

        # Logging
        logging_steps=100,
        logging_dir=str(output_dir / "logs"),
        report_to="none",

        # Multi-GPU
        ddp_find_unused_parameters=False,

        load_best_model_at_end=False,
        push_to_hub=False,
    )

    # --- Train ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    start = time.time()
    print(f"\nStarting training ({strategy} augmentation)...")
    trainer.train()
    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    # --- Save model ---
    model.save_pretrained(output_dir / "final_model")
    processor.save_pretrained(output_dir / "final_model")
    print(f"Model saved to: {output_dir / 'final_model'}")

    # --- Generate predictions (single model) ---
    print("\nGenerating predictions on validation set...")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    predictions = {}
    ground_truth = {}

    def collate_fn(batch):
        images   = [item['image'] for item in batch]
        line_ids = [item['line_id'] for item in batch]
        texts    = [unicodedata.normalize('NFD', item['text']) for item in batch]
        pixel_values = processor(images, return_tensors="pt").pixel_values
        return pixel_values, line_ids, texts

    loader = DataLoader(val_subset, batch_size=16, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    with torch.no_grad():
        for pixel_values, line_ids, texts in tqdm(loader, desc="Generating predictions"):
            pixel_values = pixel_values.to(device)
            generated_ids = model.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for lid, pred, text in zip(line_ids, preds, texts):
                predictions[lid] = pred
                ground_truth[lid] = text

    # --- Calculate CER ---
    cer_metric = CharErrorRate()
    wer_metric = WordErrorRate()

    for lid in predictions:
        if lid in ground_truth:
            cer_metric(predictions[lid], ground_truth[lid])
            wer_metric(predictions[lid], ground_truth[lid])

    final_cer = cer_metric.compute().item()
    final_wer = wer_metric.compute().item()

    print(f"\n{'='*50}")
    print(f"RESULTS ({strategy})")
    print(f"{'='*50}")
    print(f"CER: {final_cer:.4f} ({final_cer*100:.2f}%)")
    print(f"WER: {final_wer:.4f} ({final_wer*100:.2f}%)")
    print(f"{'='*50}")

    # Save predictions
    pred_file = PREDICTIONS_DIR / f"predictions_{strategy}.json"
    # Save ground truth
    gt_file = PREDICTIONS_DIR / f"ground_truth_{strategy}.json"
    with open(gt_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        "strategy": strategy,
        "model": MODEL_NAME,
        "train_samples": len(train_subset),
        "val_samples": len(val_subset),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "final_cer": final_cer,
        "final_wer": final_wer,
        "training_time_minutes": elapsed / 60,
    }
    with open(PREDICTIONS_DIR / f"summary_{strategy}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return predictions, ground_truth, final_cer


# ============================================================
# CELL 6: ENSEMBLE INFERENCE
# ============================================================

def ensemble_predictions(strategy_list: list, val_ground_truth: dict = None):
    """
    Top-5 voting ensemble as described in arxiv 2508.11499.
    Loads saved prediction files and votes per line.
    """
    print(f"\n{'='*70}")
    print("ENSEMBLE INFERENCE")
    print(f"Combining: {strategy_list}")
    print(f"{'='*70}")

    all_preds = {}
    for strategy in strategy_list:
        pred_file = PREDICTIONS_DIR / f"predictions_{strategy}.json"
        if not pred_file.exists():
            print(f"  WARNING: {pred_file} not found, skipping")
            continue
        with open(pred_file, encoding='utf-8') as f:
            all_preds[strategy] = json.load(f)
        print(f"  Loaded {len(all_preds[strategy]):,} predictions from {strategy}")

    if not all_preds:
        print("No prediction files found!")
        return {}

    # Get common line IDs across all models
    line_ids = set(list(all_preds.values())[0].keys())
    for preds in all_preds.values():
        line_ids &= set(preds.keys())
    print(f"\nVoting on {len(line_ids):,} lines...")

    # Vote: pick the prediction that appears most often
    # Tie-break: prefer elastic > rotation > gaussian_blur > underline > baseline
    tiebreak_order = ["elastic", "rotation", "gaussian_blur", "underline", "baseline"]

    ensemble_preds = {}
    for lid in line_ids:
        votes = {}
        for strategy, preds in all_preds.items():
            pred = preds.get(lid, "")
            votes[pred] = votes.get(pred, 0) + 1

        # Get prediction with most votes
        max_votes = max(votes.values())
        candidates = [p for p, v in votes.items() if v == max_votes]

        if len(candidates) == 1:
            ensemble_preds[lid] = candidates[0]
        else:
            # Tie-break: use the candidate from the highest-ranked strategy
            for strategy in tiebreak_order:
                if strategy in all_preds:
                    candidate = all_preds[strategy].get(lid, "")
                    if candidate in candidates:
                        ensemble_preds[lid] = candidate
                        break
            else:
                ensemble_preds[lid] = candidates[0]

    # Save ensemble predictions
    ensemble_file = PREDICTIONS_DIR / "predictions_ensemble.json"
    with open(ensemble_file, 'w', encoding='utf-8') as f:
        json.dump(ensemble_preds, f, indent=2, ensure_ascii=False)
    print(f"Ensemble predictions saved to: {ensemble_file}")

    # Calculate ensemble CER if ground truth available
    if val_ground_truth:
        cer_metric = CharErrorRate()
        wer_metric = WordErrorRate()
        for lid in ensemble_preds:
            if lid in val_ground_truth:
                cer_metric(ensemble_preds[lid], val_ground_truth[lid])
                wer_metric(ensemble_preds[lid], val_ground_truth[lid])
        ens_cer = cer_metric.compute().item()
        ens_wer = wer_metric.compute().item()
        print(f"\nEnsemble CER: {ens_cer*100:.2f}%")
        print(f"Ensemble WER: {ens_wer*100:.2f}%")

    return ensemble_preds


# ============================================================
# CELL 7: MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if RUN_STRATEGY == "ALL":
        # Train all 5 models sequentially
        gt = None
        for strategy in AUGMENTATION_STRATEGIES:
            preds, gt, cer = train_model(strategy)
            print(f"\n>>> {strategy}: CER = {cer*100:.2f}%\n")

        # After all models done, run ensemble
        if gt is not None:
            ensemble_predictions(AUGMENTATION_STRATEGIES, val_ground_truth=gt)

    else:
        # Train single strategy
        preds, gt, cer = train_model(RUN_STRATEGY)

        # If all prediction files exist, also run ensemble
        existing = [s for s in AUGMENTATION_STRATEGIES
                    if (PREDICTIONS_DIR / f"predictions_{s}.json").exists()]
        if len(existing) >= 2:
            print(f"\nFound {len(existing)} prediction files — running ensemble...")
            ensemble_predictions(existing, val_ground_truth=gt)
        else:
            print(f"\nOnly {len(existing)} prediction file(s) found.")
            print("Run more strategies then call ensemble_predictions() to combine them.")

    print("\n=== ALL DONE ===")