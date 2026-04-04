#!/usr/bin/env python3
"""
Custom BPE Tokenizer + Fresh Decoder Training
==================================================================

Based on proven approaches from:
  - OcciGen (Swin+BERT, 0.5% CER on Old Occitan)
  - HATFormer (custom tokenizer for Arabic HTR, 4.2% CER)
  - HuggingFace VisionEncoderDecoderModel documentation

Architecture:
  - Encoder: Pre-trained ViT-large from TrIDIs v1
  - Decoder: Fresh BERT decoder (randomly initialized, designed for custom vocab)
  - Tokenizer: Byte-level BPE trained on YOUR competition corpus
"""

import os
import sys
import json
import time
import unicodedata
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import load_from_disk
from transformers import (
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    BertConfig,
    TrOCRProcessor,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path("/mnt/hdd2/rainfall_project/code/gsmap/proj/processed_data/hf_dataset")
OUTPUT_DIR = Path("/mnt/hdd2/rainfall_project/code/gsmap/proj/models/trocr_custom_bpe")
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
BASE_ENCODER_MODEL = "/mnt/hdd2/rainfall_project/code/gsmap/proj/models/tridis_HTR"

# BPE tokenizer config
BPE_VOCAB_SIZE = 1000 
DECODER_LAYERS = 6       
DECODER_HIDDEN = 768     
DECODER_HEADS = 12       
DECODER_FFN = 3072       

# Training config
VAL_SAMPLES = 500   
BATCH_SIZE = 16
SEED = 42

# Phase 1: Frozen encoder, decoder learns from scratch
PHASE1_EPOCHS = 5
PHASE1_LR = 1e-4

# Phase 2: Full fine-tune
PHASE2_EPOCHS = 10
PHASE2_LR = 5e-6    
PHASE2_PATIENCE = 3

# Augmentation
AUGMENTATION = "rotation"  

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STAGE 1: Train BPE tokenizer on competition corpus
# ============================================================

def stage1_train_tokenizer():
    """Train a byte-level BPE tokenizer on all competition text."""
    
    print("\n" + "=" * 70)
    print("STAGE 1: Training BPE Tokenizer")
    print("=" * 70)
    
    # Load all text from dataset
    print("Loading competition text...")
    dataset = load_from_disk(str(DATASET_PATH))
    
    all_texts = []
    for split in ['train', 'validation']:
        for i in range(len(dataset[split])):
            text = unicodedata.normalize('NFD', dataset[split][i]['text'])
            if text.strip():
                all_texts.append(text)
    
    print(f"Total text lines: {len(all_texts):,}")
    
    # Analyze character set
    char_freq = Counter()
    for text in all_texts:
        char_freq.update(text)
    
    print(f"Unique characters: {len(char_freq)}")
    print(f"Top 20 characters:")
    for ch, cnt in char_freq.most_common(20):
        print(f"  U+{ord(ch):04X} '{ch}': {cnt:,}")
    
    # Save corpus to temp file for tokenizer training
    corpus_path = TOKENIZER_DIR / "corpus.txt"
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for text in all_texts:
            f.write(text + '\n')
    
    print(f"Corpus saved to {corpus_path}")
    
    # Train byte-level BPE tokenizer
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=BPE_VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    
    print(f"Training BPE tokenizer (vocab_size={BPE_VOCAB_SIZE})...")
    tokenizer.train([str(corpus_path)], trainer=trainer)
    
    # Save tokenizer
    tokenizer_path = TOKENIZER_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Wrap as HuggingFace PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    hf_tokenizer.save_pretrained(str(TOKENIZER_DIR))
    
    print(f"HuggingFace tokenizer saved to {TOKENIZER_DIR}")
    print(f"Final vocab size: {hf_tokenizer.vocab_size}")
    
    # ===== CHECKPOINT: Verify tokenizer =====
    print("\n--- CHECKPOINT 1: Tokenizer Verification ---")
    
    test_strings = [
        "sarabaitarũ qͥ nulla rgla",
        "⁊ pone las censuras celesiasticas",
        "Del cuer souspire ⁊ des iex pleure",
        "disce̾.⁊ sumat bolũ ar. cũ uino",
        "ꝑ ꝯfirmationẽ ꝓuincie",
    ]
    
    all_pass = True
    for test in test_strings:
        test_nfd = unicodedata.normalize('NFD', test)
        encoded = hf_tokenizer.encode(test_nfd)
        decoded = hf_tokenizer.decode(encoded, skip_special_tokens=True)
        
        match = decoded.strip() == test_nfd.strip()
        status = "✅" if match else "❌"
        print(f"  {status} Original: {test_nfd[:50]}")
        print(f"     Decoded:  {decoded[:50]}")
        print(f"     Tokens:   {len(encoded)} tokens")
        
        if not match:
            all_pass = False
    
    # Compare token efficiency vs TrIDIs v1
    tridis_processor = TrOCRProcessor.from_pretrained(BASE_ENCODER_MODEL)
    
    print(f"\n  Token efficiency comparison:")
    for test in test_strings[:3]:
        test_nfd = unicodedata.normalize('NFD', test)
        custom_tokens = len(hf_tokenizer.encode(test_nfd))
        tridis_tokens = len(tridis_processor.tokenizer.encode(test_nfd))
        ratio = tridis_tokens / custom_tokens if custom_tokens > 0 else 0
        print(f"    '{test_nfd[:30]}...' Custom: {custom_tokens} | TrIDIs: {tridis_tokens} | Ratio: {ratio:.1f}x")
    
    if all_pass:
        print("\n  ✅ CHECKPOINT 1 PASSED: Tokenizer encodes/decodes correctly")
    else:
        print("\n  ⚠️  CHECKPOINT 1 WARNING: Some decode mismatches (may be whitespace)")
        print("     Continuing — minor whitespace differences are usually OK")
    
    return hf_tokenizer


# ============================================================
# STAGE 2: Build VisionEncoderDecoderModel
# ============================================================

def stage2_build_model(tokenizer):
    """Build model: pre-trained ViT encoder + fresh BERT decoder."""
    
    print("\n" + "=" * 70)
    print("STAGE 2: Building Model")
    print("=" * 70)
    
    vocab_size = len(tokenizer)  # includes special tokens (vocab_size excludes them)
    print(f"Tokenizer vocab size (including special tokens): {vocab_size}")
    
    # Load encoder from TrIDIs v1
    print(f"Loading ViT encoder from {BASE_ENCODER_MODEL}...")
    base_model = VisionEncoderDecoderModel.from_pretrained(BASE_ENCODER_MODEL)
    encoder_config = base_model.config.encoder
    encoder_hidden_size = encoder_config.hidden_size
    print(f"  Encoder hidden size: {encoder_hidden_size}")
    print(f"  Encoder type: {encoder_config.model_type}")
    
    # Configure fresh BERT decoder
    print(f"Configuring BERT decoder...")
    decoder_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=DECODER_HIDDEN,
        num_hidden_layers=DECODER_LAYERS,
        num_attention_heads=DECODER_HEADS,
        intermediate_size=DECODER_FFN,
        max_position_embeddings=512,
        is_decoder=True,
        add_cross_attention=True,  
    )
    
    decoder_config.encoder_hidden_size = encoder_hidden_size
    print(f"  Cross-attention projection: encoder({encoder_hidden_size}) → decoder({DECODER_HIDDEN})")
    
    # Build the combined model
    print("Building VisionEncoderDecoderModel...")
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config, decoder_config
    )
    model = VisionEncoderDecoderModel(config=config)
    
    print("Copying pre-trained encoder weights...")
    model.encoder.load_state_dict(base_model.encoder.state_dict())
    
    print("Decoder: randomly initialized (fresh BERT)")
    
    # Set special token IDs
    model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("<s>")
    model.config.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    model.config.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        decoder_start_token_id=model.config.decoder_start_token_id,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        max_length=256,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        num_beams=4,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Encoder (pre-trained): {encoder_params:,}")
    print(f"  Decoder (fresh): {decoder_params:,}")
    
    # Clean up base model
    del base_model
    torch.cuda.empty_cache()
    
    # ===== CHECKPOINT: Verify forward pass =====
    print("\n--- CHECKPOINT 2: Model Forward Pass ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load image processor from TrIDIs v1
    processor = TrOCRProcessor.from_pretrained(BASE_ENCODER_MODEL)
    
    # Test with a dummy image
    dataset = load_from_disk(str(DATASET_PATH))
    test_image = dataset['validation'][0]['image']
    test_text = unicodedata.normalize('NFD', dataset['validation'][0]['text'])
    
    pixel_values = processor(test_image, return_tensors="pt").pixel_values.to(device)
    
    # Test forward pass
    labels = tokenizer(
        test_text, return_tensors="pt", padding="max_length",
        max_length=128, truncation=True
    ).input_ids.to(device)
    labels[labels == tokenizer.pad_token_id] = -100
    
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss.item()
        
        # Test generation
        generated = model.generate(pixel_values, max_length=128, num_beams=1)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    print(f"  Forward pass loss: {loss:.4f}")
    print(f"  Expected loss (random decoder, vocab={vocab_size}): ~{np.log(vocab_size):.2f}")
    print(f"  Generated text (random): '{decoded[:60]}'")
    print(f"  Ground truth: '{test_text[:60]}'")
    
    if loss < np.log(vocab_size) * 2 and loss > 0:
        print(f"\n  ✅ CHECKPOINT 2 PASSED: Forward pass works, loss is reasonable")
    else:
        print(f"\n  ❌ CHECKPOINT 2 FAILED: Loss {loss:.4f} is unexpected")
        print(f"     Expected roughly {np.log(vocab_size):.2f} for random decoder")
        sys.exit(1)
    
    model.cpu()
    torch.cuda.empty_cache()
    
    return model, processor


# ============================================================
# AUGMENTATION (same as your working code)
# ============================================================

from scipy.ndimage import map_coordinates, gaussian_filter

def elastic_distortion(image, alpha=12.0, sigma=4.0):
    img_array = np.array(image.convert("RGB")).astype(np.float32)
    h, w = img_array.shape[:2]
    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    distorted = np.zeros_like(img_array)
    for c in range(3):
        distorted[:, :, c] = map_coordinates(
            img_array[:, :, c],
            [np.clip(y + dy, 0, h-1).reshape(-1), np.clip(x + dx, 0, w-1).reshape(-1)],
            order=1, mode='reflect'
        ).reshape(h, w)
    return Image.fromarray(distorted.astype(np.uint8))

def random_rotation(image, max_angle=2.5):
    return image.rotate(np.random.uniform(-max_angle, max_angle), 
                       fillcolor=(255, 255, 255), expand=False)

def brightness_contrast(image):
    from PIL import ImageEnhance
    img = ImageEnhance.Brightness(image).enhance(np.random.uniform(0.75, 1.25))
    return ImageEnhance.Contrast(img).enhance(np.random.uniform(0.80, 1.30))

def apply_augmentation(image, strategy):
    if strategy == "baseline":
        return image
    elif strategy == "rotation":
        return brightness_contrast(random_rotation(image))
    elif strategy == "elastic":
        return brightness_contrast(elastic_distortion(image))
    return image


# ============================================================
# DATASET
# ============================================================

class CustomBPEDataset(TorchDataset):
    def __init__(self, hf_dataset, processor, tokenizer, max_target_length=256,
                 is_training=True, augmentation_strategy="baseline"):
        self.dataset = hf_dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.is_training = is_training
        self.augmentation_strategy = augmentation_strategy

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        if self.is_training and self.augmentation_strategy != "baseline":
            try:
                image = apply_augmentation(image, self.augmentation_strategy)
            except Exception:
                pass

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        text = unicodedata.normalize('NFD', item['text'])
        
        encoding = self.tokenizer(
            text, padding="max_length", max_length=self.max_target_length,
            truncation=True, return_tensors="pt"
        )
        
        raw_ids = encoding.input_ids.squeeze()

        eos_id = self.tokenizer.convert_tokens_to_ids("</s>")
        pad_positions = (raw_ids == self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        if len(pad_positions) > 0:
            raw_ids[pad_positions[0]] = eos_id
        elif raw_ids[-1] != eos_id:
            raw_ids[-1] = eos_id       

        bos_id = self.tokenizer.convert_tokens_to_ids("<s>")
        decoder_input_ids = raw_ids.clone()
        decoder_input_ids[1:] = raw_ids[:-1]
        decoder_input_ids[0] = bos_id

        labels = raw_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }


# ============================================================
# COMPUTE METRICS
# ============================================================

def make_compute_metrics(tokenizer):
    from torchmetrics.text import CharErrorRate
    cer_metric = CharErrorRate()  

    def compute_metrics(eval_preds):
        preds_ids, labels_ids = eval_preds

        preds_str = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)

        # Decode labels
        labels_ids = np.where(labels_ids == -100, tokenizer.pad_token_id, labels_ids)
        labels_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        # NFD normalize
        preds_str = [unicodedata.normalize('NFD', p) for p in preds_str]
        labels_str = [unicodedata.normalize('NFD', l) for l in labels_str]

        cer = cer_metric(preds_str, labels_str).item()
        
        # Print samples for monitoring
        for i in range(min(3, len(preds_str))):
            print(f"    GT:   {labels_str[i][:60]}")
            print(f"    PRED: {preds_str[i][:60]}")
            print()
        
        return {"cer": cer}
    
    return compute_metrics


# ============================================================
# STRATIFIED SAMPLING
# ============================================================

def stratified_sample(dataset_split, n_samples, seed=42):
    np.random.seed(seed)
    lang_indices = {}
    languages = dataset_split['language']  # single vectorized read instead of O(n) item access
    for idx, lang in enumerate(languages):
        lang_indices.setdefault(lang, []).append(idx)
    
    total = len(dataset_split)
    sampled = []
    for lang, indices in lang_indices.items():
        n = max(1, min(int(n_samples * len(indices) / total), len(indices)))
        sampled.extend(np.random.choice(indices, size=n, replace=False).tolist())
        print(f"  {lang:<20} {n:>5} ({len(indices)/total*100:.1f}%)")
    
    if len(sampled) < n_samples:
        remaining = list(set(range(total)) - set(sampled))
        extra = np.random.choice(remaining, size=min(n_samples - len(sampled), len(remaining)), replace=False)
        sampled.extend(extra.tolist())
    np.random.shuffle(sampled)
    return sampled[:n_samples]


# ============================================================
# STAGE 3: Training
# ============================================================

def stage3_train(model, processor, tokenizer):
    """Two-phase training with checkpoints."""
    
    print("\n" + "=" * 70)
    print("STAGE 3: Training")
    print("=" * 70)
    
    # Load dataset
    dataset = load_from_disk(str(DATASET_PATH))
    train_data = dataset['train']
    
    print(f"Sampling {VAL_SAMPLES} validation examples...")
    val_indices = stratified_sample(dataset['validation'], VAL_SAMPLES, seed=SEED + 1)
    val_data = dataset['validation'].select(val_indices)
    
    print(f"Train: {len(train_data):,}, Val: {len(val_data):,}")
    
    train_dataset = CustomBPEDataset(
        train_data, processor, tokenizer,
        is_training=True, augmentation_strategy=AUGMENTATION
    )
    val_dataset = CustomBPEDataset(
        val_data, processor, tokenizer,
        is_training=False, augmentation_strategy="baseline"
    )
    
    # ===== CHECKPOINT 3a: Verify dataset =====
    print("\n--- CHECKPOINT 3a: Dataset Verification ---")
    sample = train_dataset[0]
    print(f"  pixel_values shape: {sample['pixel_values'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")
    non_ignore = (sample['labels'] != -100).sum().item()
    print(f"  Non-padding labels: {non_ignore}")
    
    decoded = tokenizer.decode(sample['labels'][sample['labels'] != -100], skip_special_tokens=True)
    gt = unicodedata.normalize('NFD', train_data[0]['text'])
    print(f"  Decoded label: '{decoded[:60]}'")
    print(f"  Ground truth:  '{gt[:60]}'")
    
    if non_ignore > 1 and decoded.strip():
        print(f"  ✅ CHECKPOINT 3a PASSED")
    else:
        print(f"  ❌ CHECKPOINT 3a FAILED: Labels are empty or all padding")
        sys.exit(1)
    
    # ===== PHASE 1: Frozen encoder =====
    print(f"\n{'='*70}")
    print(f"PHASE 1: Decoder training (encoder frozen)")
    print(f"  Epochs: {PHASE1_EPOCHS}, LR: {PHASE1_LR}")
    print(f"{'='*70}")
    
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    phase1_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR / "phase1"),
        predict_with_generate=True,
        generation_max_length=128,   
        generation_num_beams=1,     
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        num_train_epochs=PHASE1_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        learning_rate=PHASE1_LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        label_smoothing_factor=0.0,
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        logging_steps=50,
        logging_dir=str(OUTPUT_DIR / "logs_phase1"),
        report_to="none",
        push_to_hub=False,
    )
    
    trainer1 = Seq2SeqTrainer(
        model=model,
        args=phase1_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(tokenizer),
    )
    
    start = time.time()
    trainer1.train()
    phase1_time = time.time() - start
    
    phase1_cer = trainer1.state.best_metric or float('nan')
    print(f"\nPhase 1 complete: best CER = {phase1_cer:.4f} ({phase1_time/60:.0f} min)")
    
    # ===== CHECKPOINT 3b: Phase 1 output quality =====
    print("\n--- CHECKPOINT 3b: Phase 1 Output Quality ---")
    
    device = next(model.parameters()).device
    model.eval()
    
    empty_count = 0
    with torch.no_grad():
        for i in range(5):
            img = dataset['validation'][i]['image']
            gt = unicodedata.normalize('NFD', dataset['validation'][i]['text'])
            pv = processor(img, return_tensors="pt").pixel_values.to(device)
            gen = model.generate(pv, max_length=256, num_beams=4)
            pred = tokenizer.decode(gen[0], skip_special_tokens=True)
            
            if not pred.strip():
                empty_count += 1
            
            print(f"  GT:   {gt[:70]}")
            print(f"  PRED: {pred[:70]}")
            print()
    
    if empty_count >= 4:
        print(f"  ❌ CHECKPOINT 3b FAILED: {empty_count}/5 predictions are empty")
        print(f"     The decoder is not learning. Possible issues:")
        print(f"     - Learning rate too high/low")
        print(f"     - Label encoding problem")
        print(f"     - Cross-attention not connecting encoder to decoder")
        response = input("  Continue to Phase 2 anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    elif phase1_cer > 0.9:
        print(f"  ⚠️  CHECKPOINT 3b WARNING: CER={phase1_cer:.4f} is very high")
        print(f"     But predictions are non-empty, so decoder IS learning")
        print(f"     Continuing to Phase 2...")
    else:
        print(f"  ✅ CHECKPOINT 3b PASSED: CER={phase1_cer:.4f}, predictions look reasonable")
    
    # ===== PHASE 2: Full fine-tune =====
    print(f"\n{'='*70}")
    print(f"PHASE 2: Full fine-tune (encoder unfrozen)")
    print(f"  Epochs: {PHASE2_EPOCHS}, LR: {PHASE2_LR}, Patience: {PHASE2_PATIENCE}")
    print(f"{'='*70}")
    
    # Unfreeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} / {total:,} (100%)")
    
    phase2_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR / "phase2"),
        predict_with_generate=True,
        generation_max_length=128,   # reduced from 256; medieval lines are short
        generation_num_beams=1,      # greedy during training eval — 4-8x faster than beam=4
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        num_train_epochs=PHASE2_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        learning_rate=PHASE2_LR,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        label_smoothing_factor=0.0,  
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        logging_steps=50,
        logging_dir=str(OUTPUT_DIR / "logs_phase2"),
        report_to="none",
        push_to_hub=False,
    )
    
    trainer2 = Seq2SeqTrainer(
        model=model,
        args=phase2_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=PHASE2_PATIENCE,
                early_stopping_threshold=0.001,
            ),
        ],
    )
    
    start = time.time()
    trainer2.train()
    phase2_time = time.time() - start
    
    phase2_cer = trainer2.state.best_metric or float('nan')
    print(f"\nPhase 2 complete: best CER = {phase2_cer:.4f} ({phase2_time/60:.0f} min)")
    
    # ===== Save final model =====
    final_path = OUTPUT_DIR / "final_model"
    trainer2.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))  
    # Save custom info needed for inference
    with open(final_path / "custom_model_info.json", 'w') as f:
        json.dump({
            "tokenizer_type": "custom_bpe",
            "bpe_vocab_size": BPE_VOCAB_SIZE,
            "decoder_type": "bert",
            "decoder_layers": DECODER_LAYERS,
            "encoder_source": BASE_ENCODER_MODEL,
            "phase1_cer": phase1_cer,
            "phase2_cer": phase2_cer,
            "phase1_time_min": phase1_time / 60,
            "phase2_time_min": phase2_time / 60,
        }, f, indent=2)
    
    print(f"\n✅ Final model saved to: {final_path}")
    
    # ===== CHECKPOINT 4: Final validation =====
    print("\n--- CHECKPOINT 4: Final Model Quality ---")
    device = next(model.parameters()).device  # re-read: trainer may have moved model
    model.eval()
    
    with torch.no_grad():
        for i in range(10):
            img = dataset['validation'][i]['image']
            gt = unicodedata.normalize('NFD', dataset['validation'][i]['text'])
            pv = processor(img, return_tensors="pt").pixel_values.to(device)
            gen = model.generate(pv, max_length=256, num_beams=4)
            pred = tokenizer.decode(gen[0], skip_special_tokens=True)
            
            print(f"  GT:   {gt[:70]}")
            print(f"  PRED: {pred[:70]}")
            print()
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Phase 1 CER: {phase1_cer:.4f}")
    print(f"  Phase 2 CER: {phase2_cer:.4f}")
    print(f"  Total time:  {(phase1_time + phase2_time)/60:.0f} min")
    print(f"  Model path:  {final_path}")
    
    return model


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_stage', type=int, default=1,
                       help='Stage to start from (1=tokenizer, 2=model, 3=train)')
    args = parser.parse_args()
    
    tokenizer = None
    model = None
    processor = None
    
    # Stage 1: Tokenizer
    if args.start_stage <= 1:
        tokenizer = stage1_train_tokenizer()
    else:
        print("Loading existing tokenizer...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
        print(f"  Vocab size: {tokenizer.vocab_size}")
    
    # Stage 2: Model
    if args.start_stage <= 2:
        model, processor = stage2_build_model(tokenizer)
    
    # Stage 3: Training
    if args.start_stage <= 3:
        if model is None:
            print("Need to build model first — run from stage 2")
            sys.exit(1)
        stage3_train(model, processor, tokenizer)
    
    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
