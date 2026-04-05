# Medieval Handwriting Recognition: ICDAR 2026 CMMHWR26

**Multi-architecture ensemble for medieval manuscript text recognition combining Vision Transformers, custom BPE tokenizers, and CNN+CTC models via ROVER character-level alignment.**

Built for the [ICDAR 2026 Competition on Multilingual Medieval Handwriting Recognition](https://cmmhwr26.inria.fr/).

[![Competition](https://img.shields.io/badge/ICDAR%202026-CMMHWR26-blue)](https://cmmhwr26.inria.fr/)

---

## Results

| Task       | Description                           | CER       | WER   |
| ---------- | ------------------------------------- | --------- | ----- |
| **Task 1** | Multilingual (French, Latin, Spanish) | **9.09%** | 30.5% |
| **Task 2** | Zero-shot transfer to Occitan         | **9.56%** | 35.8% |
| **Task 3** | Cross-family transfer to Czech        | **27.2%** | 80.4% |

**Task 1 per-language breakdown:**

| Language            | CER    | WER   | Training Lines |
| ------------------- | ------ | ----- | -------------- |
| French              | 3.78%  | 16.4% | 58K            |
| Latin               | 13.96% | 44.9% | 48K            |
| Spanish (Castilian) | 9.54%  | 30.2% | 34K            |

---

## System Architecture

The system ensembles three structurally different HTR architectures. The ensemble's effectiveness comes from architectural diversity, not just model diversity. Each architecture makes fundamentally different types of errors.

```
              ┌─────────────────────────────────┐
              │     Input Manuscript Page       │
              └────────────────┬────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  TrOCR x5    │  │  Custom BPE  │  │   CATMuS     │
    │  ViT-large + │  │  ViT-large + │  │   Kraken     │
    │  RoBERTa     │  │ BERT (fresh) │  │ CNN+LSTM+CTC │
    │              │  │              │  │              │
    │  + TTA (x3)  │  │  1000-token  │  │  Pre-trained │
    │  15 virtual  │  │ BPE tokenizer│  │  113K lines  │
    │  models      │  │  trained on  │  │  graphematic │
    │              │  │  competition │  │ transcription│
    │  50K BPE     │  │  corpus      │  │              │
    │  (RoBERTa)   │  │              │  │  NFD norm    │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │
           └─────────────────┬─────────────────┘
                             ▼
    ┌──────────────────────────────────────────────────┐
    │          ROVER Character-Level Alignment         │
    │                                                  │
    │  Edit-distance alignment between predictions     │
    │  Per-character majority voting                   │
    │  Safety guards: length, word count, similarity   │
    │  Fallback to primary when models disagree >20%   │
    └──────────────────────┬───────────────────────────┘
                           ▼
                   ┌────────────────┐
                   │Final Prediction│
                   └────────────────┘
```

### Why three architectures?

| Component        | Architecture                       | Tokenizer                | Strengths                                        |
| ---------------- | ---------------------------------- | ------------------------ | ------------------------------------------------ |
| TrOCR (5 models) | ViT encoder + RoBERTa decoder      | 50K BPE (generic)        | Strong baseline, pre-trained on medieval text    |
| Custom BPE       | ViT encoder + BERT decoder (fresh) | 1K BPE (domain-specific) | Medieval characters encoded efficiently          |
| CATMuS Kraken    | CNN + LSTM + CTC                   | Character-level          | Different error patterns from Transformer models |

---

## Key Technical Contributions

**1. Custom BPE tokenizer for medieval text**

Medieval manuscripts contain rare Unicode characters (⁊ Tironian et, ꝑ per, ꝯ con, combining marks like ̃ ̾ ̧) that generic tokenizers fragment into 2-3 byte tokens. We trained a 1000-token byte-level BPE tokenizer on the competition corpus, paired with a fresh 6-layer BERT decoder. The pre-trained ViT encoder was kept frozen initially, then fine-tuned with a low learning rate.

**2. Guarded ROVER ensemble**

Standard ROVER merges predictions character-by-character via edit-distance alignment. When source predictions are too dissimilar, this creates "Frankenstein" outputs. Our guarded variant only merges when predictions are within 20% edit distance, rejects merges that grow output length or change word count, and falls back to the primary model when guards trigger.

**3. Cross-architecture ensemble for zero-shot transfer**

For Task 3 (unknown language, later identified as medieval Czech), the CNN+CTC architecture (CATMuS Kraken) outperformed all Transformer models despite never seeing Czech during training, suggesting CTC-based models generalize better to unseen scripts than autoregressive decoders.

**4. Test-Time Augmentation**

Each TrOCR model processes 3 versions of each line image (original + rotation + brightness), creating 15 virtual models from 5 trained models. Majority voting across TTA copies improved CER by 0.5% absolute.

---

## Repository Structure

```
├── README.md
├── data/
│   ├── 00-eda.ipynb
│   └── 01-data-pipeline.ipynb
├── training/
│   ├── 02-train-trocr-augmented.py      # TrOCR with 5 augmentation variants
│   └── 03-train-custom-bpe.py           # BPE tokenizer training + ViT+BERT model
├── inference/
│   ├── 04-tta-beam-speacing-inference   # TrOCR ensemble with test-time augmentation
│   ├── 05-custom-bpe-inference.py       # Custom BPE model inference
│   └── 06-catmus-inference.py           # CATMuS Kraken full-page inference
└── ensemble/
    └── 07-rover-ensemble.py             # Guarded ROVER character-level alignment

```

---

## Getting Started

### Requirements

```bash
pip install transformers torch datasets torchmetrics python-Levenshtein
pip install kraken  # for CATMuS Kraken model
```

### Training

```bash
# Train TrOCR augmentation variants (rotation, elastic, blur, etc.)
python training/train_trocr_augmented.py --strategy rotation

# Train custom BPE model (Phase 1: frozen encoder, Phase 2: full fine-tune)
python training/train_custom_bpe.py
```

### Inference and Ensemble

```bash
# Step 1: TrOCR with TTA
python inference/tta_inference.py --test_dir /path/to/test \
    --model_paths model1 model2 model3 model4 model5 \
    --output predictions_tta.json --n_augmentations 3

# Step 2: Custom BPE
python inference/infer_custom_bpe.py --test_dir /path/to/test \
    --output predictions_bpe.json

# Step 3: CATMuS Kraken
python inference/06-catmus-inference.py

# Step 4: ROVER ensemble
python ensemble/07-rover-ensemble.py
```

---

## Acknowledgments

- [TrIDIs](https://huggingface.co/magistermilitum/tridis_HTR) — base TrOCR model for medieval manuscripts
- [CATMuS Medieval](https://zenodo.org/records/12743230) — Kraken HTR model by the CREMMA collaboration
- [OcciGen](https://github.com/EstebanGarces/OcciGen) — inspiration for custom BPE approach
- [Kraken](https://kraken.re/) OCR engine by Benjamin Kiessling

## License

MIT License
