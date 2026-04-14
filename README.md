# Odonata Classification — Dragonfly vs Damselfly
### End-to-End ML Pipeline | Wadi Wurayah Field Research Project

---

## Overview
An end-to-end ML pipeline that classifies dragonflies and damselflies from images.
Trained on iNaturalist research-grade observations and validated against UAE wildlife
data collected during ecological fieldwork at Wadi Wurayah National Park, Fujairah.

---

## Motivation
Dragonflies and damselflies belong to the same order (Odonata) and are visually
similar — distinguishing them requires identifying subtle morphological features
like wing position, body thickness, and eye arrangement. Manual classification
is time-consuming for ecologists processing hundreds of field photos.

---

## Dataset
- Source: iNaturalist — Dragonflies and Damselflies of Sonoma & Marin Counties
- Size: 817 images total (450 dragonfly, 367 damselfly)
- Dragonfly species: Flame Skimmer, Cardinal Meadowhawk, Blue Dasher
- Damselfly species: Vivid Dancer, Pacific Forktail, Tule Bluet
- Data collection: pyinaturalist API, research-grade observations only

---

## Model Architecture
- Base: MobileNetV2 pretrained on ImageNet (frozen)
- Added layers: GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(1, sigmoid)
- Approach: Transfer Learning
- Framework: TensorFlow/Keras
- Training: Google Colab T4 GPU

---

## Key Finding — Color Jitter Augmentation

### The Problem
V1 trained on a single species per class achieved 97.5% validation accuracy
but failed on UAE species — misclassifying Violet Dropwing as Damselfly.

Root cause analysis revealed the model learned **regional color patterns**
rather than morphological features. Trained only on California species,
it associated orange/red coloring with dragonflies specifically.

### The Fix
V2 introduced two changes:
1. **Multiple species per class** — 3 dragonfly + 3 damselfly species
2. **Color jitter augmentation** — randomly shifts brightness and color
   channels during training, forcing the model to ignore color and focus
   on body shape and wing structure instead
```python
# Color jitter in ImageDataGenerator
brightness_range=[0.6, 1.4],  # random brightness
channel_shift_range=50.0       # random color shift
```

---

## Results

### Model Versions
| Version | Val Accuracy | Key Change |
|---------|-------------|------------|
| V1 | 97.5% | Single species, no augmentation |
| V2 | 85.3% | Multi-species + color jitter |

Note: Lower V2 validation accuracy is expected and intentional —
the model is solving a genuinely harder problem and generalizes better.

### Cross-Regional Testing
| Test Set | Correct | Accuracy | Notes |
|----------|---------|----------|-------|
| iNaturalist UAE observations | 6/6 | 100% | Research grade photos |
| Wadi Wurayah field photos | 6/9 | 67% | Mixed distance/clarity |

### UAE Species Successfully Identified
- Arabicnemis caerulea (Arabian Bluet) — endemic Gulf damselfly ✅
- Trithemis arteriosa (Red-veined Dropwing) — common UAE dragonfly ✅
- Crocothemis erythraea (Little Scarlet) — common UAE dragonfly ✅
- Trithemis species (Violet Dropwing) — UAE dragonfly ✅

---

## Field Test Analysis
Model was validated against 9 original field photographs taken during
ecological survey at Wadi Wurayah National Park, UAE.

**Correct (6/9):** All close-up photos classified correctly
**Incorrect (3/9):** Distant photos where insect occupies <5% of frame

This is consistent with human visual limitations — distant small subjects
are genuinely difficult to classify regardless of method.

---

## Limitations & Future Work (V3)
Current model struggles with:
- Distant field photos where insect is small in frame
- Heavily occluded subjects

Proposed V3 solution:
- Integrate object detection (YOLOv8 fine-tuned on insects or iNaturalist Vision API)
- Automatically detect and crop insect before classification
- Expected improvement: field photo accuracy from 67% → 90%+

---

## Project Structure
```
odonata_mlops/
├── models/          ← trained model weights (odonata_v2.keras)
├── data/
│   └── field_test/  ← original Wadi Wurayah field photographs
├── notebooks/       ← training notebooks
├── app/             ← Flask API (coming soon)
├── logs/            ← prediction logs
└── README.md
```

---

## Tech Stack
- Python 3.12
- TensorFlow 2.x / Keras
- MobileNetV2 (transfer learning)
- pyinaturalist API
- Google Colab T4 GPU

---

## About the Field Data
Field photographs were collected during an ecological biodiversity survey
at Wadi Wurayah National Park, Fujairah, UAE — the first mountain national
park in the UAE and an important freshwater ecosystem supporting multiple
Odonata species endemic to the Arabian Peninsula.