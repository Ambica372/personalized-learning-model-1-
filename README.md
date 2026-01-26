# Intellia – Cognitive-Adaptive Learning System

Intellia is a system that measures **how a student thinks** and adapts learning content accordingly.

It converts raw cognitive test scores into cognitive ability profiles and uses those profiles to personalize text and audio learning content.

---
intellia/
├── train_models.py
├── app.py
├── adaptation_rules.py
├── requirements.txt
├── README.md
└── models/

## How the System Works (High Level)

Intellia works in **two separate stages**:

1. **Offline model training** (Google Colab)
2. **Online inference & content adaptation** (Streamlit app)

These stages are intentionally separated.

---

## 1. Offline Training (dataset_pca.ipynb)

**Purpose:**  
Train cognitive measurement models.

**What happens here:**
- Load historical cognitive test data (CSV)
- Clean data and fix score direction (higher = better)
- Generate synthetic data using TVAE (to stabilize patterns)
- Train PCA models for each cognitive ability
- Save trained model artifacts (`.pkl` files)

**Important:**
- No users involved
- No predictions
- No content generation
- Runs rarely (only when data or logic changes)

**Output (per cognitive pillar):**
- `<pillar>_scaler.pkl`
- `<pillar>_pca.pkl`
- `<pillar>_pc_reference.pkl`

These files are used later by the app.

---

## 2. Online Runtime (app.py)

**Purpose:**  
Score students and adapt learning content in real time.

**Runtime flow:**
1. Student enters cognitive test scores
2. Scores are preprocessed
3. Pretrained PCA models compute cognitive scores
4. Scores are converted to percentiles (0–1)
5. Percentiles are mapped to ability levels
6. Adaptation rules generate teaching instructions
7. AI rewrites learning content accordingly

**No training happens here.**

---

## Cognitive Abilities (Pillars)

Each student is measured on **six independent abilities**:

1. Attention  
2. Thinking Conversion  
3. Information Processing Ability  
4. Logical Reasoning  
5. Representational Ability  
6. Memory  

Each pillar produces a score between `0.0 – 1.0`.

---

## Why PCA Is Used

PCA is used as a **measurement tool**, not a prediction model.

It:
- Combines related test metrics
- Reduces noise
- Produces one stable score per ability

No labels. No supervision.

---

## Why Synthetic Data Exists

The real dataset is small.

Synthetic data (TVAE):
- Preserves real relationships
- Stabilizes PCA training
- Is **only used during training**

Synthetic data is **never used at runtime**.

---

## adaptation_rules.py

This file contains **teaching logic**, not machine learning.

It defines:
- How language complexity changes
- How repetition is added
- How reasoning depth is adjusted

Rules are:
- Deterministic
- Explainable
- Easy to modify

---

## What This Project Is NOT

- Not a grading system
- Not an intelligence predictor
- Not a black-box AI
- Not supervised learning

---

## Key Rule for Contributors

**Never mix these layers:**
- Measurement (PCA models)
- Teaching strategy (rules)
- Content generation (AI)

If they mix, the system breaks.

---

## Typical Setup

1. Run `dataset_pca.ipynb` in Google Colab
2. Download model artifact ZIP files
3. Place extracted `.pkl` files in `/models`
4. Run `app.py`

---

## One-Line Summary

Intellia measures how students think and uses that information to automatically adapt how learning content is explained.

---
