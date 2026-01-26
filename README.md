# Intellia 🧠  
*Cognitive-Adaptive Learning System*

Intellia is a Python-based system that measures **how a student thinks** and adapts learning content based on that cognitive profile.

Instead of predicting grades or labeling intelligence, Intellia converts raw cognitive test scores into **interpretable cognitive ability percentiles** and uses them to guide how content should be explained.

---

## 🚀 What This Project Does

- Takes raw cognitive test scores as input
- Converts them into **6 cognitive ability scores**
- Uses explainable rules to decide *how* content should be taught
- Outputs adaptation instructions (ready for AI-generated content)

This repository contains the **complete backend logic** for cognitive measurement and adaptation.

---

## 🧩 Core Concept

**Pipeline:**

Raw Test Scores
↓
Cognitive Ability Measurement (PCA)
↓
Percentile Scoring (0–1)
↓
Rule-Based Teaching Strategy
↓
Adapted Learning Instructions


No black-box AI is used for scoring or decision-making.

---

## 🧠 Cognitive Abilities (Pillars)

Each student is measured on **six independent abilities**:

1. **Attention**
2. **Thinking Conversion**
3. **Information Processing Ability**
4. **Logical Reasoning**
5. **Representational Ability**
6. **Memory**

Each pillar outputs a value between `0.0` and `1.0`, representing relative performance within the trained reference distribution.

---

## 🏗 Project Structure

intellia/
├── train_models.py # Offline training script
├── app.py # Runtime scoring & adaptation
├── adaptation_rules.py # Teaching strategy logic
├── requirements.txt
├── README.md
└── models/ # Saved model artifacts (.pkl)


---

## 🔹 Offline Training (`train_models.py`)

This script trains the cognitive measurement models.

### What it does
- Loads a cognitive test dataset (`cognitive_dataset.csv`)
- Normalizes all metrics so **higher = better**
- Trains a PCA model for each cognitive pillar
- Saves reusable model artifacts to `/models`

### Output files (per pillar)
- `<pillar>_scaler.pkl`
- `<pillar>_pca.pkl`
- `<pillar>_pc_reference.pkl`

These artifacts are required for runtime scoring.

### How to run
```bash
python train_models.py
🔹 Runtime Scoring & Adaptation (app.py)

This script:

Loads pretrained model artifacts

Scores a student’s cognitive abilities

Converts scores into percentiles

Applies adaptation rules

Prints the cognitive profile and teaching strategy

How to run
python app.py

🔹 Teaching Logic (adaptation_rules.py)

This file contains deterministic, explainable rules that define how learning content should change based on cognitive strengths and weaknesses.

Examples:

Low memory → repeat key ideas

Low processing speed → simpler language

High reasoning → deeper explanations

No machine learning is used here.

📦 Requirements

Install dependencies using:

pip install -r requirements.txt


Dependencies:

Python 3.9+

numpy

pandas

scikit-learn

joblib

❌ What This Project Is NOT

Not a grading system

Not an intelligence predictor

Not supervised learning

Not a black-box AI system

AI (e.g. LLMs) are intended only for content generation, not scoring.

🧠 Design Principles

Explainability over prediction

Measurement over labeling

Rules over opaque models

Separation of concerns:

Measurement

Teaching strategy

Content generation

📌 Typical Workflow

Prepare cognitive_dataset.csv

Run train_models.py

Place generated .pkl files in /models

Run app.py to score students and generate adaptation logic

🧾 One-Line Summary

Intellia measures how students think and adapts how learning content is explained — not what is taught.
