# Intellia 🧠 — Cognitive-Adaptive Learning System

Intellia is a Python-based backend system that measures **how a student thinks** and adapts learning content accordingly. Instead of predicting grades or labeling intelligence, Intellia converts raw cognitive test scores into **interpretable cognitive ability percentiles** and uses clear, explainable rules to decide *how* learning material should be presented.

The system focuses on **measurement and adaptation**, not prediction.

---

## What Intellia Does

Intellia takes raw cognitive test scores as input, converts them into six cognitive ability scores, normalizes those scores into percentiles (0.0–1.0), applies rule-based teaching logic, and outputs adaptation instructions that can be used to personalize learning content (text, audio, or AI-generated explanations).

There is no grading, no intelligence labeling, and no black-box AI involved in scoring or decision-making.

---

## How the System Works

The full pipeline is:

Raw Cognitive Test Scores → Preprocessing (direction fixing and scaling) → PCA-based cognitive ability measurement → Percentile scoring (0.0–1.0) → Rule-based teaching adaptation.

PCA is used strictly as a **measurement tool** to combine related test metrics and reduce noise. It is not used for prediction or classification.

---

## Cognitive Abilities (Pillars)

Each student is measured on six independent cognitive abilities:
1. Attention  
2. Thinking Conversion  
3. Information Processing Ability  
4. Logical Reasoning  
5. Representational Ability  
6. Memory  

Each pillar outputs a percentile score that represents relative performance within the trained reference distribution.

---

## Project Structure

The repository contains:
- `train_models.py`: Offline script for training cognitive measurement models  
- `app.py`: Runtime script for scoring students and generating adaptation logic  
- `adaptation_rules.py`: Deterministic teaching strategy rules  
- `models/`: Directory containing trained model artifacts (`.pkl` files)  
- `requirements.txt`: Python dependencies  
- `README.md`: Project documentation  

---

## Offline Training (`train_models.py`)

Offline training is responsible for building the cognitive measurement models. The script loads a cognitive dataset (`cognitive_dataset.csv`), cleans it, normalizes all metrics so that higher values always mean better performance, trains a separate PCA model for each cognitive pillar, and saves the resulting artifacts to the `models/` directory.

For each pillar, three files are generated:
- `<pillar>_scaler.pkl`
- `<pillar>_pca.pkl`
- `<pillar>_pc_reference.pkl`

This step is run once or whenever the training data or logic changes.

---

## Runtime Scoring and Adaptation (`app.py`)

At runtime, the system loads the pretrained model artifacts, accepts raw cognitive test scores for a student, computes percentile-based cognitive ability scores, applies adaptation rules, and outputs a cognitive profile along with teaching instructions. No training happens at runtime.

---

## Teaching Logic (`adaptation_rules.py`)

Teaching logic is rule-based and fully explainable. It maps cognitive ability levels (very low to very high) to instructional strategies. Examples include repeating key ideas for low memory, simplifying language for low processing speed, or providing deeper explanations for high reasoning ability. No machine learning is used in this layer.

---

## Requirements and Setup

The project requires Python 3.9 or higher. Install dependencies using `pip install -r requirements.txt`. Required libraries include numpy, pandas, scikit-learn, and joblib.

Typical usage:
1. Place the training dataset as `cognitive_dataset.csv`
2. Run `python train_models.py` to train and save models
3. Run `python app.py` to score students and generate adaptation instructions

---

## What This Project Is Not

Intellia is not a grading system, not an intelligence predictor, not supervised learning, and not a black-box AI system. AI models (such as LLMs) are intended only for content generation downstream, never for cognitive scoring or decision-making.

---

## One-Line Summary

Intellia measures how students think and adapts how learning content is explained — not what is taught.
