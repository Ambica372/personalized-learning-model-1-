"""
INTELLIA – SINGLE FILE REFERENCE IMPLEMENTATION

This script:
1. Trains cognitive PCA models (offline logic)
2. Saves model artifacts
3. Loads them
4. Scores a student
5. Applies adaptation rules
6. Outputs adapted learning instructions

No UI. No Streamlit. Pure logic.
"""

# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================
# CONFIG
# =========================

LOWER_IS_BETTER = {
    "tca_overall", "tca_SwitchCost", "tca_TaskInterference",
    "ipa_simpleRT", "ipa_choiceRT",
    "lra_PercPers", "lra_PercNonPers",
    "attention_FlankerEffect",
    "ra_RTCorrect"
}

PILLARS = {
    "thinking_conversion": ["tca_overall", "tca_SwitchCost", "tca_TaskInterference"],
    "information_processing": ["ipa_simpleRT", "ipa_choiceRT"],
    "logical_reasoning": ["lra_PercPers", "lra_PercNonPers"],
    "attention": ["attention_FlankerEffect"],
    "representational": ["ra_RTCorrect", "ra_PercCorrect"],
    "memory": ["ma_DigitSpan"]
}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# STEP 1 — TRAIN MODELS
# =========================

def train_models(df):
    print("Training cognitive models...")

    for pillar, cols in PILLARS.items():
        X = df[cols].copy()

        # Direction fix
        for c in cols:
            if c in LOWER_IS_BETTER:
                X[c] = -X[c]

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        pca = PCA(n_components=1)
        pc_scores = pca.fit_transform(Xs).flatten()

        pc_sorted = np.sort(pc_scores)

        joblib.dump(scaler, f"{MODEL_DIR}/{pillar}_scaler.pkl")
        joblib.dump(pca, f"{MODEL_DIR}/{pillar}_pca.pkl")
        joblib.dump(pc_sorted, f"{MODEL_DIR}/{pillar}_pc_reference.pkl")

    print("Training complete.\n")


# =========================
# STEP 2 — SCORING
# =========================

def score_pillar(raw, pillar):
    cols = PILLARS[pillar]

    scaler = joblib.load(f"{MODEL_DIR}/{pillar}_scaler.pkl")
    pca = joblib.load(f"{MODEL_DIR}/{pillar}_pca.pkl")
    pc_sorted = joblib.load(f"{MODEL_DIR}/{pillar}_pc_reference.pkl")

    df = pd.DataFrame([raw])[cols]

    for c in cols:
        if c in LOWER_IS_BETTER:
            df[c] = -df[c]

    Xs = scaler.transform(df)
    raw_pc = pca.transform(Xs)[0][0]

    percentile = np.searchsorted(pc_sorted, raw_pc, side="right") / len(pc_sorted)
    return round(float(percentile), 3)


def score_student(raw_scores):
    profile = {}
    for pillar in PILLARS:
        profile[pillar] = score_pillar(raw_scores, pillar)
    return profile


# =========================
# STEP 3 — ADAPTATION RULES
# =========================

def level(score):
    if score < 0.2: return "very_low"
    if score < 0.4: return "low"
    if score < 0.6: return "medium"
    if score < 0.8: return "high"
    return "very_high"


def generate_adaptation(profile):
    instructions = []

    if level(profile["information_processing"]) in ["very_low", "low"]:
        instructions.append("Use simple language and short sentences.")

    if level(profile["memory"]) in ["very_low", "low"]:
        instructions.append("Repeat key ideas and add summaries.")

    if level(profile["attention"]) in ["very_low", "low"]:
        instructions.append("Break content into short chunks.")

    if level(profile["logical_reasoning"]) in ["high", "very_high"]:
        instructions.append("Include deeper explanations and reasoning.")

    if level(profile["representational"]) in ["high", "very_high"]:
        instructions.append("Encourage diagrams or mental visuals.")

    return instructions


# =========================
# STEP 4 — DEMO RUN
# =========================

if __name__ == "__main__":

    # ---------- FAKE TRAINING DATA ----------
    np.random.seed(42)
    train_df = pd.DataFrame({
        "tca_overall": np.random.normal(50, 10, 120),
        "tca_SwitchCost": np.random.normal(30, 5, 120),
        "tca_TaskInterference": np.random.normal(25, 5, 120),
        "ipa_simpleRT": np.random.normal(400, 50, 120),
        "ipa_choiceRT": np.random.normal(550, 60, 120),
        "lra_PercPers": np.random.uniform(0, 40, 120),
        "lra_PercNonPers": np.random.uniform(0, 30, 120),
        "attention_FlankerEffect": np.random.normal(45, 10, 120),
        "ra_RTCorrect": np.random.normal(500, 80, 120),
        "ra_PercCorrect": np.random.uniform(70, 100, 120),
        "ma_DigitSpan": np.random.randint(3, 9, 120),
    })

    train_models(train_df)

    # ---------- SAMPLE STUDENT ----------
    student = {
        "tca_overall": 55,
        "tca_SwitchCost": 28,
        "tca_TaskInterference": 22,
        "ipa_simpleRT": 480,
        "ipa_choiceRT": 620,
        "lra_PercPers": 35,
        "lra_PercNonPers": 20,
        "attention_FlankerEffect": 60,
        "ra_RTCorrect": 520,
        "ra_PercCorrect": 85,
        "ma_DigitSpan": 6,
    }

    profile = score_student(student)
    instructions = generate_adaptation(profile)

    print("COGNITIVE PROFILE")
    for k, v in profile.items():
        print(f"- {k}: {v}")

    print("\nADAPTATION INSTRUCTIONS")
    for i in instructions:
        print("-", i)
