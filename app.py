import gradio as gr
import numpy as np
import joblib
import os

LOWER_IS_BETTER = {
    "tca_overall", "tca_SwitchCost", "tca_TaskInterference",
    "ipa_simpleRT", "ipa_choiceRT",
    "lra_PercPers", "lra_PercNonPers",
    "attention_FlankerEffect", "ra_RTCorrect"
}

PILLARS = {
    "thinking_conversion": ["tca_overall", "tca_SwitchCost", "tca_TaskInterference"],
    "information_processing_ability": ["ipa_simpleRT", "ipa_choiceRT"],
    "logical_reasoning": ["lra_PercPers", "lra_PercNonPers"],
    "attention": ["attention_FlankerEffect"],
    "representational_ability": ["ra_RTCorrect", "ra_PercCorrect"],
    "memory": ["ma_DigitSpan"]
}

def compute_scores(*inputs):
    input_dict = dict(zip(
        sum(PILLARS.values(), []),
        inputs
    ))

    results = {}

    for pillar, cols in PILLARS.items():
        X = np.array([input_dict[c] for c in cols]).reshape(1, -1)

        for i, c in enumerate(cols):
            if c in LOWER_IS_BETTER:
                X[0, i] = -X[0, i]

        scaler = joblib.load(f"models/{pillar}_scaler.pkl")
        pca = joblib.load(f"models/{pillar}_pca.pkl")
        pc_ref = joblib.load(f"models/{pillar}_pc_reference.pkl")

        Xs = scaler.transform(X)
        pc_value = pca.transform(Xs).flatten()[0]

        percentile = (pc_ref < pc_value).mean() * 100
        results[pillar] = f"{percentile:.2f}%"

    return "\n".join([f"{k}: {v}" for k, v in results.items()])


inputs = []

for cols in PILLARS.values():
    for col i
