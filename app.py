import gradio as gr
import numpy as np
import joblib

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
    all_columns = sum(PILLARS.values(), [])
    input_dict = dict(zip(all_columns, inputs))

    results = {}

    for pillar, cols in PILLARS.items():
        # Prepare input vector
        X = np.array([input_dict[c] for c in cols]).reshape(1, -1)

        # Reverse metrics where lower is better
        for i, c in enumerate(cols):
            if c in LOWER_IS_BETTER:
                X[0, i] = -X[0, i]

        # Load trained components
        scaler = joblib.load(f"models/{pillar}_scaler.pkl")
        pca = joblib.load(f"models/{pillar}_pca.pkl")
        pc_ref = joblib.load(f"models/{pillar}_pc_reference.pkl")

        # Transform
        X_scaled = scaler.transform(X)
        pc_value = pca.transform(X_scaled).flatten()[0]

        # Percentile calculation
        percentile = (pc_ref < pc_value).mean() * 100

        results[pillar] = f"{percentile:.2f}%"

    output_text = "\n".join([f"{k}: {v}" for k, v in results.items()])
    return output_text


# Create input fields
inputs = []
for cols in PILLARS.values():
    for col in cols:
        inputs.append(gr.Number(label=col))

interface = gr.Interface(
    fn=compute_scores,
    inputs=inputs,
    outputs="text",
    title="Cognitive Pillar Scoring Engine",
    description="Input cognitive test values to compute percentile scores."
)

if __name__ == "__main__":
    interface.launch()
