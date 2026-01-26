import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

os.makedirs("models", exist_ok=True)

def train(csv_path):
    df = pd.read_csv(csv_path)
    df = df.fillna(df.median(numeric_only=True))

    for pillar, cols in PILLARS.items():
        X = df[cols].copy()

        for c in cols:
            if c in LOWER_IS_BETTER:
                X[c] = -X[c]

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        pca = PCA(n_components=1)
        pc = pca.fit_transform(Xs).flatten()
        pc_sorted = np.sort(pc)

        joblib.dump(scaler, f"models/{pillar}_scaler.pkl")
        joblib.dump(pca, f"models/{pillar}_pca.pkl")
        joblib.dump(pc_sorted, f"models/{pillar}_pc_reference.pkl")

    print("Models trained and saved.")

if __name__ == "__main__":
    train("cognitive_dataset.csv")
