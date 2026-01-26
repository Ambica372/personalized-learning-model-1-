import joblib
import numpy as np
import pandas as pd
from adaptation_rules import get_adaptation

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

def score_pillar(raw, name):
    cols = PILLARS[name]

    scaler = joblib.load(f"models/{name}_scaler.pkl")
    pca = joblib.load(f"models/{name}_pca.pkl")
    ref = joblib.load(f"models/{name}_pc_reference.pkl")

    df = pd.DataFrame([raw])[cols]

    for c in cols:
        if c in LOWER_IS_BETTER:
            df[c] = -df[c]

    pc = pca.transform(scaler.transform(df))[0][0]
    return float(np.searchsorted(ref, pc) / len(ref))


def score_student(raw):
    profile = {}
    for p in PILLARS:
        profile[p] = score_pillar(raw, p)
    return profile


if __name__ == "__main__":
    student = {
        "tca_overall": 55,
        "tca_SwitchCost": 28,
        "tca_TaskInterference": 22,
        "ipa_simpleRT": 480,
        "ipa_choiceRT": 610,
        "lra_PercPers": 32,
        "lra_PercNonPers": 18,
        "attention_FlankerEffect": 58,
        "ra_RTCorrect": 520,
        "ra_PercCorrect": 84,
        "ma_DigitSpan": 6,
    }

    profile = score_student(student)
    rules = get_adaptation(profile)

    print("Cognitive Profile")
    for k, v in profile.items():
        print(k, ":", round(v, 3))

    print("\nAdaptation Rules")
    for r in rules:
        print("-", r)
