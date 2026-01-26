def level(x):
    if x < 0.2: return "very_low"
    if x < 0.4: return "low"
    if x < 0.6: return "medium"
    if x < 0.8: return "high"
    return "very_high"


def get_adaptation(profile):
    rules = []

    if level(profile["information_processing_ability"]) in ["very_low", "low"]:
        rules.append("Use simple language and short sentences.")

    if level(profile["memory"]) in ["very_low", "low"]:
        rules.append("Repeat key ideas and add summaries.")

    if level(profile["attention"]) in ["very_low", "low"]:
        rules.append("Keep explanations short and structured.")

    if level(profile["logical_reasoning"]) in ["high", "very_high"]:
        rules.append("Include deeper explanations and why-logic.")

    if level(profile["representational_ability"]) in ["high", "very_high"]:
        rules.append("Encourage diagrams and visual reasoning.")

    return rules
