# Intellia

Intellia is a cognitive-adaptive learning system.

It converts raw cognitive test scores into cognitive ability percentiles and uses those scores to adapt learning content.

## Files

- `train_models.py` – trains PCA-based cognitive models
- `app.py` – scores a student using trained models
- `adaptation_rules.py` – rule-based teaching logic
- `models/` – trained model artifacts

## Usage

1. Place cognitive dataset as `cognitive_dataset.csv`
2. Run:
