import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata

# -----------------------------
# STEP 1: LOAD DATA (LOCAL FILE)
# -----------------------------
# Make sure this file exists in your repo
INPUT_FILE = "FINAL_COGNITIVE_DATASET (3).csv"

df_raw = pd.read_csv(INPUT_FILE)

print("Loaded dataset shape:", df_raw.shape)

# -----------------------------
# STEP 2: DATA CLEANING
# -----------------------------
cols_to_remove = [
    'ParticipantName', 'attention_RTCon', 'attention_RTInc',
    'tca_RepeatTrials', 'tca_SwitchTrials', 'tca_Congruent',
    'tca_Incongruent', 'lra_NumErrors', 'lra_PercErrors',
    'lra_NumPers', 'lra_NumNonPers', 'ra_PercCorrect'
]

df_cleaned = df_raw.drop(
    columns=[c for c in cols_to_remove if c in df_raw.columns],
    errors="ignore"
)

df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))

print("Cleaned dataset shape:", df_cleaned.shape)

# -----------------------------
# STEP 3: TVAE SYNTHESIS
# -----------------------------
print("Running TVAE synthesis...")

metadata = Metadata.detect_from_dataframe(
    data=df_cleaned,
    table_name="cognitive_data"
)

tvae = TVAESynthesizer(metadata, epochs=1000)
tvae.fit(df_cleaned)

synthetic_data = tvae.sample(num_rows=7500)

df_master = pd.concat([df_cleaned, synthetic_data], ignore_index=True)

print("Master dataset shape:", df_master.shape)

# -----------------------------
# STEP 4: PILLAR GENERATION
# -----------------------------
def get_locked_pillar(df, cols, name):
    existing_cols = [c for c in cols if c in df.columns]
    if not existing_cols:
        raise ValueError(f"No valid columns found for pillar: {name}")

    pca = PCA(n_components=1)
    scaled = StandardScaler().fit_transform(df[existing_cols])
    df[name] = pca.fit_transform(scaled)

    if df[existing_cols].corrwith(df[name]).mean() < 0:
        df[name] *= -1

    return df

print("Generating cognitive pillars...")

df_master = get_locked_pillar(
    df_master,
    ['tca_overall', 'tca_SwitchCost', 'tca_TaskInterference'],
    'Thinking conversion'
)

df_master = get_locked_pillar(
    df_master,
    ['ipa_simpleRT', 'ipa_choiceRT'],
    'Information processing ability'
)

df_master = get_locked_pillar(
    df_master,
    ['lra_PercPers', 'lra_PercNonPers'],
    'logical reasoning'
)

df_master = get_locked_pillar(
    df_master,
    ['attention_FlankerEffect'],
    'Attention'
)

df_master = get_locked_pillar(
    df_master,
    ['ra_RTCorrect'],
    'representational ability'
)

df_master = get_locked_pillar(
    df_master,
    ['ma_DigitSpan'],
    'memory'
)

# -----------------------------
# STEP 5: FINAL DATASET
# -----------------------------
pillar_cols = [
    'Attention',
    'Thinking conversion',
    'Information processing ability',
    'logical reasoning',
    'representational ability',
    'memory'
]

df_final = df_master[pillar_cols].copy()

scaler = MinMaxScaler()
df_final[pillar_cols] = scaler.fit_transform(df_final[pillar_cols])

# -----------------------------
# STEP 6: SAVE OUTPUT
# -----------------------------
OUTPUT_FILE = "FINAL_6_COL_LOCKED_DATASET.csv"
df_final.to_csv(OUTPUT_FILE, index=False)

print("=" * 45)
print("FINAL DATASET CREATED SUCCESSFULLY")
print("Rows:", len(df_final))
print("Columns:", pillar_cols)
print("Saved as:", OUTPUT_FILE)
print("=" * 45)
