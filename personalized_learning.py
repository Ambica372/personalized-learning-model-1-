# ================================
# INTELLIA – OFFLINE MODEL TRAINING
# ================================

# --- STEP 0: INSTALL DEPENDENCIES ---
!pip install sdv joblib --quiet

import pandas as pd
import numpy as np
import joblib
import zipfile
import os

from google.colab import files
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata


# ================================
# STEP 1: LOAD RAW DATA
# ================================

print("Upload raw cognitive dataset (CSV)")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df_raw = pd.read_csv(file_name)


# ================================
# STEP 2: CLEANING & DIRECTION FIX
# ================================

cols_to_remove = [
    'ParticipantName', 'attention_RTCon', 'attention_RTInc',
    'tca_RepeatTrials', 'tca_SwitchTrials',
    'tca_Congruent', 'tca_Incongruent',
    'lra_NumErrors', 'lra_PercErrors',
    'lra_NumPers', 'lra_NumNonPers'
]

lower_is_better = [
    'attention_FlankerEffect',
    'tca_overall', 'tca_SwitchCost', 'tca_TaskInterference',
    'ipa_simpleRT', 'ipa_choiceRT',
    'lra_PercPers', 'lra_PercNonPers',
    'ra_RTCorrect'
]

df_raw[lower_is_better] = -df_raw[lower_is_better]

df_cleaned = df_raw.drop(
    columns=[c for c in cols_to_remove if c in df_raw.columns]
)

df_cleaned = df_cleaned.fillna(
    df_cleaned.median(numeric_only=True)
)


# ================================
# STEP 3: SYNTHETIC DATA (TVAE)
# ================================

print("Synthesizing data...")
metadata = Metadata.detect_from_dataframe(
    data=df_cleaned,
    table_name='cognitive_data'
)

tvae = TVAESynthesizer(metadata, epochs=1000)
tvae.fit(df_cleaned)
df_synth = tvae.sample(num_rows=7500)

df_master = pd.concat(
    [df_cleaned, df_synth],
    ignore_index=True
)


# ================================
# STEP 4: PILLAR TRAINING FUNCTION
# ================================

def train_pillar(df, cols, name):
    print(f"\nTraining pillar: {name}")

    X = df[cols]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=1)
    pc_scores = pca.fit_transform(Xs).flatten()

    # Percentile reference
    pc_sorted = np.sort(pc_scores)

    # Save artifacts
    joblib.dump(scaler, f"{name}_scaler.pkl")
    joblib.dump(pca, f"{name}_pca.pkl")
    joblib.dump(pc_sorted, f"{name}_pc_reference.pkl")

    # Zip artifacts
    zip_name = f"{name}_artifacts.zip"
    with zipfile.ZipFile(zip_name, "w") as z:
        z.write(f"{name}_scaler.pkl")
        z.write(f"{name}_pca.pkl")
        z.write(f"{name}_pc_reference.pkl")

    files.download(zip_name)

    # Cleanup
    os.remove(f"{name}_scaler.pkl")
    os.remove(f"{name}_pca.pkl")
    os.remove(f"{name}_pc_reference.pkl")


# ================================
# STEP 5: TRAIN ALL 6 PILLARS
# ================================

train_pillar(
    df_master,
    ['tca_overall', 'tca_SwitchCost', 'tca_TaskInterference'],
    'thinking_conversion'
)

train_pillar(
    df_master,
    ['ipa_simpleRT', 'ipa_choiceRT'],
    'information_processing_ability'
)

train_pillar(
    df_master,
    ['lra_PercPers', 'lra_PercNonPers'],
    'logical_reasoning'
)

train_pillar(
    df_master,
    ['attention_FlankerEffect'],
    'attention'
)

train_pillar(
    df_master,
    ['ra_RTCorrect', 'ra_PercCorrect'],
    'representational_ability'
)

train_pillar(
    df_master,
    ['ma_DigitSpan'],
    'memory'
)


# ================================
# DONE
# ================================

print("\nALL MODELS TRAINED AND DOWNLOADED.")
print("Use the ZIP files directly in app.py /models/")
