# 🧠 Intellia – Cognitive Model Training (Google Colab)

This notebook is used to **train the cognitive scoring models** that power Intellia.

In simple terms:  
👉 it teaches the system **how to measure how a person thinks**, using cognitive test data.

The models trained here are later used by the live app (`app.py`) to score students and adapt learning content for them.

This notebook is **offline**.  
You do **not** run it for every user.

---

## 🧩 What Problem This Solves

Cognitive test data is messy.

- Some tests are “higher is better”
- Some are “lower is better”
- Some abilities depend on **multiple test values**
- Real datasets are usually **small and noisy**

This notebook solves all of that by:
- Cleaning and normalizing the data
- Standardizing score direction
- Combining related test metrics into a single ability score
- Making the scoring system stable and reusable

The output is a set of **trained measurement models**, not predictions.

---

## 🧠 What the System Learns

The notebook trains models for **six cognitive abilities (pillars)**:

1. **Attention**
2. **Thinking Conversion**
3. **Information Processing Ability**
4. **Logical Reasoning**
5. **Representational Ability**
6. **Memory**

Each ability is learned **independently**.

That means:
- Each pillar has its own scaler
- Its own PCA model
- Its own reference distribution

This keeps the system modular and explainable.

---

## 📂 Input: What You Need to Run This

### Required
- One CSV file containing historical cognitive test scores

### Expected data format
The CSV should include columns such as:

- `tca_overall`, `tca_SwitchCost`, `tca_TaskInterference`
- `ipa_simpleRT`, `ipa_choiceRT`
- `attention_FlankerEffect`
- `lra_PercPers`, `lra_PercNonPers`
- `ra_RTCorrect`, `ra_PercCorrect`
- `ma_DigitSpan`

Not every column is used directly, but they should be present if the pillar depends on them.

---

## ⚙️ What Happens Inside the Notebook (Step by Step)

### 1️⃣ Load the Dataset
You upload a CSV file containing raw cognitive test data collected from users.

This data is assumed to be **real human test data**.

---

### 2️⃣ Clean the Data
The notebook:
- Removes columns that are not needed
- Fills missing numeric values using the column median
- Makes sure the dataset is consistent and usable

This step prevents garbage input from breaking the models.

---

### 3️⃣ Fix Score Direction (“Higher = Better”)
Not all cognitive tests work the same way.

Some scores mean:
- **higher = better** (e.g., accuracy)
- **lower = better** (e.g., reaction time)

To avoid confusion later, the notebook **inverts all “lower is better” columns** so that:

> Across the entire system, higher values always mean better performance.

This is critical for PCA and percentile scoring to behave correctly.

---

### 4️⃣ Generate Synthetic Data (TVAE)
Real cognitive datasets are often small.

To stabilize learning:
- A TVAE (Tabular Variational Autoencoder) is trained
- ~7,500 synthetic samples are generated
- Synthetic data is merged with real data

This does **not fake users**.  
It simply helps the model understand the overall structure of the data better.

---

### 5️⃣ Train the Cognitive Pillars
For each cognitive ability:

1. Relevant columns are selected
2. Features are standardized using `StandardScaler`
3. PCA is applied with **1 principal component**
4. PCA scores are collected
5. A sorted reference distribution is created

PCA is used here to:
- Combine related test scores
- Reduce noise
- Produce a clean, single cognitive score per ability

---

### 6️⃣ Export Model Artifacts
For each pillar, the notebook saves:

- `<pillar>_scaler.pkl`  
- `<pillar>_pca.pkl`  
- `<pillar>_pc_reference.pkl`  

These three files are bundled into a ZIP file and automatically downloaded.

Each ZIP is fully self-contained.

---

## 📦 Output: What You Get

After running the notebook, you will have **six ZIP files**, one for each pillar.

Example:
