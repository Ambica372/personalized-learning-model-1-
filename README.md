# Personalized Learning Model – Cognitive Dataset Generation

## Overview
This project focuses on personalized learning by processing learner-related data to generate a structured cognitive profile. The system transforms raw learner inputs into a final 6-column cognitive dataset that captures key cognitive abilities required for adaptive and personalized content delivery.

The generated dataset serves as the **input foundation for a second-stage model**, where personalized educational content is generated based on the learner’s cognitive profile.

---

## Objective
The primary goal of this project is to:
- Analyze learner data
- Derive standardized cognitive indicators
- Produce a clean, locked, and structured cognitive dataset
- Enable downstream content generation and adaptive learning systems

---

## Cognitive Dimensions Generated
The final dataset contains **six core cognitive attributes**, representing different aspects of learner capability:

1. **Attention**  
2. **Thinking Conversion**  
3. **Information Processing Ability**  
4. **Logical Reasoning**  
5. **Representational Ability**  
6. **Memory**

Each row represents an individual learner profile encoded numerically across these six dimensions.

---

## Dataset
### Input
- Raw learner data capturing behavioral, cognitive, or performance-related signals

### Output
- Final generated dataset:  
  **`FINAL_6_COL_LOCKED_DATASET.csv`**

This dataset is normalized, validated, and locked to ensure consistency and reliability when used by downstream models.

---

## Methodology
1. Load and validate raw learner dataset  
2. Feature extraction and transformation  
3. Cognitive pillar computation  
4. Data normalization and consistency checks  
5. Generation of final 6-column cognitive dataset  
6. Export of dataset for reuse in content generation models  

---

## Role in Multi-Model Pipeline
This project represents **Stage 1** of a multi-model system:

- **Stage 1 – Personalized Learning Model (this project):**  
  Generates a structured cognitive profile for each learner.

- **Stage 2 – Adaptive Content Generation Model:**  
  Uses the 6-column cognitive dataset to dynamically generate personalized learning content, difficulty levels, and presentation styles.

This separation ensures modularity, scalability, and reusability of learner intelligence across systems.

---

## Technologies Used
- Python  
- Pandas  
- NumPy  

---

## How to Run the Project

1. Install required dependencies:

pip install -r requirements.txt


2. Run the script:



3. Output:
- The final 6-column cognitive dataset is generated and saved as:



---

## Output Summary
- Clean, structured learner cognitive profiles
- Standardized numerical representation of learner abilities
- Dataset ready for content generation and adaptive learning models

---

## Applications
- Personalized and adaptive learning systems  
- AI-driven content generation  
- Cognitive profiling for education platforms  
- Learner-centered curriculum design  

---

## Author
Ambica Natraj
