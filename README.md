
These files are REQUIRED for runtime inference.

---

## 8. Online Runtime (app.py)

📍 Runs locally or on server  
📍 This is what users interact with

### Runtime flow
1. User enters cognitive test scores
2. Scores are preprocessed (direction fixed again)
3. PCA models compute pillar scores
4. Scores are converted to percentiles (0–1)
5. Percentiles are mapped to cognitive levels
6. Content is adapted using rules
7. AI generates personalized output

No training happens here.

---

## 9. adaptation_rules.py (Critical File)

This file contains **teaching logic**, not ML.

It answers:
- How should content change if memory is low?
- How complex should language be if processing speed is low?
- How much reasoning depth is appropriate?

Rules are:
- deterministic
- explainable
- easy to audit

Do NOT replace this with ML unless you know exactly why.

---

## 10. AI Usage (What It Actually Does)

AI is used ONLY for:
- rewriting content
- generating explanations
- producing podcasts

AI does NOT:
- score students
- decide abilities
- classify cognition

AI obeys rules set by the cognitive profile.

---

## 11. Cognitive Index (Optional)

There is an optional composite score (CI).

- Derived AFTER pillar scoring
- Used for analytics or research
- NOT required for adaptation
- Never drives decisions directly

---

## 12. What NOT to Do (Hard Rules)

❌ Do NOT retrain models inside app.py  
❌ Do NOT use supervised learning  
❌ Do NOT hardcode thresholds inside PCA logic  
❌ Do NOT mix training and inference  
❌ Do NOT guess cognitive meaning from AI output  

If you break these, the system becomes unreliable.

---

## 13. Mental Model (If You Remember One Thing)

Intellia is:

> A measuring system + teaching rules + AI writing engine

NOT:

> A prediction model or black-box AI

---

## 14. Where to Start as a New Contributor

Recommended order:
1. Read this file
2. Understand dataset_pca.ipynb
3. Trace app.py from input → output
4. Read adaptation_rules.py
5. Only then touch logic

---

## 15. Final Reality Check

If you change something, ask:
- Does this affect measurement?
- Does this affect teaching strategy?
- Does this affect content generation?

Never mix the three.

---

END OF DOCUMENT
