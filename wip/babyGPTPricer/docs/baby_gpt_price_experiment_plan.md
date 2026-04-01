# Baby GPT Price Prediction Experiment (Minimal Plan)

## 🎯 Goal

Test whether a small GPT-style model can learn **text → price signal** from a constrained dataset.

Not aiming for accuracy or SOTA.
Just: does it learn *anything coherent*?

---

## 1. Scope (Reduce Variance)

- Select **one category** (e.g. Clothing, Books)
- Limit dataset to **~20k–50k samples**

Why:
- Reduce noise
- Increase chance of learnable signal

---

## 2. Task Definition (Make It Learnable)

Use **price buckets** instead of exact values.

Example buckets:

- 0–10
- 10–25
- 25–50
- 50–100
- 100+

---

## 3. Dataset Transformation

Convert each item into a consistent text format:

```
Predict the price range.

Description: {cleaned_text}
Answer: {bucket}
```

Notes:
- Keep formatting consistent
- No prompt variation initially

---

## 4. Model Setup

- Use existing baby GPT setup
- No architecture changes
- Train directly on transformed dataset

---

## 5. Evaluation (Keep It Simple)

- Bucket accuracy
- Or manual inspection of outputs

Looking for:
- Better than random
- Not completely collapsed output

---

## 6. Observations (Core Learning)

Watch for:

- Keyword associations (e.g. "premium", "bundle")
- Collapse to single bucket
- Output instability or randomness

Goal:
- Understand what signal the model can extract

---

## 7. Optional Next Step

Only if initial run shows signal:

- Try **rounded price prediction** instead of buckets
- Optionally add one extra field (e.g. category)

---

## 🚫 What NOT to Do (Yet)

- Do not mix multiple categories
- Do not optimize hyperparameters yet
- Do not introduce RAG / embeddings
- Do not aim to beat course results

---

## 🧠 Mental Model

You are not training a production model.

You are testing:

> Can a weak sequence model extract pricing signal from text?

---

## One-Line Plan

```
Pick category → bucket prices → format as text → train → observe
```

