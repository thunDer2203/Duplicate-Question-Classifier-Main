# Duplicate Question Classifier using BERT

A Natural Language Processing project that detects whether two questions are duplicates using a **fine-tuned BERT model** trained on the Quora Question Pairs dataset.

This project focuses entirely on **BERT-based transfer learning** for semantic similarity classification. Unlike traditional approaches that rely on keyword overlap or handcrafted features, BERT understands context and meaning, making it highly effective for duplicate question detection.

---

## 🚀 Use Case

Question-answer platforms like **Quora**, **Stack Overflow**, community forums, and customer support systems often receive the same question phrased in different ways.

### Example

| Question 1 | Question 2 |
|------------|------------|
| How can I learn Python quickly? | What is the fastest way to master Python? |

Even though the wording is different, the **intent is the same**.

This project helps platforms:
- Detect duplicate questions automatically
- Merge redundant threads
- Improve search relevance
- Reduce content clutter
- Improve overall user experience

---

## 🤖 Why BERT?

Traditional NLP methods often struggle when two sentences have different wording, synonyms, reordered structure, or indirect phrasing. **BERT solves this by understanding language contextually.**

### How BERT Improves Performance

**1. Contextual Embeddings**

Older models assign one fixed meaning per word. BERT understands words based on surrounding context.

| Phrase | Meaning of "bank" |
|--------|-------------------|
| bank account | Financial institution |
| river bank | Side of a river |

BERT distinguishes between these automatically.

**2. Bidirectional Understanding**

BERT reads text in both directions simultaneously (Left → Right and Right ← Left), giving it a stronger understanding than older one-directional models.

**3. Sentence Pair Learning**

BERT is purpose-built for comparing two sentences together. Input format:

```text
[CLS] Question 1 [SEP] Question 2 [SEP]
```

---

## 📊 Model Performance

The model was evaluated on a held-out test set from the Quora Question Pairs dataset.

### Metrics

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8660 |
| Precision | 0.7716 |
| Recall    | 0.9182 |
| F1 Score  | 0.8386 |

### Confusion Matrix

```
                    Predicted
                    Not Dup    Duplicate
Actual  Not Dup  [   518         103   ]
        Duplicate[    31         348   ]
```

| | Predicted: Not Duplicate | Predicted: Duplicate |
|---|---|---|
| **Actual: Not Duplicate** | 518 (True Negative) | 103 (False Positive) |
| **Actual: Duplicate** | 31 (False Negative) | 348 (True Positive) |

> The model achieves a high recall of **0.9182**, meaning it correctly identifies 91.8% of actual duplicate questions — making it well-suited for production use cases where missing duplicates is costly.

---

## 🧠 Model Architecture

- **Base Model:** `bert-base-uncased` (pretrained by Google)
- **Task:** Binary classification (Duplicate / Not Duplicate)
- **Fine-tuning Dataset:** Quora Question Pairs (QQP)
- **Input Format:** `[CLS] Q1 [SEP] Q2 [SEP]`
- **Output:** Sigmoid probability → threshold at 0.5

---

## 🔍 Inference

```python
from src.predict import predict_duplicate

q1 = "How can I learn Python quickly?"
q2 = "What is the fastest way to master Python?"

result = predict_duplicate(q1, q2)
print(result)  # Output: Duplicate ✅
```

---

## 📌 Key Takeaways

- BERT's **bidirectional attention** captures semantic meaning that keyword-overlap methods miss
- High **recall (0.918)** ensures most duplicate questions are flagged
- The model handles paraphrasing, synonyms, and reordered structure gracefully
- Can be deployed in real-time pipelines for platforms like Quora or Stack Overflow

---

## 🙌 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al., 2018
