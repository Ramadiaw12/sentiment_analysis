# ⚡ Prompt Optimization for Sentiment Analysis
### *A controlled experimental framework for evaluating LLM prompting strategies on IMDb*

<br>

> **Core question:** Do advanced prompting techniques actually improve performance on tasks modern LLMs already master?

<br>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Prompt Strategies](#-prompt-strategies-compared)
- [System Architecture](#️-system-architecture)
- [Experimental Pipeline](#-experimental-pipeline)
- [Results & Insights](#-results)
- [Core Components](#-core-components)
- [Code Overview](#-code-overview)
- [Experimental Design](#-experimental-design)
- [Pitfalls Avoided](#️-pitfalls-avoided)
- [Future Work](#-future-improvements)
- [Installation](#-installation)
- [Contributing](#-contributing)

---

## 🎯 Problem Statement

Modern LLMs achieve impressive zero-shot performance across a wide range of NLP tasks.  
But does adding complexity — examples, reasoning chains — genuinely move the needle?

This project provides a **controlled, reproducible experimental framework** to answer that question using **IMDb sentiment classification** as the benchmark task.

---

## 🧠 Prompt Strategies Compared

| Strategy | Description | Complexity | Expected Gain |
|---|---|:---:|:---:|
| **Zero-shot** | Instruction only | ⭐ | Baseline |
| **Few-shot** | + labeled examples | ⭐⭐ | Medium |
| **Chain-of-Thought** | + explicit reasoning steps | ⭐⭐⭐ | High |

---

## ⚙️ System Architecture

```
IMDb Dataset (50k)
        │
        ▼
  Train / Test Split (80 / 20)
   ┌────┴────┐
   │         │
Training   Golden
Examples   Dataset
   │         │
   ▼         ▼
Few-shot  Evaluation
Sampling     Set
   │         │
   ▼         │
Prompt       │
Builder ─────┤
             │
             ▼
       LLM Inference
             │
             ▼
        Predictions
             │
             ▼
    F1 Score Calculation
```

---

## 🔬 Experimental Pipeline

```
User  ──► Prompt Builder ──► LLM ──► Evaluator
  ▲          (strategy)              │
  │                                  │  compare vs. ground truth
  └──────────── F1 Score ◄───────────┘
```

Each run follows this sequence:

1. Build a prompt using the selected strategy (zero-shot / few-shot / CoT)
2. Send the prompt + review text to the LLM
3. Parse the prediction from the model response
4. Compare prediction against the ground truth label
5. Aggregate and return the micro F1 score

---

## 📊 Results

| Strategy | F1 Score | Interpretation |
|---|:---:|---|
| 🟢 Zero-shot | **0.95** | Strong baseline — the task is already solved |
| 🟡 Few-shot | **0.95** | No measurable improvement |
| 🔵 Chain-of-Thought | **0.95** | Overkill for binary classification |

### 🔍 Key Takeaways

- LLMs already master binary sentiment classification — the ceiling is nearly reached at zero-shot
- Adding labeled examples does **not** improve an already-saturated task
- Chain-of-Thought reasoning adds latency and cost with **zero accuracy benefit** here
- **Prompt complexity ≠ better performance** — match strategy to task difficulty

---

## 🧩 Core Components

```
Dataset ──► Preprocessing ──► Prompt Builder ──► LLM ──► Parser ──► Evaluator
```

| Component | Role |
|---|---|
| `Dataset` | IMDb 50k reviews, balanced positive/negative |
| `Preprocessing` | Clean split to avoid data leakage |
| `Prompt Builder` | Assembles system + few-shot examples + user turn |
| `LLM` | GPT-4, temperature = 0 (deterministic) |
| `Parser` | Robust keyword extraction from model output |
| `Evaluator` | Micro F1 score against gold labels |

---

## 💻 Code Overview

### Prompt Construction

```python
def create_prompt(system_message, examples, user_template):
    """
    Build a multi-turn prompt from a system message,
    optional few-shot examples, and a user template.
    """
    prompt = [{'role': 'system', 'content': system_message}]

    for ex in examples:
        prompt += [
            {'role': 'user',      'content': user_template.format(movie_review=ex['text'])},
            {'role': 'assistant', 'content': ex['sentiment']},
        ]

    return prompt
```

### Evaluation Engine

```python
def evaluate_prompt(prompt, gold_examples, user_template, model):
    """
    Run the full evaluation loop and return micro F1.
    """
    predictions, truths = [], []

    for example in gold_examples:
        response = model.invoke(
            prompt + [{'role': 'user', 'content': user_template.format(
                movie_review=example['text']
            )}]
        )
        predictions.append(safe_parse(response))
        truths.append(example['sentiment'])

    return f1_score(truths, predictions, average='micro')
```

### Robust Output Parser

```python
def safe_parse(response):
    """
    Extract sentiment label from free-form model output.
    Falls back to 'unknown' rather than crashing.
    """
    text = response.content.lower()

    if 'negative' in text:
        return 'negative'
    elif 'positive' in text:
        return 'positive'

    return 'unknown'
```

---

## 🧪 Experimental Design

| Variable | Value | Rationale |
|---|---|---|
| Model | GPT-4 | Stable, well-documented reference point |
| Temperature | 0 | Deterministic — reproducible across runs |
| Dataset | IMDb | Large, balanced, widely benchmarked |
| Metric | Micro F1 | Robust to class imbalance |
| Split | 80 / 20 | Standard train/eval split |

---

## ⚠️ Pitfalls Avoided

### ❌ Data Leakage

Few-shot examples must be sampled **after** the train/test split — never before.

```python
# ❌ BAD — examples may bleed into the evaluation set
examples = sample(full_dataset)
train, test = split(full_dataset)

# ✅ GOOD — evaluation set is never touched during example selection
train, test = split(full_dataset)
examples = sample(train)
```

### ✅ Robust Parsing

The parser handles varied model phrasing without throwing errors:

```python
def safe_parse(response):
    text = response.content.lower()
    if 'negative' in text:
        return 'negative'
    elif 'positive' in text:
        return 'positive'
    return 'unknown'   # graceful fallback, tracked separately
```

---

## 🚀 Future Improvements

| Direction | Description |
|---|---|
| **Prompt Ensembling** | Aggregate predictions across multiple prompt variants |
| **Dynamic Few-shot** | Select examples by semantic similarity to the input |
| **Multi-Model Comparison** | Replicate across GPT-3.5, Claude, Mistral, Llama |
| **Self-Consistency** | Sample multiple reasoning paths and take majority vote |
| **Error Analysis** | Characterise the ~5% of misclassified reviews |
| **Harder Tasks** | Move to aspect-level or multi-class sentiment where CoT may shine |

---

## 📦 Installation

```bash
git clone https://github.com/Ramadiaw12/sentiment_analysis
cd prompt-optimization-imdb
```

### Configuration

```bash
cp config/.env.example .env
```

```env
OPENAI_API_KEY=your_key_here
```

---

## 🤝 Contributing

Contributions are welcome — especially new prompt strategies or additional benchmark datasets.

```bash
# 1. Fork the repository
# 2. Create your feature branch
git checkout -b feature/your-improvement

# 3. Commit your changes
git commit -m "feat: add dynamic few-shot selection"

# 4. Push and open a Pull Request
git push origin feature/your-improvement
```

Please keep pull requests focused on a single concern and include evaluation results where applicable.

---

## 📄 License

Distributed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

<div align="center">

*If this project was useful to you, consider leaving a ⭐ on GitHub.*

</div>