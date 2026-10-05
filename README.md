# NLP Project – Automated Short-Answer Grader

## Overview

This project builds a system that automatically grades short student answers against rubric criteria. Given a question, an optional model solution, a list of rubric criteria, and a list of student answers, the system decides for each (answer, rubric) pair whether the criterion is met.

| Decision | Meaning |
|----------|---------|
| `yes`  | The answer satisfies the rubric criterion. |
| `half` | The answer partially satisfies the criterion. Only applicable if rubric is marked as ternary. |
| `no`   | The answer does not satisfy the criterion. |

---

## Final Results

After a systematic investigation and model comparison study, the best
model for submission is **v11-base** (`amrit97/xlm-roberta-base-grader-v11`):

| Model | RMSE | Accuracy | Ternary Acc | Notes |
|---|---|---|---|---|
| v9-large | 0.3136 | 70.7% | 55.4% | Ternary rubrics unsolved |
| v10-large (CV) | **0.2549** | **80.3%** | - best | Best CV result, ternary weak in deployment |
| v11-base | 0.2243 | 94.1% | 93.8% | 3x faster than large |
| **v11-large** | **0.2248** | **94.0%** | **92.8%** | **Submission model** |

The full comparison of models is in `results/comparison_original.csv`.
Experiment history is in [EXPERIMENTS.md](EXPERIMENTS.md).
For all numerical results, see [results/experiment_results_summary.xlsx](results/experiment_results_summary.xlsx).

---

## Active Team Members & Focus areas

| 2s ID  | Member | Role / Focus |
|--------|--------|-------------|
| aamrit2s | **Amrit** | Core fine-tuning pipeline, multilingual investigation suite (v2–v9), SBERT baseline |
| jthyri2s | **Jannen** | TF-IDF baseline, preprocessing overhaul, k-fold setup, ensemble threshold optimisation, documentation |
| kali2s   | **Kamran** | GBERT experiments, German language model evaluation, model comparison (RoBERTa vs DeBERTa) |
| nravi2s  | **Nikhil** | Cross-encoder experiments, dual-head regression architecture, documentation |

---

## Approach & Experiments

The project progressed through a systematic series of investigations, each building on the previous one. All fine-tuning used a 5-fold `GroupKFold` split (grouped by `answer_id` to prevent leakage) with oversampling of minority score classes and a weighted MSE loss.

### Baselines

| Baseline | RMSE | Accuracy | Script |
|----------|------|----------|--------|
| Random | - | 33% | `app/grader.py` (original) |
| TF-IDF cosine similarity + grid-search thresholds | 0.5055 | 43.2% | `src/comparison/baseline_tfidf.py` |



### Fine-tuning Investigation Series

All experiments below use 5-fold cross-validation on combined multilingual datasets (German + English).

| Version | Key Change | Best RMSE | Best Acc | Script |
|---------|-----------|-----------|----------|--------|
| **v2** | First XLM-RoBERTa fine-tuning, majority-class collapse identified | 0.2774 (gbert) | 74.4% | `src/comparison/investigation_v2.py` |
| **v3** | GroupKFold + WeightedMSE + oversampling, majority-class collapse resolved | 0.2675 | 77.6% | `src/comparison/investigation_v3.py` |
| **v4** | HP ablation on GBERT; solution field excluded | 0.2541 | 80.1% | `src/comparison/investigation_v4.py` |
| **v5** | Multilingual: 3 datasets combined; XLM-RoBERTa-Base | 0.2673 | 76.2% | `src/comparison/investigation_v5.py` |
| **v6** | All field orderings tested; rubric-first layout wins | 0.2664 | 77.7% | `src/comparison/investigation_v6.py` |
| **v7** | Scale to XLM-RoBERTa-Large (~560M params) | 0.2586 | 76.6% | `src/comparison/investigation_v7.py` |
| **v8** | CORAL ordinal loss + LLRD + FGM adversarial training | 0.2614 | 79.9% | `src/comparison/investigation_v8.py` |
| **v9** | 138-config grid search; question_id generalization study on base and large models | 0.2561 | 79.5% | `src/comparison/investigation_v9.py` |
| **v10** | All techniques combined on xlm-roberta-large | 0.2549 | 80.3% | `src/comparison/investigation_v10.py` |
| **v11** | **is_ternary-aware training; back-translation augmentation; ternary rubric accuracy raised from 55% -> 93%** | **0.2243** | **94.1%** | `src/train_and_inference/train_final_v11.py` |

> **Bottom line:** **v11** is the best performing model to date, achieving 94.1% accuracy and RMSE 0.2243 which is a significant leap over v10 (80.3%). The key breakthrough was properly training with type-aware ternary rubric labels. v10 remains the best result from 5-fold cross-validation (RMSE 0.2549, honest held-out metric). See `results/comparison_original.csv` for full model comparison.

### Trained Models on HuggingFace

1. **https://huggingface.co/amrit97/xlm-roberta-base-grader-v11** - submission model
2. https://huggingface.co/amrit97/xlm-roberta-large-grader-v11
2. https://huggingface.co/amrit97/xlm-roberta-large-grader-v9
2. https://huggingface.co/amrit97/xlm-roberta-base-grader-v9
3. https://huggingface.co/jan024/asag-xlm-roberta-large-v10


### Key Findings

- **GroupKFold on answer_id is critical** - random KFold causes data leakage. Fixing this alone lifted xlm-roberta from 60% to 77.6% (v3).
- **Rubric-first ordering** - placing the rubric at the start of the input sequence primes the model's attention, reducing RMSE and variance (v6).
- **Solution field hurts** - including the reference solution adds noise without benefit; excluded from v5 onwards (v4).
- **Model scale** - xlm-roberta-large reduces RMSE by ~3% vs base but requires VRAM-aware batch settings (v7).
- **CORAL + LLRD + FGM** - ordinal loss maps better to discrete grades; layer-wise LR and adversarial training improve generalization (v8).
- **Generalization gap** - the model is strong on new answers to known questions (RMSE=0.2549) but weaker on completely new question topics (RMSE=0.3362). This is a fundamental cross-encoder property (v9).
- **Type-aware thresholds** - binary rubrics (is_ternary=False) never have label=1.0 in the training data; using a separate threshold prevents impossible predictions.

---

## Team Contributions & Related Scripts

| Focus Area | Work | Scripts |
|---|---|---|
| Training pipeline | GroupKFold, WeightedMSE, oversampling, token budgets, ensemble threshold optimization, TensorBoard overfitting analysis, is_ternary prefix, type-aware thresholds, v9 and v10 grid search, then we came up with the data augmented training with v11 | `src/train_and_inference/train_final_v11.py`, `src/train_and_inference/train_final_v10.py`, `src/train_and_inference/train_final_v9_large.py`, `src/train_and_inference/test_inference.py` |
| Field ablation study | `investigation_v2.py`, `investigation_v3.py`, `investigation_v6.py`, `investigation_v7.py` | `investigation_v8.py`, `investigation_v9.py`, `investigation_v10.py`, `investigation_v11.py` |
| HP optimization and scaling | Systematic HP tuning, field ablation, multilingual expansion, field ordering study, xlm-roberta-large scaling, v9 grid search | `investigation_v4.py` through `investigation_v9.py` |
| Advanced training | CORAL ordinal loss, LLRD, FGM adversarial training, dual-head architecture | `investigation_v8.py`, `notebooks/investigation_v8_dual_head.ipynb` |
| German model evaluation | gbert-base evaluation across datasets, RoBERTa vs DeBERTa comparison, language detection | `gbert_testing_with_lang/` |
| TF-IDF baseline | Non-semantic baseline with threshold tuning | `baseline_tfidf.py` |

---

## Contributions by Member

### Amrit
- Designed and ran the full investigation pipeline from v2 through v9 (`origin/multi_language_model_comparision`, `origin/model_ablation_experiments`)
- Integrated three multilingual datasets (WuS-WS25, DAV-SS2425, NLP-SS2425) and added WuS-SS24
- Implemented the SBERT cosine-similarity baseline ([`comparision/baseline_sbert.py`](comparision/baseline_sbert.py))
- Introduced GroupKFold, oversampling, and weighted MSE (v3) - the fixes that eliminated majority-class collapse
- Scaled training to XLM-RoBERTa-Large with gradient checkpointing and bf16 (v7)
- Added LLRD and FGM adversarial training (v8) and ran the 180-configuration grid search (v9)

### Jannen
- Implemented the TF-IDF cosine-similarity baseline with grid-search threshold tuning (`origin/baseline_tfidf`)
- Rewrote the preprocessing pipeline (HTML unescaping, per-field token budgets, text normalisation)
- Added k-fold support to the training script
- Designed and implemented per-question ensemble threshold optimisation (`origin/feature/ensemble-threshold-optimization`)
- Added LLRD and FGM on top of v6 hyper parameters and implemented v10 with large model scaling. 
- Documentation and final submission branch.
- Model comparison study with augmented data.

### Kamran
- Conducted all GBERT experiments - first model to surpass 80 % accuracy on the original dataset ([`gbert_kamran/`](gbert_kamran/))
- Built the initial RoBERTa vs DeBERTa comparison that motivated the investigation series ([`comparision/`](comparision/))
- Maintained the results folder and evaluated models on both old and new multilingual datasets
- Helped with dcumentation and testing scripts + dockerfile
- Original data model comparison study.


### Nikhil
- Explored cross-encoder architectures with answer summarisation (`origin/cross_encoder_experiments`)
- Implemented the **Regression + Classification Dual-Head** model (XLM-RoBERTa-Large backbone, separate regression and classification heads with a combined weighted loss): [`comparision/notebooks/investigation_v8_dual_head.ipynb`](comparision/notebooks/investigation_v8_dual_head.ipynb)
- Helped with Documetation and README.md
- Testing out models and verifying reproducibility of results.
- Model comparison study with data augmentation.

---

## Repository Structure

```
.
├── app/                              # FastAPI grading service (submission)
│   ├── grader.py                     # grade() implementation - model inference
│   ├── models.py                     # Pydantic request/response models (do not modify)
│   ├── main.py                       # FastAPI entry point (do not modify)
│   ├── Dockerfile                    # CPU-only PyTorch for portability
│   └── requirements.txt             # Service dependencies
├── src/
│   ├── comparison/                   # Investigation scripts and notebooks
│   │   ├── investigation_v2.py       # First fine-tuning pipeline
│   │   ├── investigation_v3.py       # GroupKFold + WeightedMSE fix
│   │   ├── investigation_v4.py       # HP tuning and field ablation
│   │   ├── investigation_v5.py       # Multilingual datasets
│   │   ├── investigation_v6.py       # Field ordering study
│   │   ├── investigation_v7.py       # XLM-RoBERTa-Large scaling
│   │   ├── investigation_v8.py       # CORAL + LLRD + FGM
│   │   ├── investigation_v9.py       # 138-config grid search
│   │   ├── investigation_v10.py      # Final ablation (best result)
│   │   ├── baseline_sbert.py         # SBERT cosine baseline
│   │   └── notebooks/                # Jupyter notebooks
│   └── train_and_inference/
│       ├── train_final_v10.py        # Reproduce v10 large model
        ├── train_final_v9_large.py   # Reproduce v9 large production model
│       ├── test_inference.py         # Verify any checkpoint before deploying
│       └── upload_model_to_hub.py    # Upload trained model to HuggingFace Hub
├── results/
│   ├── experiment_results_summary.xlsx  # All 26 experiments with color coding
│   └── investigation_v9_results.csv     # Full v9 grid (138 configurations)
├── data/                             # Datasets (not committed, see below)
├── tests/                            # Unit and integration tests
├── EXPERIMENTS.md                    # Full experiment history with analysis
├── .env.example                      # Environment variable documentation
├── docker-compose.yml                # Service definition (do not modify)
└── eval.env                          # Filled in by evaluator (do not modify)
```

---

## Getting Started

### 1. Your Task

Open `app/grader.py` and implement the `grade` function. The signature is fixed - **do not change it**.

```python
def grade(request: GradeRequest) -> GradeResponse:
    ...
```

You may add helper functions and modules inside `app/`, add dependencies to `app/requirements.txt`, and add environment variables to `.env` (document them in `.env.example`).

You must **not** modify `app/models.py`, `app/main.py`, or `docker-compose.yml`.

### 2. Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/)
- Python 3.11+ (for local testing without Docker)
- A HuggingFace account with a read token (to download the model)

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in HF_TOKEN and MODEL_REPO
```

### 4. Start the service

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`. The model downloads from HuggingFace Hub on first startup.

### 5. Verify the service works

```bash
curl -s -X POST http://localhost:8000/grade \
  -H "Content-Type: application/json" \
  -d @data/example_input.json | python3 -m json.tool
```

### 6. Run tests

```bash
# Unit tests (no Docker needed)
pytest tests/test_grader.py -v

# Integration tests (service must be running)
docker compose up --build -d
pytest tests/test_api.py -v
```

### 7. Test with sample payloads

A set of realistic test payloads covering English and German questions is
provided in `data/`. To run all of them against the running service:

```bash
python3 tests/test_run_grader.py
```

To test a single payload:

```bash
python3 tests/test_run_grader.py --payload data/test_payload_nlp.json
```

Available payloads:
- `data/example_input.json` - Earth seasons (given)
- `data/test_payload_nlp.json` - Language model evaluation (English, NLP course style)
- `data/test_payload_word2vec.json` - Skip-gram and word embeddings (English)
- `data/test_payload_german.json` - Linear regression (German, WuS course style)


### 8. Run the official test suite

Integration tests (service must be running, no torch needed locally):

```bash
pytest tests/test_api.py -v --ignore=tests/test_grader.py
```

For local development dependencies:

```bash
pip install -r requirements-dev.txt
```

---

## API Reference

### `GET /health`

Returns `{"status": "ok"}` when the service is ready.

### `POST /grade`

**Request body:**

```json
{
  "question_id": "q42",
  "question": "What causes seasons on Earth?",
  "solution": "Seasons are caused by the tilt of Earth's axis.",
  "rubrics": [
    {"id": "r1", "text": "The answer mentions the tilt of Earth's axis."},
    {"id": "r2", "text": "The answer rejects distance from the Sun.", "is_ternary": true}
  ],
  "student_answers": [
    {"id": "a1", "text": "The Earth is tilted on its axis."},
    {"id": "a2", "text": "The Earth gets closer to the Sun in summer."}
  ]
}
```

`solution` is optional. `is_ternary` defaults to `false`. Binary rubrics only return `yes` or `no` - never `half`.

**Response body:**

```json
{
  "question_id": "q42",
  "results": [
    {
      "answer_id": "a1",
      "rubric_decisions": [
        {"rubric_id": "r1", "decision": "yes"},
        {"rubric_id": "r2", "decision": "no"}
      ]
    }
  ]
}
```

---

## Evaluation

The evaluator will:

1. Run `docker compose up --build`
2. Poll `GET /health` until the service responds (max 60 s)
3. Send grading requests and compare decisions against a gold standard
4. Run `docker compose down` to stop the service

The service must respond to each `/grade` request within **30 seconds**.

---

## Data

Training datasets are not committed to this repository. They should be placed in `data/` with the following structure:

```
data/
  WuS-WS25/
    questions.csv  answers.csv  rubrics.csv  answer_rubrics.csv
  DAV-SS2425/
    questions.csv  answers.csv  rubrics.csv  answer_rubrics.csv
  NLP-SS2425/
    questions.csv  answers.csv  rubrics.csv  answer_rubrics.csv
  WuS-SS24/
    questions.csv  answers.csv  rubrics.csv  answer_rubrics.csv
```