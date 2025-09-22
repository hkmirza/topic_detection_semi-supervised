# Semi-Supervised Topic Detection (Term Similarity → k-means (Elbow) → PLDA with Label Constraints)

This repository contains a **reproducible implementation** of the semi-supervised topic detection pipeline described in your thesis (Chapter 5). The pipeline integrates TF–IDF term similarity, **k-means** with **elbow-based** model selection, and **PLDA** (Partially Labeled LDA) with **per-document label constraints**. It outputs ranked topic–keyword lists, document–topic assignments, and evaluation metrics that match the reporting style used in the thesis (accuracy, precision, recall, F1; plus NPMI topic coherence).

> **High-level flow:** Preprocess → TF–IDF → Elbow → k-means → seed lexicons → PLDA (with label constraints) → topics & assignments → evaluation & coherence.

---

## 1) Features

- **Unsupervised backbone:** TF–IDF + k-means with elbow selection (discrete curvature knee).
- **Semi-supervised refinement:** PLDA constrained by partial labels (`label` column, optional) and seeded by cluster top-terms.
- **Outputs aligned with thesis tables:**
  - Top-*N* keywords per topic (CSV)
  - Representative utterances per topic (CSV)
  - Document–topic distributions (CSV)
  - Topic–word distributions (CSV)
  - Evaluation metrics (JSON): accuracy, precision, recall, F1 (macro/micro), and confusion matrix (PNG)
  - **Topic coherence (NPMI)** per topic (CSV)
  - Elbow plot and label-coverage diagnostics (PNGs)
- **Reproducible**: fixed seeds, version-pinned `requirements.txt`, deterministic sklearn initialisation.

---

## 2) Installation

### 2.1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2.2. Install dependencies

```bash
pip install -r requirements.txt
```

### 2.3. Download NLTK resources (first run only)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

> If you are behind a proxy/firewall, set your proxy env vars or download these once and cache.

---

## 3) Data Format

Provide a single CSV file with the following columns:

| column         | required | description                                                                 |
|----------------|----------|-----------------------------------------------------------------------------|
| `dialogue_id`  | ✓        | Conversation/session identifier                                             |
| `utterance`    | ✓        | Single utterance text (one row per utterance)                               |
| `label`        | ✗        | Optional **partial** topic label (string). Leave empty for unlabeled rows.  |

**Notes**

- Multiple datasets (e.g., Switchboard, PersonaChat, MultiWOZ) can be processed separately by calling the script per dataset.
- Keep labels **consistent** (e.g., `travel`, `food`, `hotel`). The PLDA stage constrains topics only for documents with observed labels.
- If no labels are present, the pipeline still runs; PLDA will behave like LDA with seeding.

---

## 4) Usage

### 4.1. Basic run

```bash
python semi_supervised_topic_detection.py \
  --data path/to/dataset.csv \
  --outdir outputs/exp1_switchboard \
  --ngram 1 2 \
  --min_df 5 \
  --max_df 0.95 \
  --kmin 5 \
  --kmax 60 \
  --seed_terms 10 \
  --topics auto \
  --alpha 0.5 \
  --beta 0.1 \
  --gibbs_sweeps 1000
```

### 4.2. Important arguments

- `--data`: path to the CSV (see *Data Format*).
- `--outdir`: directory for all outputs and figures.
- `--ngram`: n-gram range for TF–IDF (default `1 2`).
- `--min_df`, `--max_df`: vocabulary pruning thresholds.
- `--kmin`, `--kmax`: candidate K range for elbow selection.
- `--seed_terms`: number of top centroid-aligned terms to seed each topic.
- `--topics`: number of PLDA topics. Use `auto` to set `T = K*` from elbow.
- `--alpha`, `--beta`: symmetric Dirichlet priors (as per thesis defaults).
- `--gibbs_sweeps`: collapsed Gibbs sampling sweeps for PLDA (e.g., 1000).

Run `python semi_supervised_topic_detection.py -h` for the full list of options.

---

## 5) Outputs

All artifacts are written into `--outdir`:

- `topics_top_keywords.csv` — per-topic ranked top-*N* keywords (as in thesis tables)
- `topics_representative_utterances.csv` — top utterances per topic by θ
- `theta_doc_topic.csv` — document–topic distribution (θ)
- `phi_topic_word.csv` — topic–word distribution (φ)
- `assignments.csv` — per-document hard topic label (argmax θ)
- `metrics.json` — accuracy / precision / recall / F1 (macro & micro) + counts
- `confusion_matrix.png` — standard confusion matrix
- `elbow_plot.png` — WCSS vs. K with detected knee
- `label_coverage.png` — diagnostic of partial-label coverage
- `coherence_npmi.csv` — NPMI per topic (top-*N* words), averaged and individual scores

You can directly import the CSVs to recreate the *per-topic keyword* tables that appear in the thesis.

---

## 6) Reproducibility

- Deterministic initialisation for k-means (`k-means++` with fixed `random_state` and `n_init`).
- Fixed RNG seeds across NumPy, Python, and scikit-learn.
- Version-pinned dependencies in `requirements.txt`.
- Exported configs: the script writes the resolved CLI args to `run_config.yaml` inside `--outdir`.

To **fully reproduce** the thesis tables/figures, run the script separately for each dataset (Switchboard, PersonaChat, MultiWOZ) using the same preprocessing thresholds and the default priors `α=0.5, β=0.1`, `gibbs_sweeps=1000`, and `seed_terms=10`.

---

## 7) Example: PersonaChat

```bash
python semi_supervised_topic_detection.py \
  --data data/personachat.csv \
  --outdir outputs/personachat_semisup \
  --ngram 1 2 \
  --min_df 5 --max_df 0.95 \
  --kmin 5 --kmax 60 \
  --seed_terms 10 \
  --topics auto \
  --alpha 0.5 --beta 0.1 \
  --gibbs_sweeps 1000
```

Inspect `outputs/personachat_semisup/topics_top_keywords.csv` to view the topic tables and `coherence_npmi.csv` for coherence.

---

## 8) Troubleshooting

- **No knee detected:** tighten/expand `--kmin/--kmax`, or increase TF–IDF granularity (use `--ngram 1 2`, lower `min_df` slightly).
- **Sparse labels (semi-supervision ineffective):** ensure `label` values are consistent and present for a meaningful subset of rows.
- **Low NPMI:** increase `seed_terms` modestly (e.g., 15) or raise `gibbs_sweeps`.
- **Memory issues:** reduce vocabulary via higher `min_df` and/or lower `max_df`.

---

## 9) Citation

If you use this code in academic work, please cite the thesis chapter that introduced the method and the core PLDA work:

- Ramage, D., Hall, D., Nallapati, R., & Manning, C. D. (2011). *Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora.*
- (Your Thesis) Chapter 5: Semi-supervised Topic Detection (provide full BibTeX in your repository).

---

## 10) License

Specify a license (e.g., MIT) in `LICENSE` if you plan to release the repository publicly.
