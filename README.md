Á
- `--alpha`, `--beta`: symmetric Dirichlet priors (as per thesis defaults).
- `--gibbs_sweeps`: collapsed Gibbs sampling sweeps for PLDA (e.g., 1000).

Run `python semi_supervised_topic_detection.py -h` for the full list of options.

---

## Outputs

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

## Reproducibility

- Deterministic initialisation for k-means (`k-means++` with fixed `random_state` and `n_init`).
- Fixed RNG seeds across NumPy, Python, and scikit-learn.
- Version-pinned dependencies in `requirements.txt`.
- Exported configs: the script writes the resolved CLI args to `run_config.yaml` inside `--outdir`.

To **fully reproduce** the thesis tables/figures, run the script separately for each dataset (Switchboard, PersonaChat, MultiWOZ) using the same preprocessing thresholds and the default priors `α=0.5, β=0.1`, `gibbs_sweeps=1000`, and `seed_terms=10`.

---

## Example: PersonaChat

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

## Troubleshooting

- **No knee detected:** tighten/expand `--kmin/--kmax`, or increase TF–IDF granularity (use `--ngram 1 2`, lower `min_df` slightly).
- **Sparse labels (semi-supervision ineffective):** ensure `label` values are consistent and present for a meaningful subset of rows.
- **Low NPMI:** increase `seed_terms` modestly (e.g., 15) or raise `gibbs_sweeps`.
- **Memory issues:** reduce vocabulary via higher `min_df` and/or lower `max_df`.

---


---

## License

Specify a license (e.g., Apache) in `LICENSE` if you plan to release the repository publicly.
