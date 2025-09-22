#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semi-supervised Topic Detection:
Term Similarity (TF–IDF) -> Elbow-selected k-means -> Seed-biased PLDA with per-document label constraints.

Implements(semi-supervised) pipeline using notation:
- U: utterances (documents)
- T: number of topics
- theta (θ): document–topic distribution
- phi (Φ): topic–word distribution
- Lambda_d (Λ_d): per-document allowed label set (constraints)
- Seeds: top-M terms per k-means cluster (initial bias for topics)

Outputs:
- topics_top_terms.csv              # top-N keywords per topic (table-like)
- doc_topics.csv                    # θ per utterance + argmax topic + mapped gold
- evaluation.json                   # accuracy, precision, recall, F1 (Hungarian alignment)
- npmi.csv                          # per-topic NPMI + macro-average
- elbow_plot.png                    # inertia curve with K*
- seeds.json                        # seed words per topic
"""

from __future__ import annotations
import argparse
import ast
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


# ---------------------------
# Utilities: data loading / parsing
# ---------------------------

def _parse_allowed_labels(val: str | list | float | None) -> List[str]:
    """Parse Λ_d from a cell that might be JSON, semicolon-separated, or NaN."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if not s:
        return []
    # Try JSON first
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    # Try Python literal
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    # Fallback: split on semicolon or comma
    if ";" in s:
        return [x.strip() for x in s.split(";") if x.strip()]
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s] if s else []


def load_dialogue_csv(paths: List[str]) -> pd.DataFrame:
    """Load one or more CSVs with columns:
       required: dialogue_id, utterance_id, text
       optional: label (gold), allowed_labels (Λ_d)
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    req = ["dialogue_id", "utterance_id", "text"]
    for col in req:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV.")

    # Normalise optional columns
    if "label" not in df.columns:
        df["label"] = np.nan
    if "allowed_labels" in df.columns:
        df["Lambda"] = df["allowed_labels"].apply(_parse_allowed_labels)
    else:
        df["Lambda"] = [[] for _ in range(len(df))]

    # Basic cleaning of text (keep robust; Vectorizer will also handle tokenization)
    df["text"] = (df["text"].astype(str)
                  .str.replace(r"https?://\S+", " ", regex=True)
                  .str.replace(r"[\r\n\t]+", " ", regex=True)
                  .str.replace(r"[^A-Za-z0-9'\- ]+", " ", regex=True)
                  .str.replace(r"\s{2,}", " ", regex=True)
                  .str.strip()
                  .str.lower())

    return df


# ---------------------------
# A. Term Similarity: TF–IDF + elbow-selected k-means
# ---------------------------

def build_tfidf(
    texts: Sequence[str],
    ngram_range=(1, 2),
    min_df:int=5,
    max_df:float=0.95
) -> Tuple[TfidfVectorizer, np.ndarray, List[str]]:
    vect = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,           # already lowercased
        token_pattern=r"(?u)\b\w[\w\-']+\b"
    )
    X = vect.fit_transform(texts)
    X = normalize(X, norm="l2", axis=1)
    vocab = vect.get_feature_names_out().tolist()
    return vect, X, vocab


def _elbow_knee(inertias: List[float], Ks: List[int]) -> int:
    """Select K* via maximum perpendicular distance to the line between (Kmin, Jmin) and (Kmax, Jmax)."""
    x = np.array(Ks, dtype=float)
    y = np.array(inertias, dtype=float)
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    v = p2 - p1
    v_norm = v / (np.linalg.norm(v) + 1e-12)
    dists = []
    for xi, yi in zip(x, y):
        pi = np.array([xi, yi])
        proj_len = np.dot(pi - p1, v_norm)
        proj = p1 + proj_len * v_norm
        d = np.linalg.norm(pi - proj)
        dists.append(d)
    return int(x[int(np.argmax(dists))])


def choose_k_by_elbow(X, Kmin:int=5, Kmax:int=60, n_init:int=10, random_state:int=42, plot_path:Optional[Path]=None) -> Tuple[int, List[int], List[float]]:
    Ks = list(range(Kmin, Kmax+1))
    inertias = []
    for K in Ks:
        km = KMeans(n_clusters=K, n_init=n_init, random_state=random_state, init="k-means++", max_iter=300)
        km.fit(X)
        inertias.append(km.inertia_)
    Kstar = _elbow_knee(inertias, Ks)
    if plot_path:
        plt.figure(figsize=(6,4))
        plt.plot(Ks, inertias, marker='o')
        plt.axvline(Kstar, color='r', linestyle='--', label=f"K*={Kstar}")
        plt.xlabel("K (clusters)")
        plt.ylabel("Inertia (within-cluster SSE)")
        plt.title("Elbow selection")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
    return Kstar, Ks, inertias


def cluster_and_seeds(X, vocab: List[str], Kstar:int, topM:int=10, random_state:int=42) -> Tuple[np.ndarray, Dict[int, List[str]]]:
    km = KMeans(n_clusters=Kstar, n_init=20, random_state=random_state, init="k-means++", max_iter=300)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_  # shape (K, |V|)
    seeds = {}
    for k in range(Kstar):
        idx = np.argsort(-centroids[k])[:topM]
        seeds[k] = [vocab[i] for i in idx]
    return labels, seeds


# ---------------------------
# Tokenization for PLDA (bag-of-words indices)
# ---------------------------

def build_vocab_from_vectorizer(vocab: List[str]) -> Tuple[Dict[str,int], List[str]]:
    word2id = {w:i for i,w in enumerate(vocab)}
    id2word = vocab[:]
    return word2id, id2word


def doc_to_ids(text: str, word2id: Dict[str,int]) -> List[int]:
    toks = re.findall(r"(?u)\b\w[\w\-']+\b", text.lower())
    return [word2id[t] for t in toks if t in word2id]


# ---------------------------
# PLDA-like semi-supervised LDA with label constraints (collapsed Gibbs)
# ---------------------------

@dataclass
class PLDAConfig:
    T:int
    alpha:float=0.5
    beta:float=0.1
    sweeps:int=1000
    burn_in:int=200
    seed_bias:float=0.05   # pseudo-count to add for seed words in their topic
    random_state:int=42


@dataclass
class PLDAState:
    # Count tables
    C_wt: np.ndarray      # |V| x T
    C_dt: np.ndarray      # |D| x T
    z: List[List[int]]    # topic assignment per token for each doc
    docs: List[List[int]] # tokenized doc word ids
    V:int
    T:int
    D:int


def init_plda_state(
    docs_ids: List[List[int]],
    T:int,
    V:int,
    Lambda: List[List[int]],               # per-doc candidate topic set indices (subset of [0..T-1])
    seed_lexicon: Dict[int, List[int]],    # topic -> word ids (seeds)
    cfg: PLDAConfig
) -> PLDAState:
    rng = np.random.RandomState(cfg.random_state)
    D = len(docs_ids)
    C_wt = np.zeros((V, T), dtype=np.int32)
    C_dt = np.zeros((D, T), dtype=np.int32)

    # Bias seed words with small pseudo-counts
    seed_bias = np.zeros((V, T), dtype=np.float64)
    for t, wid_list in seed_lexicon.items():
        for wid in wid_list:
            if 0 <= wid < V:
                seed_bias[wid, t] += cfg.seed_bias

    z = []
    for d, doc in enumerate(docs_ids):
        cand = Lambda[d] if Lambda[d] else list(range(T))
        if not cand:
            cand = list(range(T))
        z_doc = []
        for w in doc:
            t = rng.choice(cand)
            z_doc.append(t)
            C_wt[w, t] += 1
            C_dt[d, t] += 1
        z.append(z_doc)

    state = PLDAState(C_wt=C_wt, C_dt=C_dt, z=z, docs=docs_ids, V=V, T=T, D=D)
    # Store seed_bias as float; we’ll add during sampling probabilities
    state.seed_bias = seed_bias
    return state


def run_plda_sampling(
    state: PLDAState,
    Lambda: List[List[int]],
    cfg: PLDAConfig
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(cfg.random_state)
    V, T, D = state.V, state.T, state.D
    alpha, beta = cfg.alpha, cfg.beta

    # Precompute denominator trackers for efficiency
    n_wt_sum = state.C_wt.sum(axis=0)  # length T (words per topic)
    n_dt_sum = state.C_dt.sum(axis=1)  # length D (tokens per doc)

    for sweep in range(cfg.sweeps):
        for d in range(D):
            doc = state.docs[d]
            cand = Lambda[d] if Lambda[d] else list(range(T))
            cand_arr = np.array(cand, dtype=int)
            for i, w in enumerate(doc):
                t_old = state.z[d][i]
                # decrement
                state.C_wt[w, t_old] -= 1
                state.C_dt[d, t_old] -= 1
                n_wt_sum[t_old] -= 1
                n_dt_sum[d] -= 1

                # compute conditional for allowed topics only
                # p(t) ∝ (C_wt[w,t] + seed_bias[w,t] + β) / (n_wt_sum[t] + Vβ) * (C_dt[d,t] + α) / (n_dt_sum[d] + Tα)
                num_w = state.C_wt[w, cand_arr] + state.seed_bias[w, cand_arr] + beta
                den_w = n_wt_sum[cand_arr] + V * beta
                num_d = state.C_dt[d, cand_arr] + alpha
                den_d = n_dt_sum[d] + T * alpha
                p = (num_w / den_w) * (num_d / den_d)
                # sample new topic
                if p.sum() <= 0:
                    # numerical fallback: uniform over cand
                    t_new = rng.choice(cand)
                else:
                    p = np.asarray(p, dtype=float)
                    p = p / p.sum()
                    # multinomial sample
                    t_new = rng.choice(cand, p=p)

                # increment
                state.z[d][i] = t_new
                state.C_wt[w, t_new] += 1
                state.C_dt[d, t_new] += 1
                n_wt_sum[t_new] += 1
                n_dt_sum[d] += 1

    # Estimate Φ and θ (with priors)
    phi = (state.C_wt.astype(float) + beta)
    phi = phi / phi.sum(axis=0, keepdims=True)  # |V| x T

    theta = (state.C_dt.astype(float) + alpha)
    theta = theta / theta.sum(axis=1, keepdims=True)  # D x T

    return phi, theta


# ---------------------------
# map k-means clusters to topics & Λ
# ---------------------------

def map_clusters_to_topics(Kstar:int) -> Dict[int,int]:
    """Here we keep a one-to-one mapping cluster k -> topic t=k."""
    return {k:k for k in range(Kstar)}


def build_Lambda_from_labels(
    df: pd.DataFrame,
    label2topic: Dict[str,int] | None
) -> List[List[int]]:
    """Build Λ_d (per-doc allowed topic indices). If label2topic is None, we leave Λ_d empty (no constraint)."""
    Lambda = []
    if (df["Lambda"].apply(len).sum() == 0) and (label2topic is None):
        return [[] for _ in range(len(df))]
    for _, row in df.iterrows():
        cand = set()
        if isinstance(row.get("Lambda", []), list):
            for lab in row["Lambda"]:
                if label2topic and lab in label2topic:
                    cand.add(label2topic[lab])
        Lambda.append(sorted(list(cand)))
    return Lambda


# ---------------------------
# Seeds: convert term strings to ids
# ---------------------------

def seeds_to_ids(seeds: Dict[int, List[str]], word2id: Dict[str,int]) -> Dict[int, List[int]]:
    out = {}
    for t, words in seeds.items():
        out[t] = [word2id[w] for w in words if w in word2id]
    return out


# ---------------------------
# Evaluation: Hungarian matching + metrics + NPMI
# ---------------------------

def hungarian_align(pred_topics: np.ndarray, gold_labels: List[str]) -> Tuple[Dict[int,str], Dict[str,int]]:
    """Align discovered topics (0..T-1) to gold label strings via cost = -confusion."""
    # Build label set
    labels = [g for g in gold_labels if isinstance(g, str) and g.strip()]
    unique_labels = sorted(list({g for g in labels}))
    if not unique_labels:
        return {}, {}
    # Map labels to indices
    lab2i = {lab:i for i,lab in enumerate(unique_labels)}
    T = int(pred_topics.max()) + 1 if len(pred_topics) else 0
    if T == 0:
        return {}, {}
    C = np.zeros((T, len(unique_labels)), dtype=int)
    for t, g in zip(pred_topics, gold_labels):
        if isinstance(g, str) and g in lab2i:
            C[int(t), lab2i[g]] += 1
    # Hungarian on cost = -C
    cost = C.max() - C  # maximize C -> minimize cost
    row_ind, col_ind = linear_sum_assignment(cost)
    topic2label = {int(t): unique_labels[int(lab_idx)] for t, lab_idx in zip(row_ind, col_ind)}
    label2topic = {lab: t for t, lab in topic2label.items()}
    return topic2label, label2topic


def compute_classification_metrics(y_true: List[str], y_pred_lab: List[str]) -> Dict[str,float]:
    mask = [isinstance(y, str) and y.strip() for y in y_true]
    y_true_f = [yt for yt, m in zip(y_true, mask) if m]
    y_pred_f = [yp for yp, m in zip(y_pred_lab, mask) if m]
    if not y_true_f:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}
    acc = accuracy_score(y_true_f, y_pred_f)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_f, y_pred_f, average="macro", zero_division=0)
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def compute_npmi(topics: Dict[int,List[int]], docs_ids: List[List[int]], V:int, eps:float=1e-12) -> Tuple[pd.DataFrame, float]:
    """Compute NPMI for each topic using top words (utterance-level co-occurrence)."""
    # term frequencies across docs (presence/absence per doc)
    D = len(docs_ids)
    df_counts = np.zeros(V, dtype=int)
    for doc in docs_ids:
        uniq = set(doc)
        for w in uniq:
            df_counts[w] += 1

    # For efficiency, we only compute pairwise co-occur counts for top words in each topic
    topic_rows = []
    for t, word_ids in topics.items():
        if len(word_ids) < 2:
            topic_rows.append({"topic": t, "npmi": 0.0, "pairs": 0})
            continue
        # Build co-occ matrix only for this set
        wset = set(word_ids)
        # Co-occurrences across docs (presence)
        co_counts = defaultdict(int)
        for doc in docs_ids:
            present = sorted(wset.intersection(doc))
            if len(present) < 2:
                continue
            # unique ids in doc
            uniq = sorted(set(present))
            for i in range(len(uniq)):
                for j in range(i+1, len(uniq)):
                    co_counts[(uniq[i], uniq[j])] += 1

        # compute NPMI
        npmis = []
        for i in range(len(word_ids)):
            for j in range(i+1, len(word_ids)):
                wi, wj = word_ids[i], word_ids[j]
                # order pair
                a, b = (wi, wj) if wi < wj else (wj, wi)
                p_i = df_counts[wi] / (D + eps)
                p_j = df_counts[wj] / (D + eps)
                p_ij = co_counts.get((a, b), 0) / (D + eps)
                if p_ij <= 0 or p_i <= 0 or p_j <= 0:
                    continue
                pmi = math.log(p_ij / (p_i * p_j) + eps)
                npmi = pmi / (-math.log(p_ij + eps))
                npmis.append(npmi)
        if npmis:
            topic_rows.append({"topic": t, "npmi": float(np.mean(npmis)), "pairs": int(len(npmis))})
        else:
            topic_rows.append({"topic": t, "npmi": 0.0, "pairs": 0})
    df = pd.DataFrame(topic_rows).sort_values("topic").reset_index(drop=True)
    macro = float(df["npmi"].mean()) if len(df) else 0.0
    return df, macro


# ---------------------------
# Main pipeline
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Semi-supervised Topic Detection.")
    parser.add_argument("--csv", nargs="+", required=True, help="Path(s) to CSV with dialogue_id, utterance_id, text, [label], [allowed_labels].")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--min_df", type=int, default=5)
    parser.add_argument("--max_df", type=float, default=0.95)
    parser.add_argument("--Kmin", type=int, default=5)
    parser.add_argument("--Kmax", type=int, default=60)
    parser.add_argument("--topM", type=int, default=10, help="Seed size per cluster (top terms).")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sweeps", type=int, default=1000)
    parser.add_argument("--burn_in", type=int, default=200)
    parser.add_argument("--seed_bias", type=float, default=0.05)
    parser.add_argument("--topN_print", type=int, default=10, help="Top-N words to display per topic.")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    df = load_dialogue_csv(args.csv).reset_index(drop=True)
    print(f"Loaded {len(df)} utterances.")

    # A) TF–IDF
    print("Building TF–IDF representation ...")
    vect, X, vocab = build_tfidf(
        df["text"].tolist(),
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df
    )

    # B) Elbow selection + k-means
    print("Selecting K via elbow method ...")
    elbow_path = out_dir / "elbow_plot.png"
    Kstar, Ks, inertias = choose_k_by_elbow(X, args.Kmin, args.Kmax, n_init=10, random_state=args.random_state, plot_path=elbow_path)
    print(f"Selected K* = {Kstar}")

    print("Clustering with k-means and extracting seed terms ...")
    cluster_labels, seeds_str = cluster_and_seeds(X, vocab, Kstar, topM=args.topM, random_state=args.random_state)
    with open(out_dir / "seeds.json", "w") as f:
        json.dump(seeds_str, f, indent=2)

    # C) Map clusters -> topics (1:1) and build Λ_d (constraints)
    cluster2topic = map_clusters_to_topics(Kstar)
    T = Kstar

    # If gold label mapping to topics is provided (optional), build label2topic
    # Otherwise, we rely on Λ_d in the CSV (allowed_labels) mapped externally.
    label_uniqs = sorted({l for l in df["label"].dropna().astype(str) if l.strip()})
    label2topic = None
    if label_uniqs:
        # naive mapping by majority cluster within each label
        tmp = pd.DataFrame({"lab": df["label"].astype(str), "c": cluster_labels})
        lab2c = tmp.groupby("lab")["c"].agg(lambda x: x.value_counts().idxmax()).to_dict()
        # map cluster to topic, then label->topic index
        label2topic = {lab: cluster2topic[c] for lab, c in lab2c.items()}

    # Build Λ_d as topic indices
    Lambda = build_Lambda_from_labels(df, label2topic)

    # D) Prepare PLDA inputs
    print("Preparing PLDA documents and vocabulary ...")
    word2id, id2word = build_vocab_from_vectorizer(vocab)
    docs_ids = [doc_to_ids(t, word2id) for t in df["text"].tolist()]
    V = len(id2word)

    # Convert seeds to word ids
    seeds_ids = seeds_to_ids(seeds_str, word2id)

    # Configure and run PLDA
    cfg = PLDAConfig(T=T, alpha=args.alpha, beta=args.beta,
                     sweeps=args.sweeps, burn_in=args.burn_in,
                     seed_bias=args.seed_bias, random_state=args.random_state)

    print("Initialising PLDA state ...")
    state = init_plda_state(docs_ids, T=T, V=V, Lambda=Lambda, seed_lexicon=seeds_ids, cfg=cfg)

    print("Running collapsed Gibbs sampling with label constraints ...")
    phi, theta = run_plda_sampling(state, Lambda=Lambda, cfg=cfg)   # Φ and θ

    # E) Outputs: top-N words per topic, representative utterances
    print("Extracting top-N words per topic ...")
    topN = args.topN_print
    topic_rows = []
    topic_top_ids = {}
    for t in range(T):
        order = np.argsort(-phi[:, t])[:topN]
        words = [id2word[i] for i in order]
        topic_rows.append({"topic": t, **{f"w{i+1}": w for i, w in enumerate(words)}})
        topic_top_ids[t] = order.tolist()
    topics_df = pd.DataFrame(topic_rows).sort_values("topic")
    topics_df.to_csv(out_dir / "topics_top_terms.csv", index=False)

    # Representative utterances = top-3 θ_{d,t}
    rep_rows = []
    for t in range(T):
        top_doc_idx = np.argsort(-theta[:, t])[:3]
        for rank, d_idx in enumerate(top_doc_idx, 1):
            rep_rows.append({
                "topic": t,
                "rank": rank,
                "dialogue_id": df.loc[d_idx, "dialogue_id"],
                "utterance_id": df.loc[d_idx, "utterance_id"],
                "theta": float(theta[d_idx, t]),
                "text": df.loc[d_idx, "text"]
            })
    pd.DataFrame(rep_rows).to_csv(out_dir / "representative_utterances.csv", index=False)

    # Predicted topic per doc
    pred_topic = theta.argmax(axis=1)

    #  gold labels exist, align discovered topics -> labels via Hungarian and compute metrics
    eval_payload = {}
    if label_uniqs:
        topic2label, label2topic_map = hungarian_align(pred_topic, df["label"].astype(str).tolist())
        pred_label = [topic2label.get(int(t), None) for t in pred_topic]
        metrics = compute_classification_metrics(df["label"].astype(str).tolist(), pred_label)
        eval_payload["topic2label"] = topic2label
        eval_payload["metrics"] = metrics
        print("Evaluation (Hungarian-aligned):", json.dumps(metrics, indent=2))
    else:
        print("No gold labels found; skipping classification metrics.")
        eval_payload["topic2label"] = {}
        eval_payload["metrics"] = {}

    # NPMI coherence (semi-supervised setting)
    print("Computing NPMI coherence ...")
    npmi_df, npmi_macro = compute_npmi(topic_top_ids, docs_ids, V)
    npmi_df.to_csv(out_dir / "npmi.csv", index=False)
    eval_payload["npmi_macro"] = float(npmi_macro)

    # Save θ, Φ summaries
    doc_topics_out = pd.DataFrame({
        "dialogue_id": df["dialogue_id"],
        "utterance_id": df["utterance_id"],
        "pred_topic": pred_topic
    })
    if "label" in df.columns:
        doc_topics_out["gold_label"] = df["label"].astype(str)
    # attach θ columns (can be large; include if desired)
    for t in range(T):
        doc_topics_out[f"theta_t{t}"] = theta[:, t]
    doc_topics_out.to_csv(out_dir / "doc_topics.csv", index=False)

    # Save evaluation JSON
    with open(out_dir / "evaluation.json", "w") as f:
        json.dump(eval_payload, f, indent=2)

    print("\n=== Summary ===")
    print(f"K* (topics)        : {T}")
    print(f"Top terms per topic: saved to {out_dir/'topics_top_terms.csv'}")
    print(f"Representative utt.: saved to {out_dir/'representative_utterances.csv'}")
    print(f"Doc-topic (θ)      : saved to {out_dir/'doc_topics.csv'}")
    print(f"NPMI (per topic)   : macro={npmi_macro:.3f}; saved to {out_dir/'npmi.csv'}")
    if eval_payload.get("metrics"):
        m = eval_payload["metrics"]
        print(f"Accuracy={m['accuracy']:.3f}  Precision={m['precision']:.3f}  Recall={m['recall']:.3f}  F1={m['f1']:.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
