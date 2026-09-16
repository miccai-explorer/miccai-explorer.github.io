#!/usr/bin/env python3
"""Show what each KMeans cluster actually contains, so it can be named.

Naming twenty clusters used to be an act of recall: embed.py wrote
`{"0": "Cluster 0", ...}` and logged nothing but cluster sizes, so the only
way to name them was to have been there when they were made. That is
REPRODUCIBILITY.md gap G1, and it is why the specter2 names could rot without
anyone being able to tell: with no way to re-derive them, there was nothing to
compare the file against.

This prints, per cluster, the most distinctive terms and the titles nearest
the centroid. Terms are ranked by class-based TF-IDF over each cluster's
pooled titles and abstracts, with the title weighted three times, because a
title states the topic and an abstract wanders into method and dataset.

Usage:
    python analysis/describe_clusters.py                # the active model
    python analysis/describe_clusters.py bge-large      # a specific model
    python analysis/describe_clusters.py --check        # names vs clustering
    python analysis/describe_clusters.py --stamp        # bind names to it

`--stamp` records which partition the names were written for. Run it only
after reading the report and confirming the names fit; it is an assertion,
not a formality, and every later build trusts it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cluster_labels as cl
import numpy as np

PROCESSED = Path("data/processed")
ALL_JSON = PROCESSED / "miccai_all.json"
TITLE_WEIGHT = 3


def _load(model: str):
    data = np.load(PROCESSED / f"embeddings_{model}.npz", allow_pickle=True)
    with open(ALL_JSON, encoding="utf-8") as f:
        papers = json.load(f)
    by_id = {p["paper_id"]: p for p in papers}
    return list(data["paper_ids"]), data["cluster_ids"].astype(int), by_id, data


def _terms(paper_ids, cluster_ids, by_id, topn):
    from sklearn.feature_extraction.text import TfidfVectorizer

    ids = sorted(set(int(c) for c in cluster_ids))
    docs = []
    for k in ids:
        pooled = []
        for pid, c in zip(paper_ids, cluster_ids):
            if int(c) != k:
                continue
            p = by_id[pid]
            pooled.append((p["title"] + " ") * TITLE_WEIGHT)
            pooled.append(p.get("abstract") or "")
        docs.append(" ".join(pooled))
    vec = TfidfVectorizer(
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        token_pattern=r"[A-Za-z][A-Za-z\-]{2,}",
    )
    matrix = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for i, k in enumerate(ids):
        row = matrix[i].toarray().ravel()
        out[k] = [vocab[j] for j in row.argsort()[::-1][:topn]]
    return out


def _nearest(embeddings, cluster_ids, k, n):
    """Indices of the n papers closest to cluster k's centroid, by cosine.

    Nearest-centroid rather than a random sample: a random draw of a
    150-paper cluster shows you its edges as often as its middle, which is
    what makes a cluster look like it has no topic when it does.
    """
    idx = np.where(cluster_ids == k)[0]
    unit = embeddings[idx] / np.linalg.norm(embeddings[idx], axis=1, keepdims=True)
    centroid = unit.mean(0)
    centroid /= np.linalg.norm(centroid)
    return idx[(unit @ centroid).argsort()[::-1][:n]]


def describe(model: str, n_terms: int, n_titles: int) -> None:
    paper_ids, cluster_ids, by_id, data = _load(model)
    embeddings = data["embeddings"].astype(np.float64)
    terms = _terms(paper_ids, cluster_ids, by_id, n_terms)

    path = cl.path_for(model)
    labels, clustering, _ = cl.read(path) if path.exists() else ({}, None, None)
    current = cl.fingerprint(paper_ids, cluster_ids)

    print(f"model      : {model}")
    print(f"clustering : {current[:16]}  ({len(set(cluster_ids))} clusters, "
          f"{len(paper_ids)} papers)")
    if clustering is None:
        print("labels     : present but unbound (no fingerprint)"
              if labels else "labels     : none")
    elif clustering == current:
        print("labels     : bound to this clustering")
    else:
        print(f"labels     : WRITTEN FOR A DIFFERENT CLUSTERING "
              f"({clustering[:16]}) - every name below is on the wrong cluster")

    for k in sorted(terms):
        size = int((cluster_ids == k).sum())
        print(f"\n=== cluster {k}  (n={size})")
        if labels:
            print(f"    name : {labels.get(str(k), '(unnamed)')}")
        print(f"    terms: {', '.join(terms[k])}")
        for j in _nearest(embeddings, cluster_ids, k, n_titles):
            print(f"      - {by_id[paper_ids[j]]['title']}")


def check(model: str) -> int:
    paper_ids, cluster_ids, _, _ = _load(model)
    expected = cl.fingerprint(paper_ids, cluster_ids)
    try:
        cl.verify(cl.path_for(model), expected)
    except (cl.ClusteringMismatch, FileNotFoundError) as exc:
        print(f"FAIL {model}: {exc}", file=sys.stderr)
        return 1
    print(f"OK   {model}: names are bound to the clustering on disk")
    return 0


def stamp(model: str) -> int:
    paper_ids, cluster_ids, _, _ = _load(model)
    path = cl.path_for(model)
    labels, _, _ = cl.read(path)
    ids = {str(k) for k in set(int(c) for c in cluster_ids)}
    missing = ids - set(labels)
    if missing:
        print(
            f"ERROR: {path} has no name for cluster(s) "
            f"{sorted(missing, key=int)}",
            file=sys.stderr,
        )
        return 1
    cl.write(path, labels, cl.fingerprint(paper_ids, cluster_ids), model)
    print(f"Stamped {path} with the clustering in embeddings_{model}.npz")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", nargs="?", help="model slug (default: active)")
    ap.add_argument("--check", action="store_true",
                    help="verify names are bound to the clustering; exit 1 if not")
    ap.add_argument("--stamp", action="store_true",
                    help="bind the current names to the clustering on disk")
    ap.add_argument("--terms", type=int, default=18)
    ap.add_argument("--titles", type=int, default=12)
    args = ap.parse_args()

    model = args.model
    if not model:
        active = PROCESSED / "active_model.txt"
        if not active.exists():
            print("ERROR: no model given and no active_model.txt", file=sys.stderr)
            return 1
        model = active.read_text().strip()

    if not (PROCESSED / f"embeddings_{model}.npz").exists():
        print(f"ERROR: no embeddings_{model}.npz", file=sys.stderr)
        return 1

    if args.check:
        return check(model)
    if args.stamp:
        return stamp(model)
    describe(model, args.terms, args.titles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
