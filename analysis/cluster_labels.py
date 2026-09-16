#!/usr/bin/env python3
"""Cluster names, bound to the clustering they were written for.

KMeans cluster ids are positions in an arbitrary numbering, not meanings. Two
runs over the same papers can produce the same twenty groups and hand them out
in a different order, and nothing in the numbers says so. The names in
`cluster_labels_{model}.json` are written by a human reading the clusters, so
the moment the partition is recomputed the file becomes a list of twenty
correct names attached to the wrong groups.

That is not hypothetical. It happened to `specter2`, the active model, and it
survived into production: every cluster on the semantic map was mislabelled,
"Deformable Image Registration" sat on the pathology cluster, and no build
step, test or eyeball caught it, because a wrong name is still a name.
Found 2026-09-08.

The fix is to make the file say which partition it describes. `fingerprint()`
hashes the (paper_id -> cluster_id) assignment; `load()` refuses to hand back
names whose fingerprint does not match the clustering in front of it. A
relabelled run now fails loudly at build time instead of publishing confident
nonsense.

File format:

    {"model": "specter2",
     "clustering": "<64 hex chars>",
     "labels": {"0": "Deformable Image Registration", ...}}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROCESSED = Path("data/processed")


class ClusteringMismatch(RuntimeError):
    """The names on disk were written for a different partition."""


def fingerprint(paper_ids, cluster_ids) -> str:
    """Hash a paper -> cluster assignment.

    Sorted by paper_id so the digest describes the partition and its
    numbering, not the order the arrays happen to arrive in: `use_model.py`
    joins by paper_id, so row order is not part of the meaning.
    """
    pairs = sorted(
        (str(pid), int(cid)) for pid, cid in zip(paper_ids, cluster_ids)
    )
    h = hashlib.sha256()
    for pid, cid in pairs:
        h.update(f"{pid}\t{cid}\n".encode())
    return h.hexdigest()


def fingerprint_of_papers(papers) -> str:
    """Fingerprint the clustering carried by miccai_all.json records."""
    rows = [
        (p["paper_id"], p["cluster_id"])
        for p in papers
        if p.get("cluster_id") is not None
    ]
    return fingerprint([r[0] for r in rows], [r[1] for r in rows])


def path_for(model: str) -> Path:
    return PROCESSED / f"cluster_labels_{model}.json"


def read(path: Path) -> tuple[dict[str, str], str | None, str | None]:
    """Return (labels, clustering, model) without checking anything.

    Accepts the flat `{"0": "name"}` shape that predates the fingerprint and
    reports its clustering as None, so a caller can tell "written before this
    existed" apart from "written for another partition". Only `verify()`
    decides whether that is acceptable.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if "labels" in raw and isinstance(raw["labels"], dict):
        return dict(raw["labels"]), raw.get("clustering"), raw.get("model")
    return {str(k): str(v) for k, v in raw.items()}, None, None


def write(path: Path, labels: dict[str, str], clustering: str, model: str) -> None:
    payload = {
        "model": model,
        "clustering": clustering,
        "labels": {str(k): labels[str(k)] for k in sorted(labels, key=int)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def verify(path: Path, expected: str) -> dict[str, str]:
    """Load names and prove they belong to the `expected` clustering.

    Raises rather than warns. A warning here is worth nothing: the build
    still succeeds, the site still renders, and the only thing that tells you
    the map is wrong is reading the titles under twenty legend entries.
    """
    labels, clustering, _ = read(path)
    if clustering is None:
        raise ClusteringMismatch(
            f"{path} carries no 'clustering' fingerprint, so there is no way "
            f"to tell which partition these names describe.\n"
            f"       Check the names against the clusters "
            f"(python analysis/describe_clusters.py), then stamp the file:\n"
            f"       python analysis/describe_clusters.py --stamp <model>"
        )
    if clustering != expected:
        raise ClusteringMismatch(
            f"{path} names a different clustering than the one on disk.\n"
            f"       labels written for : {clustering[:16]}\n"
            f"       clustering present : {expected[:16]}\n"
            f"       The embeddings were recomputed after these names were "
            f"written, so every name is now on the wrong cluster.\n"
            f"       Re-read the clusters and rename them:\n"
            f"       python analysis/describe_clusters.py <model>"
        )
    return labels
