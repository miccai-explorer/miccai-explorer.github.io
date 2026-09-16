#!/usr/bin/env python3
"""
Switch the active embedding model and/or projection without re-running embed.py.

Usage:
    python analysis/use_model.py specter2
    python analysis/use_model.py specter2 umap-tight
    python analysis/use_model.py specter2 pacmap
    python analysis/use_model.py bge-large

If projection is omitted, keeps the current active_proj.txt (defaults to umap-tight).
"""

import json
import shutil
import sys
from pathlib import Path

import cluster_labels as cl
import numpy as np

ALL_JSON = Path("data/processed/miccai_all.json")
ACTIVE_FILE = Path("data/processed/active_model.txt")
ACTIVE_PROJ_FILE = Path("data/processed/active_proj.txt")
DEFAULT_PROJ = "umap-tight"


def _usage(out=sys.stdout) -> None:
    """Usage plus whatever models and projections are actually on disk.

    Listing what exists beats listing what the registry claims: a model is
    only switchable once embed.py has written its .npz, and the whole point
    of this script is to switch between the ones already computed.
    """
    print("Usage: python analysis/use_model.py <model> [<proj>]", file=out)
    avail = sorted(Path("data/processed").glob("embeddings_*.npz"))
    if avail:
        print(
            "  Models: "
            + ", ".join(p.stem.replace("embeddings_", "") for p in avail),
            file=out,
        )
    avail_projs = sorted(Path("data/processed").glob("proj_*.npy"))
    if avail_projs:
        print(
            "  Projections: " + ", ".join(p.stem for p in avail_projs),
            file=out,
        )


def main() -> int:
    # No argparse here: the script takes one or two bare positional values and
    # adding a parser for that is more code than it saves. -h has to be handled
    # by hand as a result, or it is read as a model name and reported as a
    # missing embeddings_--help.npz.
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        print()
        _usage()
        return 0

    if len(sys.argv) < 2:
        _usage(sys.stderr)
        return 1

    model = sys.argv[1]
    proj = (
        sys.argv[2]
        if len(sys.argv) > 2
        else (
            ACTIVE_PROJ_FILE.read_text().strip()
            if ACTIVE_PROJ_FILE.exists()
            else DEFAULT_PROJ
        )
    )

    emb_path = Path(f"data/processed/embeddings_{model}.npz")
    proj_path = Path(f"data/processed/proj_{model}_{proj}.npy")
    cl_path = Path(f"data/processed/cluster_labels_{model}.json")

    if not emb_path.exists():
        print(f"ERROR: {emb_path} not found.", file=sys.stderr)
        print(
            f"       Run: python analysis/embed.py --model {model}",
            file=sys.stderr,
        )
        return 1
    if not cl_path.exists():
        print(f"ERROR: {cl_path} not found.", file=sys.stderr)
        return 1

    data = np.load(emb_path, allow_pickle=True)
    paper_ids = list(data["paper_ids"])
    cluster_ids = data["cluster_ids"]

    if proj_path.exists():
        coords = np.load(proj_path)
    elif "umap_coords" in data:
        print(
            f"WARNING: {proj_path} not found; falling back to umap_coords in {emb_path}"
        )
        coords = data["umap_coords"]
    else:
        print(f"ERROR: {proj_path} not found.", file=sys.stderr)
        print(
            f"       Run: python analysis/embed.py --model {model} --proj {proj} --no-recompute",
            file=sys.stderr,
        )
        return 1

    # Names are only meaningful for the partition they were read off. KMeans
    # renumbers freely between runs, so activating a model whose labels
    # predate its current clustering silently relabels the whole map; refuse
    # instead. See analysis/cluster_labels.py.
    try:
        labels = cl.verify(cl_path, cl.fingerprint(paper_ids, cluster_ids))
    except cl.ClusteringMismatch as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    with open(ALL_JSON, encoding="utf-8") as f:
        papers = json.load(f)

    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    n_updated = 0
    for p in papers:
        i = id_to_idx.get(p["paper_id"])
        if i is not None:
            p["umap_x"] = float(coords[i, 0])
            p["umap_y"] = float(coords[i, 1])
            p["cluster_id"] = int(cluster_ids[i])
            p["cluster_label"] = labels.get(
                str(cluster_ids[i]), f"Cluster {cluster_ids[i]}"
            )
            n_updated += 1

    with open(ALL_JSON, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    shutil.copy(cl_path, "data/processed/cluster_labels.json")
    ACTIVE_FILE.write_text(model)
    ACTIVE_PROJ_FILE.write_text(proj)

    print(f"Active model : {model}")
    print(f"Active proj  : {proj}")
    print(f"Papers updated: {n_updated}/{len(papers)}")
    print(f"Clusters: {len(labels)}")
    print(
        "\nNext: python analysis/build_charts.py && python analysis/build_site.py"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
