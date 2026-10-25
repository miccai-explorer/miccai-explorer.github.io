#!/usr/bin/env python3
"""
Build sentence embeddings, dimensionality-reduction projections, and KMeans clusters.

Embedding models (--model):
  specter2    allenai/specter2_base + proximity adapter  [recommended: trained on citation graphs]
  bge-large   BAAI/bge-large-en-v1.5
  gte-large   Alibaba-NLP/gte-large-en-v1.5
  e5-large    intfloat/e5-large-v2
  all-mpnet   sentence-transformers/all-mpnet-base-v2

Projection methods (--proj, default: all):
  umap          UMAP original params  (n_neighbors=15, min_dist=0.12)
  umap-tight    UMAP tuned            (n_neighbors=30, min_dist=0.05, n_epochs=500) [recommended]
  pacmap        PaCMAP, explicit local+global balance (requires: uv add pacmap)
  densmap       densMAP, a UMAP variant that also preserves local density

Reads  : data/processed/miccai_all.json
Writes : data/processed/embeddings_{model}.npz       (embeddings, cluster_ids, paper_ids)
         data/processed/proj_{model}_{proj}.npy       (2D coords per projection, N×2 float32)
         data/processed/cluster_labels_{model}.json   (placeholder; edit before building charts)
         data/processed/active_model.txt
         data/processed/active_proj.txt
         data/processed/miccai_all.json               (umap_x, umap_y, cluster_id, cluster_label)
         data/processed/cluster_labels.json           (copy of cluster_labels_{model}.json)

Usage:
    python analysis/embed.py --model specter2                         # embed + all projs + activate umap-tight
    python analysis/embed.py --model specter2 --proj pacmap           # run only one projection
    python analysis/embed.py --model specter2 --no-recompute          # skip embeddings, redo all projs
    python analysis/embed.py --model specter2 --no-recompute --proj umap-tight   # skip embed, redo one
    python analysis/embed.py --model specter2 --no-recompute --no-reproject      # skip both if files exist

To switch without re-running:
    python analysis/use_model.py specter2
    python analysis/use_model.py specter2 pacmap
"""

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

import cluster_labels as cl
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

ALL_JSON = Path("data/processed/miccai_all.json")
ACTIVE_FILE = Path("data/processed/active_model.txt")
ACTIVE_PROJ_FILE = Path("data/processed/active_proj.txt")

MODELS: dict[str, dict] = {
    "gte-large": {
        "hf_id": "Alibaba-NLP/gte-large-en-v1.5",
        "loader": "sentence-transformers",
        "trust_remote_code": True,
        "normalize_embeddings": True,
        "prefix": "",
    },
    "bge-large": {
        "hf_id": "BAAI/bge-large-en-v1.5",
        "loader": "sentence-transformers",
        "trust_remote_code": False,
        "normalize_embeddings": True,
        "prefix": "",
    },
    "e5-large": {
        "hf_id": "intfloat/e5-large-v2",
        "loader": "sentence-transformers",
        "trust_remote_code": False,
        "normalize_embeddings": True,
        "prefix": "passage: ",
    },
    "all-mpnet": {
        "hf_id": "sentence-transformers/all-mpnet-base-v2",
        "loader": "sentence-transformers",
        "trust_remote_code": False,
        "normalize_embeddings": True,
        "prefix": "",
    },
    "specter2": {
        "hf_id": "allenai/specter2_base",
        "loader": "specter2",
        "trust_remote_code": False,
        "normalize_embeddings": False,
        "prefix": "",
    },
}

PROJECTIONS: dict[str, dict] = {
    "umap": {
        "method": "umap",
        "n_neighbors": 15,
        "min_dist": 0.12,
        "n_epochs": 200,
    },
    "umap-tight": {
        "method": "umap",
        "n_neighbors": 30,
        "min_dist": 0.05,
        "n_epochs": 500,
    },
    "pacmap": {
        "method": "pacmap",
        "n_neighbors": None,  # auto (10)
        "mn_ratio": 0.5,  # mid-near pairs weight
        "fp_ratio": 2.0,  # further-point pairs weight
    },
    "densmap": {
        "method": "densmap",  # UMAP with densmap=True
        "n_neighbors": 30,
        "min_dist": 0.05,
        "n_epochs": 500,
    },
}

DEFAULT_PROJ = "umap-tight"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def build_embeddings(
    papers: list[dict], model_slug: str, batch_size: int = 32
) -> np.ndarray:
    cfg = MODELS[model_slug]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Loading {cfg['hf_id']}…")

    if cfg["loader"] == "specter2":
        return _encode_specter2(papers, cfg, device, batch_size)
    else:
        return _encode_sentence_transformers(papers, cfg, device, batch_size)


def _encode_sentence_transformers(
    papers: list[dict], cfg: dict, device: str, batch_size: int
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        cfg["hf_id"],
        trust_remote_code=cfg["trust_remote_code"],
        device=device,
    )
    prefix = cfg.get("prefix", "")
    texts = [
        f"{prefix}{p.get('title', '')}. {p.get('abstract', '') or ''}"
        for p in papers
    ]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=cfg["normalize_embeddings"],
    )
    return embeddings.astype(np.float32)


def _encode_specter2(
    papers: list[dict], cfg: dict, device: str, batch_size: int
) -> np.ndarray:
    from adapters import AutoAdapterModel
    from tqdm import tqdm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    model = AutoAdapterModel.from_pretrained(cfg["hf_id"])
    model.load_adapter("allenai/specter2", source="hf", set_active=True)
    model.to(device)
    model.eval()

    texts = [
        f"{p.get('title', '')} {tokenizer.sep_token} {p.get('abstract', '') or ''}"
        for p in papers
    ]

    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        all_emb.append(out.last_hidden_state[:, 0, :].cpu().numpy())

    return np.vstack(all_emb).astype(np.float32)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def build_projection(embeddings: np.ndarray, proj_slug: str) -> np.ndarray:
    cfg = PROJECTIONS[proj_slug]
    logger.info(f"Running projection '{proj_slug}'…")

    if cfg["method"] in ("umap", "densmap"):
        import umap as umap_lib

        reducer = umap_lib.UMAP(
            n_neighbors=cfg["n_neighbors"],
            min_dist=cfg["min_dist"],
            n_epochs=cfg["n_epochs"],
            metric="cosine",
            densmap=(cfg["method"] == "densmap"),
            random_state=42,
            verbose=False,
        )
        return reducer.fit_transform(embeddings).astype(np.float32)

    elif cfg["method"] == "pacmap":
        try:
            import pacmap as pacmap_lib
        except ImportError:
            raise ImportError("pacmap not installed; run: uv add pacmap")
        reducer = pacmap_lib.PaCMAP(
            n_components=2,
            n_neighbors=cfg["n_neighbors"],
            MN_ratio=cfg["mn_ratio"],
            FP_ratio=cfg["fp_ratio"],
            random_state=42,
        )
        return reducer.fit_transform(embeddings, init="pca").astype(np.float32)

    else:
        raise ValueError(f"Unknown projection method: {cfg['method']}")


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------


def build_clusters(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    logger.info(f"Running KMeans (k={n_clusters})…")
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return km.fit_predict(embeddings).astype(np.int32)


# ---------------------------------------------------------------------------
# Extend - add papers to an existing clustering without refitting it
# ---------------------------------------------------------------------------


def extend(model_slug: str, batch_size: int) -> None:
    """Embed papers missing from embeddings_{model}.npz and place them in the
    existing clusters and map, leaving every existing paper exactly where it is.

    Refitting KMeans after adding a year would renumber the clusters and very
    likely redraw some of them, so the hand-written names would no longer fit
    (the 2026-09-08 bug, on purpose). Instead each new paper joins the cluster
    whose centroid is nearest. A converged KMeans centroid is the mean of its
    members, so the centroids are recomputed from the stored embeddings and
    the fitted model does not need to have been saved.

    The names stay valid only because the old papers did not move, so that is
    checked first: the label file's fingerprint must match the stored
    partition, or this refuses. The file is then restamped for the extended
    partition, recording the fingerprint it was extended from.
    """
    emb_path = Path(f"data/processed/embeddings_{model_slug}.npz")
    cl_path = cl.path_for(model_slug)

    data = np.load(emb_path, allow_pickle=True)
    old_emb = data["embeddings"]
    old_cids = data["cluster_ids"]
    old_ids = [str(x) for x in data["paper_ids"]]
    old_fp = cl.fingerprint(old_ids, old_cids)
    labels = cl.verify(cl_path, old_fp)

    with open(ALL_JSON, encoding="utf-8") as f:
        papers = json.load(f)
    known = set(old_ids)
    new = [p for p in papers if p["paper_id"] not in known]
    if not new:
        logger.info("No papers missing from the embeddings; nothing to extend.")
        return
    logger.info(f"Extending with {len(new)} new papers "
                f"({dict(sorted(Counter(p['year'] for p in new).items()))})")

    new_emb = build_embeddings(new, model_slug, batch_size=batch_size)

    k = int(old_cids.max()) + 1
    centroids = np.stack(
        [old_emb[old_cids == c].astype(np.float64).mean(axis=0) for c in range(k)]
    )

    def nearest(emb):
        d = ((emb[:, None, :].astype(np.float64) - centroids[None]) ** 2).sum(-1)
        return d.argmin(1).astype(np.int32), np.sqrt(d.min(1))

    # Assigning the fitted papers the same way must give back their own
    # clusters; if it does not, these centroids are not the ones KMeans
    # converged to and the assignment below would mean nothing.
    fit_assign, fit_dist = nearest(old_emb)
    agree = float((fit_assign == old_cids).mean())
    if agree < 0.999:
        raise RuntimeError(
            f"Recomputed centroids reproduce only {agree:.2%} of the stored "
            f"assignment; refusing to extend."
        )
    new_cids, new_dist = nearest(new_emb)

    # Does the new year fit the clusters? Compare its distance to the nearest
    # centroid with the fitted papers' distance to their own.
    q = [50, 90, 99]
    logger.info(f"Distance to nearest centroid, fitted papers p50/p90/p99: "
                f"{np.round(np.percentile(fit_dist, q), 3).tolist()}")
    logger.info(f"Distance to nearest centroid, new papers    p50/p90/p99: "
                f"{np.round(np.percentile(new_dist, q), 3).tolist()}")
    p99 = np.percentile(fit_dist, 99)
    logger.info(f"New papers beyond the fitted p99: "
                f"{int((new_dist > p99).sum())} of {len(new)}")

    all_emb = np.vstack([old_emb, new_emb]).astype(np.float32)
    all_cids = np.concatenate([old_cids, new_cids]).astype(np.int32)
    all_ids = old_ids + [p["paper_id"] for p in new]

    # Projections. UMAP refit on the old embeddings with the same seed
    # reproduces the stored coordinates exactly (umap 0.5.12, single thread),
    # so the fitted reducer is rebuilt, proven equal to what is on disk, and
    # used to transform the new papers onto the existing map. densMAP has no
    # transform and PaCMAP no reproducibility guarantee, so those files are
    # left at the old length; activate() refuses a short file.
    import umap as umap_lib

    for slug, pcfg in PROJECTIONS.items():
        proj_path = Path(f"data/processed/proj_{model_slug}_{slug}.npy")
        if not proj_path.exists():
            continue
        if pcfg["method"] != "umap":
            logger.warning(f"{proj_path}: {pcfg['method']} cannot place new "
                           f"points; left at {len(old_ids)} rows")
            continue
        stored = np.load(proj_path)
        reducer = umap_lib.UMAP(
            n_neighbors=pcfg["n_neighbors"], min_dist=pcfg["min_dist"],
            n_epochs=pcfg["n_epochs"], metric="cosine", random_state=42,
            verbose=False,
        )
        refit = reducer.fit_transform(old_emb).astype(np.float32)
        if not np.array_equal(refit, stored):
            raise RuntimeError(
                f"Refitting {slug} does not reproduce {proj_path} (max diff "
                f"{float(np.abs(refit - stored).max()):.4g}); the existing map "
                f"would move. Refusing to extend."
            )
        placed = reducer.transform(new_emb).astype(np.float32)
        np.save(proj_path, np.vstack([stored, placed]))
        logger.info(f"Saved {proj_path} ({len(all_ids)} rows)")

    np.savez(emb_path, embeddings=all_emb, cluster_ids=all_cids,
             paper_ids=np.array(all_ids))
    logger.info(f"Saved {emb_path}")

    cl.write(cl_path, labels, cl.fingerprint(all_ids, all_cids), model_slug,
             extended_from=old_fp, fit_papers=len(old_ids))
    logger.info(f"Restamped {cl_path} for the extended partition")

    dist = Counter(int(c) for c in new_cids)
    for c in range(k):
        logger.info(f"  Cluster {c:2d} +{dist.get(c, 0):4d}  {labels.get(str(c))}")


# ---------------------------------------------------------------------------
# Activate - patches miccai_all.json from saved files
# ---------------------------------------------------------------------------


def activate(model_slug: str, proj_slug: str) -> None:
    emb_path = Path(f"data/processed/embeddings_{model_slug}.npz")
    proj_path = Path(f"data/processed/proj_{model_slug}_{proj_slug}.npy")
    cl_path = Path(f"data/processed/cluster_labels_{model_slug}.json")

    data = np.load(emb_path, allow_pickle=True)
    paper_ids = list(data["paper_ids"])
    cluster_ids = data["cluster_ids"]

    if proj_path.exists():
        coords = np.load(proj_path)
    elif "umap_coords" in data:
        logger.warning(
            f"{proj_path} not found; falling back to umap_coords in {emb_path}"
        )
        coords = data["umap_coords"]
    else:
        raise FileNotFoundError(
            f"{proj_path} not found. Run: "
            f"python analysis/embed.py --model {model_slug} --proj {proj_slug} --no-recompute"
        )
    if len(coords) != len(paper_ids):
        raise ValueError(
            f"{proj_path} has {len(coords)} rows for {len(paper_ids)} papers; "
            f"it was not extended with the newest papers. Re-project it: "
            f"python analysis/embed.py --model {model_slug} --proj {proj_slug} "
            f"--no-recompute"
        )

    # Refuse to activate names that were written for another partition.
    # KMeans numbering is arbitrary, so a relabelled run reuses all twenty
    # names on the wrong clusters and nothing downstream can notice.
    labels = cl.verify(cl_path, cl.fingerprint(paper_ids, cluster_ids))

    with open(ALL_JSON, encoding="utf-8") as f:
        papers = json.load(f)

    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    for p in papers:
        i = id_to_idx.get(p["paper_id"])
        if i is not None:
            p["umap_x"] = float(coords[i, 0])
            p["umap_y"] = float(coords[i, 1])
            p["cluster_id"] = int(cluster_ids[i])
            p["cluster_label"] = labels.get(
                str(cluster_ids[i]), f"Cluster {cluster_ids[i]}"
            )

    with open(ALL_JSON, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    logger.info(
        f"Updated {ALL_JSON} with '{proj_slug}' coords + cluster fields"
    )

    shutil.copy(cl_path, "data/processed/cluster_labels.json")
    ACTIVE_FILE.write_text(model_slug)
    ACTIVE_PROJ_FILE.write_text(proj_slug)
    logger.info(f"Active: model={model_slug}, proj={proj_slug}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build embeddings, projections, and KMeans clusters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODELS),
        help="Embedding model slug",
    )
    parser.add_argument(
        "--proj",
        choices=list(PROJECTIONS),
        default=None,
        help="Projection to compute (default: all)",
    )
    parser.add_argument(
        "--no-recompute",
        action="store_true",
        help="Skip embeddings+KMeans if embeddings_{model}.npz already exists",
    )
    parser.add_argument(
        "--no-reproject",
        action="store_true",
        help="Skip projection if proj_{model}_{proj}.npy already exists",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=20, help="Number of KMeans clusters"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Encoding batch size"
    )
    parser.add_argument(
        "--extend",
        action="store_true",
        help="Embed only papers missing from embeddings_{model}.npz and add "
        "them to the existing clusters and maps; nothing existing moves",
    )
    args = parser.parse_args()

    if args.extend:
        extend(args.model, args.batch_size)
        proj = ACTIVE_PROJ_FILE.read_text().strip()
        activate(args.model, proj)
        return 0

    emb_path = Path(f"data/processed/embeddings_{args.model}.npz")
    cl_path = Path(f"data/processed/cluster_labels_{args.model}.json")
    projs_to_run = [args.proj] if args.proj else list(PROJECTIONS)

    # 1. Embeddings + KMeans
    if args.no_recompute and emb_path.exists():
        logger.info(
            f"{emb_path} exists; skipping embeddings+KMeans (--no-recompute)"
        )
        data = np.load(emb_path, allow_pickle=True)
        embeddings = data["embeddings"]
        cluster_ids = data["cluster_ids"]
        paper_ids = list(data["paper_ids"])
    else:
        with open(ALL_JSON, encoding="utf-8") as f:
            papers = json.load(f)
        logger.info(f"Loaded {len(papers)} papers")

        embeddings = build_embeddings(
            papers, args.model, batch_size=args.batch_size
        )
        cluster_ids = build_clusters(embeddings, n_clusters=args.n_clusters)
        paper_ids = [p["paper_id"] for p in papers]

        np.savez(
            emb_path,
            embeddings=embeddings.astype(np.float32),
            cluster_ids=cluster_ids,
            paper_ids=np.array(paper_ids),
        )
        logger.info(f"Saved {emb_path}")

        # A fresh KMeans run renumbers the clusters, so any names already on
        # disk now sit on the wrong groups. Say so here, at the moment it
        # becomes true, rather than leaving the stale file to be found later
        # by whoever reads the map legend. The file is not rewritten: those
        # names are hand-written and are usually still the right twenty names
        # in the wrong order, which is worth keeping while renaming.
        fp = cl.fingerprint(paper_ids, cluster_ids)
        if not cl_path.exists():
            cl.write(
                cl_path,
                {str(i): f"Cluster {i}" for i in range(args.n_clusters)},
                fp,
                args.model,
            )
            logger.info(
                f"Created placeholder {cl_path}; name the clusters before "
                f"building charts (python analysis/describe_clusters.py "
                f"{args.model})"
            )
        else:
            _, stamped, _ = cl.read(cl_path)
            if stamped != fp:
                logger.warning(
                    f"{cl_path} was written for a different clustering. Every "
                    f"name in it is now on the wrong cluster, and the build "
                    f"will refuse to use it. Re-read the clusters and rename:"
                )
                logger.warning(
                    f"    python analysis/describe_clusters.py {args.model}"
                )
                logger.warning(
                    f"    python analysis/describe_clusters.py --stamp {args.model}"
                )

    # 2. Projections
    for proj_slug in projs_to_run:
        proj_path = Path(f"data/processed/proj_{args.model}_{proj_slug}.npy")
        if args.no_reproject and proj_path.exists():
            logger.info(f"{proj_path} exists; skipping (--no-reproject)")
            continue
        coords = build_projection(embeddings, proj_slug)
        np.save(proj_path, coords)
        logger.info(f"Saved {proj_path}")

    # 3. Activate best available projection
    activate_proj = DEFAULT_PROJ
    if not Path(
        f"data/processed/proj_{args.model}_{activate_proj}.npy"
    ).exists():
        # Fall back to first proj that was actually computed/exists
        for p in projs_to_run:
            if Path(f"data/processed/proj_{args.model}_{p}.npy").exists():
                activate_proj = p
                break

    if cl_path.exists():
        activate(args.model, activate_proj)
    else:
        logger.warning(
            f"{cl_path} not found; skipping activate. Edit it and run use_model.py."
        )

    # 4. Cluster size summary
    dist = dict(sorted(Counter(int(c) for c in cluster_ids).items()))
    logger.info("\nCluster size distribution:")
    for cid, count in dist.items():
        logger.info(f"  Cluster {cid:2d}: {count:4d} papers")

    return 0


if __name__ == "__main__":
    sys.exit(main())
