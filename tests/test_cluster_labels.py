"""Guards on the cluster names shown in the semantic map legend.

Every page on the site carries the map, and the map's legend is these twenty
names. They are hand-written against a KMeans partition whose numbering is
arbitrary, so recomputing the embeddings renames every cluster without
changing a single character of the file. That is exactly what happened to
`specter2` before 2026-09-08, and nothing failed: the map drew, the legend
filled, the counts were right, and the names were wrong.

These tests make that state impossible to ship. They check the binding, not
the wording; whether "Retinal & Ophthalmic Imaging" is a better name than
"Retinal Disease Analysis" is a human call, but whether it is attached to the
retina cluster is not.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import cluster_labels as cl  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
ALL_JSON = PROCESSED / "miccai_all.json"
N_CLUSTERS = 20
# Measured against the fixed r=300 legend margin in map_all.html.
MAX_NAME_CHARS = 37


def _models():
    return sorted(
        p.stem.replace("embeddings_", "")
        for p in PROCESSED.glob("embeddings_*.npz")
    )


@pytest.fixture(scope="module")
def papers():
    with open(ALL_JSON, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("model", _models())
def test_labels_match_their_clustering(model):
    """Each model's names were written for that model's partition."""
    path = cl.path_for(model)
    if not path.exists():
        pytest.skip(f"no label file for {model}")
    data = np.load(PROCESSED / f"embeddings_{model}.npz", allow_pickle=True)
    expected = cl.fingerprint(data["paper_ids"], data["cluster_ids"])
    labels = cl.verify(path, expected)
    assert len(labels) == N_CLUSTERS
    assert all(labels[str(k)].strip() for k in range(N_CLUSTERS))


def test_active_labels_match_the_published_clustering(papers):
    """cluster_labels.json describes the clustering baked into miccai_all.json.

    This is the one the site actually renders: build_charts.py patches every
    paper's cluster_label from this file, keyed on the cluster_id already in
    miccai_all.json. If the two disagree, the legend is wrong on every page.
    """
    expected = cl.fingerprint_of_papers(papers)
    labels = cl.verify(PROCESSED / "cluster_labels.json", expected)
    assert len(labels) == N_CLUSTERS


def test_active_copy_matches_the_active_model(papers):
    """cluster_labels.json is a copy of the active model's file, not a fork."""
    model = (PROCESSED / "active_model.txt").read_text().strip()
    assert cl.read(PROCESSED / "cluster_labels.json") == cl.read(
        cl.path_for(model)
    )


def test_every_paper_has_a_named_cluster(papers):
    """No paper falls through to a bare 'Cluster 7' in the legend."""
    labels, _, _ = cl.read(PROCESSED / "cluster_labels.json")
    missing = {
        p["cluster_id"]
        for p in papers
        if str(p.get("cluster_id")) not in labels
    }
    assert not missing, f"cluster ids with no name: {sorted(missing)}"


def test_names_fit_the_map_legend(papers):
    """map_all.html cuts names off past roughly this width.

    Its legend lives in a fixed `r=300` margin with `autoexpand=False`, so the
    plot keeps its width when the reader switches between the 5-entry Year
    legend and the 20-entry Cluster legend. The cost is that a long name is
    truncated rather than given room, with nothing failing: three names in the
    2026-09-08 rewrite were cut off and the page looked fine.
    """
    labels, _, _ = cl.read(PROCESSED / "cluster_labels.json")
    too_long = {k: v for k, v in labels.items() if len(v) > MAX_NAME_CHARS}
    assert not too_long, f"names past {MAX_NAME_CHARS} chars get clipped: {too_long}"


def test_names_are_distinct(papers):
    """Two clusters sharing a name make the legend unreadable."""
    labels, _, _ = cl.read(PROCESSED / "cluster_labels.json")
    names = [v.strip().lower() for v in labels.values()]
    assert len(set(names)) == len(names), "duplicate cluster names"


# --- the guard, re-breaking the bug it exists for -------------------------
#
# A check that never fails looks exactly like a check that passes, so each of
# these reproduces a way the names have gone or could go wrong and asserts
# the loader refuses it.


def test_renumbered_clustering_is_rejected(papers, tmp_path):
    """The actual 2026-09-08 bug: same groups, different numbering.

    Rotating every cluster id by one leaves the partition, the sizes and the
    colours identical and moves all twenty names one cluster over. Nothing
    downstream can see it; the fingerprint must.
    """
    rotated = [
        {"paper_id": p["paper_id"], "cluster_id": (p["cluster_id"] + 1) % 20}
        for p in papers
    ]
    path = PROCESSED / "cluster_labels.json"
    with pytest.raises(cl.ClusteringMismatch, match="different clustering"):
        cl.verify(path, cl.fingerprint_of_papers(rotated))


def test_reassigned_paper_is_rejected(papers, tmp_path):
    """One paper moving cluster is enough to invalidate the binding."""
    moved = [
        {"paper_id": p["paper_id"], "cluster_id": p["cluster_id"]}
        for p in papers
    ]
    moved[0]["cluster_id"] = (moved[0]["cluster_id"] + 1) % 20
    with pytest.raises(cl.ClusteringMismatch):
        cl.verify(
            PROCESSED / "cluster_labels.json", cl.fingerprint_of_papers(moved)
        )


def test_unstamped_legacy_file_is_rejected(tmp_path):
    """The flat {"0": "name"} shape carries no proof and is not trusted."""
    legacy = tmp_path / "cluster_labels_legacy.json"
    legacy.write_text(json.dumps({str(i): f"Name {i}" for i in range(20)}))
    labels, clustering, _ = cl.read(legacy)
    assert len(labels) == 20 and clustering is None
    with pytest.raises(cl.ClusteringMismatch, match="no 'clustering'"):
        cl.verify(legacy, "whatever")


def test_roundtrip_preserves_names_and_binding(tmp_path):
    names = {str(i): f"Name {i}" for i in range(20)}
    path = tmp_path / "cluster_labels_rt.json"
    cl.write(path, names, "abc123", "somemodel")
    labels, clustering, model = cl.read(path)
    assert labels == names and clustering == "abc123" and model == "somemodel"
    assert cl.verify(path, "abc123") == names
