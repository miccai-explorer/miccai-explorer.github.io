"""Guards on website/papers.json, the data file behind the All Papers page.

The counts here are not arbitrary. Every one of them is a number the page
displays or filters on, so a change that silently drops papers, loses a
presentation type, or breaks the subject-area lookup moves one of these and
fails here rather than shipping a browse page that quietly omits papers.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import oral_stats  # noqa: E402
import papers_index  # noqa: E402

ALL_JSON = ROOT / "data" / "processed" / "miccai_all.json"

N_PAPERS = 3717
N_AREAS = 161
PER_YEAR = {2021: 531, 2022: 573, 2023: 730, 2024: 856, 2025: 1027}
SCALES = {2021: 9, 2022: 8, 2023: 8, 2024: 6, 2025: 6}
TYPE_COUNTS = {0: 3340, 1: 323, 2: 54}


@pytest.fixture(scope="module")
def papers():
    with open(ALL_JSON, encoding="utf-8") as f:
        data = json.load(f)
    oral_stats.attach_types(data, oral_stats.load_orals())
    return data


@pytest.fixture(scope="module")
def index(papers, tmp_path_factory):
    out = tmp_path_factory.mktemp("idx") / "papers.json"
    return papers_index.build(papers, out, built="2026-08-22")


def test_every_paper_appears_exactly_once(index):
    ids = [p["id"] for p in index["papers"]]
    assert len(ids) == N_PAPERS
    assert len(set(ids)) == N_PAPERS


def test_per_year_counts(index):
    counts = {}
    for p in index["papers"]:
        counts[p["year"]] = counts.get(p["year"], 0) + 1
    assert counts == PER_YEAR


def test_area_ids_all_resolve(index):
    n = len(index["areas"])
    assert n == N_AREAS
    for p in index["papers"]:
        for a in p["areas"]:
            assert 0 <= a < n


def test_scales_read_from_data_not_hardcoded(index):
    assert {int(y): v for y, v in index["scales"].items()} == SCALES


def test_presentation_type_counts_match_orals(index):
    counts = {0: 0, 1: 0, 2: 0}
    for p in index["papers"]:
        counts[p["type"]] += 1
    assert counts == TYPE_COUNTS


def test_normalized_score_matches_raw_and_scale(index):
    scales = {int(y): v for y, v in index["scales"].items()}
    for p in index["papers"]:
        expected = (p["score"] - 1) / (scales[p["year"]] - 1)
        assert abs(p["score_norm"] - expected) < 1e-3, p["id"]


def test_no_null_reviewer_scores(index):
    # Every review should have a score. A null one used to mean the parser had
    # dropped fields (see the 2026-08-22 parser fixes); the index filters nulls
    # defensively, but if this ever fails the right response is to fix the
    # parser, not to accept the gap.
    for p in index["papers"]:
        assert all(isinstance(s, int) for s in p["reviews"]), p["id"]


def test_review_counts_match_source(papers, index):
    # Guards the filtering above: a paper's reviews array must not be shorter
    # than the number of reviews it actually has. This is what would have caught
    # miccai-2024-Paper0405, where one reviewer's whole field list was lost.
    by_id = {p["paper_id"]: p for p in papers}
    for rec in index["papers"]:
        assert len(rec["reviews"]) == len(by_id[rec["id"]]["reviews"]), rec["id"]


def test_absent_links_are_omitted_not_null(index):
    optional = (
        "code_url", "pdf_url", "doi", "sharedit_url",
        "supp_url", "lncs", "dataset_urls", "session",
    )
    for p in index["papers"]:
        for key in optional:
            if key in p:
                assert p[key], f"{p['id']}: {key} present but empty"


def test_session_only_on_talks(index):
    for p in index["papers"]:
        if p["type"] == 0:
            assert "session" not in p


def test_builds_without_orals_data(papers, tmp_path):
    # A checkout without data/processed/orals.json must still build, with every
    # paper falling back to poster. This mirrors how the Orals page degrades and
    # keeps the new page from becoming a hard dependency on the oral pipeline.
    stripped = [
        {k: v for k, v in p.items()
         if k not in ("presentation_type", "oral_session")}
        for p in papers
    ]
    idx = papers_index.build(stripped, tmp_path / "p.json", built="2026-08-22")
    assert all(p["type"] == 0 for p in idx["papers"])
    assert all("session" not in p for p in idx["papers"])


def test_written_file_is_valid_json_and_compact(papers, tmp_path):
    # Compactness is worth about 100 KB over the wire, so assert the file really
    # is the compact form. Searching the text for ", " does not work: subject
    # area names contain it ("Diagnosis, Treatment Response, and Outcome
    # Prediction"). Re-dumping the parsed object and comparing does.
    out = tmp_path / "papers.json"
    papers_index.build(papers, out, built="2026-08-22")
    text = out.read_text(encoding="utf-8")
    data = json.loads(text)
    assert data["built"] == "2026-08-22"
    assert text == json.dumps(data, separators=(",", ":"))
