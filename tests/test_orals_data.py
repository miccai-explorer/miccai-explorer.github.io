"""Guards on the extracted and matched oral/spotlight data.

Two layers:

* Structural checks on the committed JSON, which run anywhere.
* A golden-count check that re-parses the source PDFs and compares against the
  committed extraction. It skips when the PDFs or pdfplumber are unavailable,
  so a checkout without the (large, manually collected) program books still
  passes.

The counts below are not arbitrary. Each year's program is highly regular -
in 2025 every combined session holds 6 spotlights and 6 orals, in 2024 every
combined session holds 3 and 5, so a parser change that drops or duplicates a
presentation moves these numbers and fails here rather than quietly biasing
every statistic on the page.
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

RAW_DIR = ROOT / "data" / "raw"
ORALS_JSON = ROOT / "data" / "processed" / "orals.json"
PDF_DIR = ROOT / "data" / "manually_downloaded" / "OralSchedules"

# year: (sessions, presentations)
GOLDEN = {
    2021: (12, 60),
    2022: (8, 41),
    2023: (12, 68),
    2024: (16, 96),
    2025: (14, 112),
}

YEARS = sorted(GOLDEN)


def _raw(year: int):
    path = RAW_DIR / f"orals_{year}.json"
    if not path.exists():
        pytest.skip(f"{path} not present")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("year", YEARS)
def test_committed_extraction_matches_golden_counts(year):
    data = _raw(year)
    n_sessions, n_pres = GOLDEN[year]
    assert data["n_sessions"] == n_sessions
    assert data["n_presentations"] == n_pres
    assert (
        sum(len(s["presentations"]) for s in data["sessions"]) == n_pres
    )


@pytest.mark.parametrize("year", YEARS)
def test_session_numbering_is_contiguous(year):
    """A gap means a page of the program is missing from the PDF."""
    data = _raw(year)
    ids = [s["session_id"] for s in data["sessions"]]
    numeric = sorted(int(i) for i in ids if str(i).isdigit())
    if numeric:
        assert numeric == list(range(1, max(numeric) + 1))
    assert len(ids) == len(set(ids)), "duplicate session ids"


@pytest.mark.parametrize("year", YEARS)
def test_every_presentation_has_a_usable_title(year):
    data = _raw(year)
    for s in data["sessions"]:
        for p in s["presentations"]:
            title = p["title_raw"].strip()
            assert len(title) > 10, f"suspiciously short title: {title!r}"
            assert not title.startswith("Speaker"), title
            assert not title.startswith("Authors"), title
            assert "Panel Discussion" not in title, title


@pytest.mark.parametrize("year", YEARS)
def test_tiers_are_valid_and_only_recent_years_have_spotlights(year):
    data = _raw(year)
    ptypes = {
        p["type"] for s in data["sessions"] for p in s["presentations"]
    }
    assert ptypes <= {"oral", "spotlight"}
    if year <= 2023:
        assert "spotlight" not in ptypes, (
            "the oral/spotlight split only exists from 2024"
        )


def test_orals_json_matches_every_extracted_presentation():
    if not ORALS_JSON.exists():
        pytest.skip("orals.json not present")
    with open(ORALS_JSON, encoding="utf-8") as f:
        orals = json.load(f)
    for year, summary in orals["summary"].items():
        assert summary["unmatched"] == 0, (
            f"{year}: {summary['unmatched']} presentations did not match a "
            f"paper - add them to data/oral_title_overrides.yaml"
        )
        assert summary["matched"] + summary["skipped"] == summary["extracted"]
    assert len(orals["papers"]) == sum(
        s["matched"] for s in orals["summary"].values()
    )


def test_orals_json_tiers_are_valid():
    if not ORALS_JSON.exists():
        pytest.skip("orals.json not present")
    with open(ORALS_JSON, encoding="utf-8") as f:
        orals = json.load(f)
    for pid, rec in orals["papers"].items():
        assert rec["type"] in ("oral", "spotlight"), pid
        assert pid.startswith("miccai-"), pid


@pytest.mark.parametrize("year", YEARS)
def test_reparsing_the_pdf_reproduces_the_committed_counts(year):
    """End-to-end guard: the parser still reads the PDFs the same way."""
    pytest.importorskip("pdfplumber")
    pdf = PDF_DIR / f"{year}.pdf"
    if not pdf.exists():
        pytest.skip(f"{pdf} not present (program PDFs are collected by hand)")

    import extract_orals

    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["oral_schedules"][year]
    sessions = extract_orals.parse_year(pdf, cfg)
    n_sessions, n_pres = GOLDEN[year]
    assert len(sessions) == n_sessions
    assert sum(len(s["presentations"]) for s in sessions) == n_pres
