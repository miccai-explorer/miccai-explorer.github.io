"""Known-answer tests for analysis/oral_stats.py.

Run from the repo root:  python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import oral_stats as os_  # noqa: E402

# ---------------------------------------------------------------- Cliff's delta


def test_cliffs_delta_complete_separation():
    """Every value of a below every value of b → delta = -1."""
    delta, mag = os_.cliffs_delta([1, 2, 3], [4, 5, 6])
    assert delta == pytest.approx(-1.0)
    assert mag == "large"


def test_cliffs_delta_complete_separation_reversed():
    delta, mag = os_.cliffs_delta([4, 5, 6], [1, 2, 3])
    assert delta == pytest.approx(1.0)
    assert mag == "large"


def test_cliffs_delta_identical_samples_is_zero():
    """3 pairs where a>b and 3 where a<b out of 9 → delta = 0."""
    delta, mag = os_.cliffs_delta([1, 2, 3], [1, 2, 3])
    assert delta == pytest.approx(0.0)
    assert mag == "negligible"


def test_cliffs_delta_handles_ties():
    """Ties count toward neither direction."""
    delta, _ = os_.cliffs_delta([2, 2], [2, 2])
    assert delta == pytest.approx(0.0)


def test_cliffs_delta_known_fraction():
    # a=[1,3], b=[2,2]: 3>2 twice → gt=2; 1<2 twice → lt=2; delta = 0/4
    assert os_.cliffs_delta([1, 3], [2, 2])[0] == pytest.approx(0.0)
    # a=[3,3], b=[1,2]: all greater → 4/4 = 1
    assert os_.cliffs_delta([3, 3], [1, 2])[0] == pytest.approx(1.0)


def test_cliffs_delta_empty_is_nan():
    delta, mag = os_.cliffs_delta([], [1, 2])
    assert np.isnan(delta)
    assert mag == "n/a"


@pytest.mark.parametrize(
    "n_above,expected_delta,expected_mag",
    [
        (50, 0.00, "negligible"),
        (57, 0.14, "negligible"),   # just under the .147 cut point
        (60, 0.20, "small"),
        (70, 0.40, "medium"),
        (80, 0.60, "large"),
    ],
)
def test_cliffs_delta_magnitude_thresholds(n_above, expected_delta, expected_mag):
    """Romano et al. cut points: .147 / .33 / .474.

    Group a holds `n_above` values above every b and the rest below, so
    delta is (n_above - (100 - n_above)) / 100 exactly.
    """
    a = [2] * n_above + [0] * (100 - n_above)
    d, mag = os_.cliffs_delta(a, [1] * 100)
    assert d == pytest.approx(expected_delta)
    assert mag == expected_mag


# ------------------------------------------------------------------ Wilson CI


def test_wilson_ci_known_value():
    """1/10 → (0.0179, 0.4042), the standard published Wilson interval."""
    p, lo, hi = os_.wilson_ci(1, 10)
    assert p == pytest.approx(0.1)
    assert lo == pytest.approx(0.0179, abs=1e-3)
    assert hi == pytest.approx(0.4042, abs=1e-3)


def test_wilson_ci_stays_in_unit_interval():
    """The normal approximation would run below 0 here; Wilson must not."""
    _, lo, hi = os_.wilson_ci(0, 12)
    assert lo == 0.0
    assert 0.0 < hi < 1.0
    _, lo, hi = os_.wilson_ci(12, 12)
    assert hi == pytest.approx(1.0)
    assert 0.0 < lo < 1.0


def test_wilson_ci_zero_n_is_nan():
    p, lo, hi = os_.wilson_ci(0, 0)
    assert np.isnan(p) and np.isnan(lo) and np.isnan(hi)


# ------------------------------------------------------------------- bootstrap


def test_bootstrap_ci_constant_sample_has_zero_width():
    point, lo, hi = os_.bootstrap_ci([5.0] * 20)
    assert point == pytest.approx(5.0)
    assert lo == pytest.approx(5.0)
    assert hi == pytest.approx(5.0)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    vals = rng.normal(10, 2, 200).tolist()
    point, lo, hi = os_.bootstrap_ci(vals)
    assert lo < point < hi


def test_bootstrap_ci_is_deterministic():
    vals = [1, 2, 3, 4, 5, 6, 7, 8]
    assert os_.bootstrap_ci(vals) == os_.bootstrap_ci(vals)


def test_bootstrap_ci_tiny_sample_returns_nan_bounds():
    """Two observations cannot support an interval; say so rather than fake one."""
    point, lo, hi = os_.bootstrap_ci([1.0, 2.0])
    assert point == pytest.approx(1.5)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_empty_is_nan():
    point, lo, hi = os_.bootstrap_ci([])
    assert np.isnan(point)


# ------------------------------------------------------------------ confidence


def test_confidence_ordinal_all_four_levels():
    assert os_.confidence_ordinal("Not confident") == 1
    assert os_.confidence_ordinal("Somewhat confident") == 2
    assert os_.confidence_ordinal("Confident but not absolutely certain") == 3
    assert os_.confidence_ordinal("Very confident") == 4


def test_confidence_ordinal_is_case_and_space_insensitive():
    """Capitalization differs between years ('Somewhat Confident' in 2022)."""
    assert os_.confidence_ordinal("somewhat CONFIDENT") == 2
    assert os_.confidence_ordinal("  Very   confident  ") == 4


def test_confidence_ordinal_unknown_is_none():
    assert os_.confidence_ordinal(None) is None
    assert os_.confidence_ordinal("") is None
    assert os_.confidence_ordinal("Extremely sure") is None


def test_mean_confidence_ignores_unparseable_labels():
    paper = {
        "reviews": [
            {"confidence_label": "Very confident"},          # 4
            {"confidence_label": "Somewhat confident"},      # 2
            {"confidence_label": "Mysterious"},              # skipped
        ]
    }
    assert os_.mean_confidence(paper) == pytest.approx(3.0)


def test_mean_confidence_none_when_no_labels():
    assert os_.mean_confidence({"reviews": []}) is None


# ----------------------------------------------------------------- type join


def _papers():
    return [
        {"paper_id": "p1", "year": 2025},
        {"paper_id": "p2", "year": 2025},
        {"paper_id": "p3", "year": 2025},
    ]


def test_attach_tiers_marks_unlisted_papers_as_poster():
    orals = {
        "papers": {
            "p1": {"type": "oral", "session_id": "3", "session_title": "S"},
            "p2": {"type": "spotlight", "session_id": "3", "session_title": "S"},
        },
        "summary": {"2025": {}},
    }
    papers = _papers()
    n = os_.attach_types(papers, orals)
    assert n == 2
    assert [p["presentation_type"] for p in papers] == [
        "oral",
        "spotlight",
        "poster",
    ]
    assert papers[0]["oral_session"] == "S"
    assert papers[2]["oral_session"] is None


def test_attach_tiers_without_data_leaves_tier_none():
    """No orals.json → the feature is simply absent, not silently wrong."""
    papers = _papers()
    n = os_.attach_types(papers, {})
    assert n == 0
    assert all(p["presentation_type"] is None for p in papers)


def test_rate_by_tier_counts_and_intervals():
    orals = {
        "papers": {
            "p1": {"type": "oral", "session_id": "1", "session_title": "S"},
            "p2": {"type": "oral", "session_id": "1", "session_title": "S"},
        },
        "summary": {"2025": {}},
    }
    papers = _papers()
    papers[0]["has_code"] = True
    papers[1]["has_code"] = False
    papers[2]["has_code"] = True
    os_.attach_types(papers, orals)
    out = os_.rate_by_type(papers, 2025, lambda p: p.get("has_code"))
    assert out["oral"][3:] == (1, 2)      # k, n
    assert out["poster"][3:] == (1, 1)
    assert out["oral"][0] == pytest.approx(0.5)
