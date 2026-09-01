#!/usr/bin/env python3
"""
Statistics and the type join for the oral / spotlight analysis.

Pure functions plus one loader; no plotting, no file writing, so the numbers
behind the charts can be tested directly.

Why these particular statistics
-------------------------------
Review scores are ordinal on a short integer scale (1-6, 1-8 or 1-9 depending on
the year), so Cohen's d and t-based intervals do not apply: their interval and
normality assumptions are not met and the effect size would not be comparable
across years whose scales differ. Cliff's delta is used instead; it asks only
"how often does a paper from group A outscore one from group B", which is
meaningful on any ordinal scale and is directly comparable between years.

Proportions (early-accept rate, code-release rate) use Wilson intervals rather
than the normal approximation, because several per-type cells are small enough
(a dozen spotlight papers in a year) that the normal approximation would produce
intervals running past 0 or 1.

Confidence encoding
-------------------
CLAUDE.md documents `confidence_raw` as null before 2024, which would restrict a
confidence analysis to two years. But `confidence_label` is populated for every
review in all five years, on a consistent four-level scale (capitalization
varies between years). Encoding the label gives a five-year analysis.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ORALS_JSON = Path("data/processed/orals.json")

TYPE_ORDER = ["oral", "spotlight", "poster"]
TYPE_LABEL = {
    "oral": "Oral",
    "spotlight": "Spotlight",
    "poster": "Poster only",
}

# A type palette distinct from the per-year logo colors, so type identity reads
# the same on every chart of the page regardless of which year is shown.
TYPE_COLORS = {
    "oral": "#7c3aed",       # violet
    "spotlight": "#0891b2",  # teal
    "poster": "#94a3b8",     # slate; deliberately recessive, it is the baseline
}

CONFIDENCE_LEVELS = {
    "not confident": 1,
    "somewhat confident": 2,
    "confident but not absolutely certain": 3,
    "very confident": 4,
}
CONFIDENCE_LABEL = {
    1: "Not confident",
    2: "Somewhat confident",
    3: "Confident, not certain",
    4: "Very confident",
}


# ---------------------------------------------------------------------------
# Type join
# ---------------------------------------------------------------------------


def load_orals(path: Path = ORALS_JSON) -> dict:
    """Load data/processed/orals.json, or {} when the file is absent."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def attach_types(papers: list, orals: dict) -> int:
    """Set presentation_type / oral_session on each paper, in memory.

    Mirrors how build_charts.py already patches cluster labels from
    cluster_labels.json: miccai_all.json is never rewritten, so re-running
    normalize.py cannot silently drop the type data and no ordering dependency
    is introduced into the pipeline.

    Every accepted MICCAI paper gets a poster; orals and spotlights are
    additional slots. So "poster" means poster-only and is the comparison
    group, not a leftover bucket.

    Returns the number of papers matched to a presentation.
    """
    by_id = orals.get("papers", {}) if orals else {}
    n = 0
    for p in papers:
        rec = by_id.get(p["paper_id"])
        if rec:
            p["presentation_type"] = rec["type"]
            p["oral_session"] = rec.get("session_title")
            p["oral_session_id"] = rec.get("session_id")
            n += 1
        else:
            p["presentation_type"] = "poster" if by_id else None
            p["oral_session"] = None
            p["oral_session_id"] = None
    return n


# ---------------------------------------------------------------------------
# Review-level helpers
# ---------------------------------------------------------------------------


def confidence_ordinal(label: str | None) -> int | None:
    """Four-level reviewer confidence → 1-4, or None if unrecognized."""
    if not label:
        return None
    key = re.sub(r"\s+", " ", label.strip().lower())
    return CONFIDENCE_LEVELS.get(key)


def paper_confidences(paper: dict) -> list:
    out = []
    for r in paper.get("reviews", []):
        v = confidence_ordinal(r.get("confidence_label"))
        if v is not None:
            out.append(v)
    return out


def mean_confidence(paper: dict) -> float | None:
    vals = paper_confidences(paper)
    return float(np.mean(vals)) if vals else None


def type_values(papers: list, ptype: str, key: str) -> list:
    """Values of `key` for one type, skipping papers where it is missing."""
    return [
        p[key]
        for p in papers
        if p.get("presentation_type") == ptype and p.get(key) is not None
    ]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values,
    stat=np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple:
    """Percentile bootstrap interval for `stat`. Returns (point, lo, hi).

    Returns (point, nan, nan) for fewer than 3 observations, rather than a
    misleadingly tight interval.
    """
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return (float("nan"),) * 3
    point = float(stat(a))
    if a.size < 3:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = stat(a[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def cliffs_delta(a, b) -> tuple:
    """Cliff's delta between samples a and b, plus a magnitude label.

    delta = P(a > b) - P(a < b), in [-1, 1]. Positive means group a tends to
    score higher. Magnitude thresholds follow Romano et al. (2006).
    """
    x = np.asarray([v for v in a if v is not None], dtype=float)
    y = np.asarray([v for v in b if v is not None], dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan"), "n/a"
    # Rank-based rather than an O(n*m) comparison matrix.
    gt = 0
    lt = 0
    ys = np.sort(y)
    for v in x:
        gt += int(np.searchsorted(ys, v, side="left"))
        lt += int(y.size - np.searchsorted(ys, v, side="right"))
    delta = (gt - lt) / (x.size * y.size)
    m = abs(delta)
    if m < 0.147:
        mag = "negligible"
    elif m < 0.33:
        mag = "small"
    elif m < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return float(delta), mag


def cliffs_delta_ci(
    a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple:
    """Bootstrap interval for Cliff's delta. Returns (delta, lo, hi, magnitude)."""
    x = np.asarray([v for v in a if v is not None], dtype=float)
    y = np.asarray([v for v in b if v is not None], dtype=float)
    delta, mag = cliffs_delta(x, y)
    if x.size < 3 or y.size < 3:
        return delta, float("nan"), float("nan"), mag
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        xi = x[rng.integers(0, x.size, x.size)]
        yi = y[rng.integers(0, y.size, y.size)]
        boots[i] = cliffs_delta(xi, yi)[0]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return delta, float(lo), float(hi), mag


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple:
    """Wilson score interval for a proportion. Returns (p, lo, hi) as fractions."""
    if n <= 0:
        return (float("nan"),) * 3
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(p), float(max(0.0, center - half)), float(min(1.0, center + half))


def rate_by_type(papers: list, year: int, predicate) -> dict:
    """{type: (rate, lo, hi, k, n)} for a boolean property, one year."""
    out = {}
    for ptype in TYPE_ORDER:
        sub = [
            p
            for p in papers
            if p["year"] == year and p.get("presentation_type") == ptype
        ]
        if not sub:
            continue
        k = sum(1 for p in sub if predicate(p))
        p_, lo, hi = wilson_ci(k, len(sub))
        out[ptype] = (p_, lo, hi, k, len(sub))
    return out


# ---------------------------------------------------------------------------
# Page summary (consumed by build_site.py → templates/orals.html)
# ---------------------------------------------------------------------------


def page_summary(papers: list, orals: dict) -> dict:
    """Headline numbers for the Orals & Spotlights page.

    Everything the template prints is computed here, so no figure is ever
    hardcoded into the prose and the page cannot drift from the data.
    """
    years = sorted(
        {
            p["year"]
            for p in papers
            if p.get("presentation_type") in ("oral", "spotlight")
        }
    )
    if not years:
        return {}

    per_year = []
    for y in years:
        sub = [p for p in papers if p["year"] == y]
        n_oral = sum(1 for p in sub if p.get("presentation_type") == "oral")
        n_spot = sum(
            1 for p in sub if p.get("presentation_type") == "spotlight"
        )
        a = type_values(sub, "oral", "avg_score_raw")
        b = type_values(sub, "poster", "avg_score_raw")
        delta, lo, hi, mag = cliffs_delta_ci(a, b)
        early = rate_by_type(papers, y, lambda p: p.get("early_accepted"))
        summary = (orals.get("summary", {}) or {}).get(str(y), {})
        per_year.append(
            {
                "year": y,
                "n_papers": len(sub),
                "n_oral": n_oral,
                "n_spotlight": n_spot,
                "n_selected": n_oral + n_spot,
                "pct_selected": round(100 * (n_oral + n_spot) / len(sub), 1),
                "delta": round(delta, 3),
                "delta_lo": round(lo, 3),
                "delta_hi": round(hi, 3),
                "magnitude": mag,
                "early_oral_pct": (
                    round(100 * early["oral"][0], 1)
                    if "oral" in early
                    else None
                ),
                "early_poster_pct": (
                    round(100 * early["poster"][0], 1)
                    if "poster" in early
                    else None
                ),
                "extracted": summary.get("extracted"),
                "matched": summary.get("matched"),
                "match_rate": summary.get("match_rate_pct"),
            }
        )

    total_oral = sum(r["n_oral"] for r in per_year)
    total_spot = sum(r["n_spotlight"] for r in per_year)
    total_papers = sum(r["n_papers"] for r in per_year)
    deltas = [r["delta"] for r in per_year if not np.isnan(r["delta"])]

    return {
        "years": years,
        "per_year": per_year,
        "total_oral": total_oral,
        "total_spotlight": total_spot,
        "total_selected": total_oral + total_spot,
        "total_papers": total_papers,
        "pct_selected": round(
            100 * (total_oral + total_spot) / total_papers, 1
        ),
        "delta_min": min(deltas) if deltas else None,
        "delta_max": max(deltas) if deltas else None,
        "spotlight_years": sorted(
            {r["year"] for r in per_year if r["n_spotlight"]}
        ),
        "all_matched": all(
            r["match_rate"] == 100.0
            for r in per_year
            if r["match_rate"] is not None
        ),
    }
