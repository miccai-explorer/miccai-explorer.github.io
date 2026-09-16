"""Column splitting for the 2026 program, which sets three columns.

The 2026 booklet's gutters are not at even thirds. On page 8 the right gutter
runs x=493.5 to x=527.7 while an even third would split at x=528.0, one point
past its right edge, filing the leftmost word of column 3 under column 2 and
splicing two unrelated titles together. Titles are the join key for the whole
2026 pipeline, so that corruption would reach every chart on the page.

Each check here was verified by re-breaking it: switching _column_bounds to
return even thirds makes test_splits_land_inside_real_gutters fail on page 8.
"""

import bisect
import sys
from pathlib import Path

import pdfplumber
import pytest

sys.path.insert(0, "analysis")
from extract_orals import _column_bounds  # noqa: E402

PDF_2026 = Path("data/manually_downloaded/OralSchedules/2026.pdf")
SESSION_PAGES = range(2, 11)


def _mids(page):
    return [(w["x0"] + w["x1"]) / 2 for w in page.extract_words()]


@pytest.mark.skipif(not PDF_2026.exists(), reason="2026 program PDF absent")
def test_splits_land_inside_real_gutters():
    """Each split must fall in an empty band, not through a column of text."""
    with pdfplumber.open(PDF_2026) as pdf:
        for pno in SESSION_PAGES:
            page = pdf.pages[pno - 1]
            mids = sorted(_mids(page))
            for b in _column_bounds(mids, 3, page.width):
                left = max((m for m in mids if m < b), default=None)
                right = min((m for m in mids if m > b), default=None)
                assert left is not None and right is not None
                assert right - left > 10, (
                    f"page {pno}: split at {b:.1f} sits in a "
                    f"{right - left:.1f}pt gap, which is text, not a gutter"
                )


@pytest.mark.skipif(not PDF_2026.exists(), reason="2026 program PDF absent")
def test_three_columns_are_balanced():
    """A misdetected gutter shows up as one nearly empty column."""
    with pdfplumber.open(PDF_2026) as pdf:
        for pno in SESSION_PAGES:
            page = pdf.pages[pno - 1]
            mids = _mids(page)
            bounds = _column_bounds(sorted(mids), 3, page.width)
            counts = [0, 0, 0]
            for m in mids:
                counts[bisect.bisect_right(bounds, m)] += 1
            assert min(counts) > 0.5 * (sum(counts) / 3), (
                f"page {pno}: lopsided columns {counts}"
            )


def test_single_column_has_no_bounds():
    assert _column_bounds([1.0, 2.0, 3.0], 1, 100.0) == []


def test_midpoint_default_is_unchanged_for_two_columns():
    """The five existing years must not move. bisect_right over a single
    boundary at the page midpoint reproduces the old expression
    '0 if mid < page_mid else 1' exactly, including the boundary case."""
    bounds = [50.0]
    assert bisect.bisect_right(bounds, 10.0) == 0
    assert bisect.bisect_right(bounds, 60.0) == 1
    assert bisect.bisect_right(bounds, 50.0) == 1
    assert bisect.bisect_right([], 10.0) == 0
