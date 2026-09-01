"""Parser behaviour that the committed data cannot vouch for.

These run against hand-written HTML fragments, not the network and not
`data/raw/`, because both of those are downstream of the bug being guarded
against: a parser that mishandles a field writes a plausible-looking value, and
the stored result then agrees with the parser forever. The only way to catch it
is to state the expected behaviour independently, which is what a fixture does.

Both era modules are exercised together wherever they implement the same
contract, since the 2024/2025 module has twice drifted from its 2022/2023
sibling on exactly that kind of shared behaviour.
"""

import sys
from pathlib import Path

import pytest

# beautifulsoup4 belongs to the scraping half of the pipeline, so an install of
# requirements-ci.txt alone (what the deploy workflow uses) does not have it.
# Skip the module rather than fail collection, matching how test_orals_data.py
# handles a missing pdfplumber: someone who only wants to rebuild the website
# should still be able to run the test suite and see it pass.
BeautifulSoup = pytest.importorskip("bs4").BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scraper"))

from parsers import era_2022_2023, era_2024_2025  # noqa: E402

# The two eras use different section ids for author feedback: 2022 and
# 2024/2025 use "authorFeedback-id", 2023 uses "author-feedback".
ERAS = [
    pytest.param(era_2024_2025, "authorFeedback-id", id="2024_2025"),
    pytest.param(era_2022_2023, "authorFeedback-id", id="2022_2023"),
    pytest.param(era_2022_2023, "author-feedback", id="2023"),
]


def _feedback_page(section_id: str, blockquote_html: str) -> BeautifulSoup:
    return BeautifulSoup(
        f'<h1 id="{section_id}">Author Feedback</h1>{blockquote_html}',
        "html.parser",
    )


def _blank_result() -> dict:
    return {"author_feedback_text": None, "rebuttal_provided": False}


@pytest.mark.parametrize("module, section_id", ERAS)
@pytest.mark.parametrize("na_text", ["N/A", "n/a", " N/A "])
def test_na_feedback_is_not_a_rebuttal(module, section_id, na_text):
    """MICCAI writes the literal "N/A" when the author did not rebut.

    Every accepted paper has an Author Feedback section, so the presence of a
    blockquote says nothing; only its contents do. The 2024/2025 parser was
    missing this guard, which marked 134 papers (71 in 2024, 63 in 2025) as
    having provided a rebuttal when they had declined to.
    """
    soup = _feedback_page(section_id, f"<blockquote><p>{na_text}</p></blockquote>")
    result = _blank_result()
    module._parse_author_feedback(soup, result)
    assert result["rebuttal_provided"] is False
    assert result["author_feedback_text"] is None


@pytest.mark.parametrize("module, section_id", ERAS)
def test_real_feedback_is_a_rebuttal(module, section_id):
    soup = _feedback_page(
        section_id,
        "<blockquote><p>We thank all reviewers for their comments.</p>"
        "<p>R1 asked about the ablation.</p></blockquote>",
    )
    result = _blank_result()
    module._parse_author_feedback(soup, result)
    assert result["rebuttal_provided"] is True
    assert "We thank all reviewers" in result["author_feedback_text"]
    assert "R1 asked about the ablation." in result["author_feedback_text"]


@pytest.mark.parametrize("module, section_id", ERAS)
def test_missing_section_leaves_defaults(module, section_id):
    soup = BeautifulSoup("<h1 id='abstract-id'>Abstract</h1>", "html.parser")
    result = _blank_result()
    module._parse_author_feedback(soup, result)
    assert result["rebuttal_provided"] is False
    assert result["author_feedback_text"] is None


@pytest.mark.parametrize("module, section_id", ERAS)
def test_empty_blockquote_is_not_a_rebuttal(module, section_id):
    soup = _feedback_page(section_id, "<blockquote></blockquote>")
    result = _blank_result()
    module._parse_author_feedback(soup, result)
    assert result["rebuttal_provided"] is False
    assert result["author_feedback_text"] is None


def test_both_eras_agree_on_every_case():
    """The two implementations must not drift apart again.

    They have differed twice: once on this guard, and once on how review
    headings are found. Comparing them directly is cheaper than remembering.
    """
    cases = [
        "<blockquote><p>N/A</p></blockquote>",
        "<blockquote><p>n/a</p></blockquote>",
        "<blockquote></blockquote>",
        "<blockquote><p>Real rebuttal text.</p></blockquote>",
    ]
    for html in cases:
        a, b = _blank_result(), _blank_result()
        era_2024_2025._parse_author_feedback(
            _feedback_page("authorFeedback-id", html), a
        )
        era_2022_2023._parse_author_feedback(
            _feedback_page("authorFeedback-id", html), b
        )
        assert a == b, f"eras disagree on {html!r}: {a} vs {b}"


# ---------------------------------------------------------------------------
# Reviewer numbering
# ---------------------------------------------------------------------------

# One review's worth of markup. The label has to match the score field for the
# era under test, which config.yaml gives as "scale of 1-6" for 2024/2025.
_REVIEW_LI = (
    "<li><strong>Please rate the paper on a scale of 1-6</strong>"
    "<blockquote><p>({score}) Weak Accept</p></blockquote></li>"
)


def _reviews_page(numbers, scores=None):
    """A paper page carrying review headings with the given numbers."""
    scores = scores or [4] * len(numbers)
    body = ['<h1 id="review-id">Reviews</h1>']
    for n, sc in zip(numbers, scores):
        body.append(f'<h3 id="review-{n}">Review #{n}</h3>')
        body.append("<ul>" + _REVIEW_LI.format(score=sc) + "</ul>")
    return BeautifulSoup("".join(body), "html.parser")


def _parse_2024_reviews(soup):
    result = {"early_accepted": False, "reviews": [], "num_reviews": 0}
    era_2024_2025._parse_reviews(
        soup,
        result,
        cfg={},
        score_field_sub="scale of 1-6",
        scale_max=6,
        score_denom=5,
        post_rebuttal_has_score=True,
    )
    return result


def test_reviews_numbered_from_one_are_all_parsed():
    result = _parse_2024_reviews(_reviews_page([1, 2, 3]))
    assert result["num_reviews"] == 3
    assert [r["reviewer_num"] for r in result["reviews"]] == [1, 2, 3]
    assert result["early_accepted"] is False


def test_reviews_not_starting_at_one_are_kept():
    """A paper can carry review-3, review-4, review-5 and no review-1.

    Reviewer reassignment leaves the remaining reviewers on their original
    numbers. 82 papers in 2022 and 108 in 2023 look like this. The 2024/2025
    parser used to read the missing review-1 as proof of early acceptance,
    return immediately, and store nothing.
    """
    result = _parse_2024_reviews(_reviews_page([3, 4, 5]))
    assert result["num_reviews"] == 3
    assert [r["reviewer_num"] for r in result["reviews"]] == [3, 4, 5]
    assert result["early_accepted"] is False, (
        "an unusual heading number is not evidence of early acceptance"
    )


def test_gap_in_numbering_does_not_truncate():
    """Counting upward from 1 stopped at the first gap and dropped the rest."""
    result = _parse_2024_reviews(_reviews_page([1, 2, 4]))
    assert result["num_reviews"] == 3
    assert [r["reviewer_num"] for r in result["reviews"]] == [1, 2, 4]


def test_headings_are_sorted_even_if_the_page_is_not():
    result = _parse_2024_reviews(_reviews_page([10, 2, 1]))
    assert [r["reviewer_num"] for r in result["reviews"]] == [1, 2, 10]


def test_no_reviews_at_all_is_not_early_acceptance():
    """Early acceptance is decided from the meta-review section, not here."""
    soup = BeautifulSoup('<h1 id="review-id">Reviews</h1>', "html.parser")
    result = _parse_2024_reviews(soup)
    assert result["num_reviews"] == 0
    assert result["early_accepted"] is False


def test_mean_score_uses_every_review_found():
    result = _parse_2024_reviews(_reviews_page([3, 4, 5], scores=[3, 4, 5]))
    assert result["avg_score_raw"] == 4.0
    assert result["num_reviews"] == 3
