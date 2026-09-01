"""Parser for MICCAI 2024 and 2025 paper pages."""

import logging
import re

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Categories page
# ---------------------------------------------------------------------------


def parse_categories_page(html: str, cfg: dict) -> dict[str, list[str]]:
    """
    Parse the categories page and return {filename: [subject_areas]}.

    The categories page is the only source of subject areas. Each <h3 id="...">
    element's id IS the subject area name. Papers follow as a series of
    <div class="posts-list-item"> siblings (not <ul><li> as docs suggest).
    A paper may appear under multiple areas.
    """
    soup = BeautifulSoup(html, "lxml")
    area_map: dict[str, list[str]] = {}

    for h3 in soup.find_all("h3"):
        area = h3.get("id", "").strip()
        if not area:
            continue
        # Walk siblings until the next <h3> or end, collecting paper divs
        node = h3.next_sibling
        while node:
            if getattr(node, "name", None) == "h3":
                break
            if getattr(node, "name", None) == "div":
                a = node.find("a", href=True)
                if a:
                    filename = a["href"].rstrip("/").split("/")[-1]
                    if filename.endswith(".html"):
                        area_map.setdefault(filename, []).append(area)
            node = node.next_sibling

    return area_map


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------


def parse_index_page(html: str, cfg: dict) -> list[tuple[str, str]]:
    """
    Parse the index page and return an ordered list of (paper_url, paper_id).

    Matches hrefs that look like NNNN-PaperPPPP.html (3-4-digit prefix).
    """
    soup = BeautifulSoup(html, "lxml")
    year = cfg["_year"]
    base = cfg["paper_url_base"].rstrip("/")

    papers: list[tuple[str, str]] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.search(r"(\d{3,4})-(Paper\d{4})\.html", href)
        if not m:
            continue
        filename = f"{m.group(1)}-{m.group(2)}.html"
        if filename in seen:
            continue
        seen.add(filename)
        paper_num = m.group(2).replace("Paper", "")
        paper_url = f"{base}/{filename}"
        paper_id = f"miccai-{year}-Paper{paper_num}"
        papers.append((paper_url, paper_id))

    return papers


# ---------------------------------------------------------------------------
# Single paper page
# ---------------------------------------------------------------------------


def parse_paper_page(
    html: str,
    paper_url: str,
    paper_id: str,
    cfg: dict,
    subject_areas: list[str],
) -> dict:
    """Parse one paper page and return the full paper dict."""
    soup = BeautifulSoup(html, "lxml")
    year: int = cfg["_year"]
    scale_max: int = cfg["review_scale_max"]
    score_denom: int = cfg["score_normalization_denominator"]
    post_rebuttal_has_score: bool = cfg.get("post_rebuttal_has_score", False)
    score_field_sub: str = cfg["score_field_substring"].lower()

    result: dict = {
        "paper_id": paper_id,
        "year": year,
        "title": None,
        "authors": [],
        "abstract": None,
        "subject_areas": subject_areas,
        "url": paper_url,
        "pdf_url": None,
        "doi": None,
        "shared_it_url": None,
        "supp_url": None,
        "code_url": None,
        "has_code": False,
        "dataset_urls": [],
        "lncs_volume": None,
        "bibtex_key": None,
        "bibtex_raw": None,
        "early_accepted": False,
        "rebuttal_provided": False,
        "reviews": [],
        "num_reviews": 0,
        "avg_score_raw": None,
        "avg_score_normalized": None,
        "score_range": None,
        "meta_reviews": [],
        "author_feedback_text": None,
    }

    _parse_title(soup, result)
    _parse_authors(soup, result)
    _parse_abstract(soup, result)
    _parse_links(soup, result)
    _parse_code(soup, result)
    _parse_datasets(soup, result)
    _parse_bibtex(soup, result)
    _parse_reviews(
        soup,
        result,
        cfg,
        score_field_sub,
        scale_max,
        score_denom,
        post_rebuttal_has_score,
    )
    _parse_author_feedback(soup, result)
    _parse_meta_reviews(soup, result)

    # 2025 early-accepted: single meta-review with "Provisional Accept"
    if not result["early_accepted"] and len(result["meta_reviews"]) == 1:
        rec = (result["meta_reviews"][0].get("recommendation") or "").lower()
        if "provisional accept" in rec:
            result["early_accepted"] = True

    return result


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------


def _parse_title(soup: BeautifulSoup, result: dict) -> None:
    title_div = soup.find("div", class_="post-title")
    if not title_div:
        return
    b = title_div.find("b")
    if b:
        result["title"] = b.get_text(strip=True)
    else:
        h1 = title_div.find("h1")
        if h1:
            result["title"] = h1.get_text(strip=True)


def _parse_authors(soup: BeautifulSoup, result: dict) -> None:
    tags_div = soup.find("div", class_="post-tags")
    if not tags_div:
        return
    for a in tags_div.find_all("a", class_="post-category"):
        if "/categories#" in a.get("href", ""):
            continue
        name = a.get_text(strip=True)
        if name:
            result["authors"].append(name)


def _parse_abstract(soup: BeautifulSoup, result: dict) -> None:
    h1 = soup.find("h1", id="abstract-id")
    if not h1:
        return
    p = h1.find_next_sibling("p")
    if p:
        text = p.get_text(strip=True)
        text = re.sub(r"\\url\{[^}]*\}", "", text)
        text = re.sub(r"\$[^$]+\$", "", text)
        result["abstract"] = text.strip() or None


def _parse_links(soup: BeautifulSoup, result: dict) -> None:
    h1 = soup.find("h1", id="link-id")
    if not h1:
        return
    node = h1.find_next_sibling()
    while node and node.name not in ("h1", "hr"):
        if node.name == "p":
            text = node.get_text(strip=True)
            text_lower = text.lower()
            a = node.find("a", href=True)
            href = a["href"] if a else None
            if href:
                if "main paper" in text_lower:
                    result["pdf_url"] = href
                elif "sharedit" in text_lower or "rdcu.be" in href:
                    result["shared_it_url"] = href
                elif "springerlink" in text_lower or "doi" in text_lower:
                    result["doi"] = href
                elif "supplementary" in text_lower:
                    if "not submitted" not in text_lower:
                        result["supp_url"] = href
        node = node.find_next_sibling()


def _parse_code(soup: BeautifulSoup, result: dict) -> None:
    h1 = soup.find("h1", id="code-id")
    if not h1:
        return
    p = h1.find_next_sibling("p")
    if p:
        a = p.find("a", href=True)
        if a:
            url = a["href"].strip()
            if url and url.upper() not in ("N/A", ""):
                result["code_url"] = url
                result["has_code"] = True


def _parse_datasets(soup: BeautifulSoup, result: dict) -> None:
    h1 = soup.find("h1", id="dataset-id")
    if not h1:
        return
    p = h1.find_next_sibling("p")
    if p:
        for a in p.find_all("a", href=True):
            result["dataset_urls"].append(a["href"])


def _parse_bibtex(soup: BeautifulSoup, result: dict) -> None:
    h1 = soup.find("h1", id="bibtex-id")
    if not h1:
        return
    pre = h1.find_next_sibling("pre")
    if not pre:
        return
    code = pre.find("code")
    raw = (code or pre).get_text()
    result["bibtex_raw"] = raw.strip()

    m = re.search(r"@\w+\{([^,\n]+)", raw)
    if m:
        result["bibtex_key"] = m.group(1).strip()

    m = re.search(r"volume\s*=\s*\{([^}]+)\}", raw, re.IGNORECASE)
    if m:
        result["lncs_volume"] = m.group(1).strip()


def _parse_reviews(
    soup: BeautifulSoup,
    result: dict,
    cfg: dict,
    score_field_sub: str,
    scale_max: int,
    score_denom: int,
    post_rebuttal_has_score: bool,
) -> None:
    review_h1 = soup.find("h1", id="review-id")
    # Reviewer numbering does not always start at 1 and can have gaps: when a
    # reviewer is reassigned the others keep their original numbers, so a paper
    # can carry review-3, review-4, review-5 and no review-1 at all. This bit
    # 2022 and 2023 hard (82 and 108 papers with no review-1, 100 and 137 more
    # with a gap after it), which is why era_2022_2023 was rewritten to collect
    # every heading and sort them. This module counted upward from 1 until
    # 2026-08-30 and so had both faults: it stopped at the first gap, and it
    # read "no review-1" as proof of early acceptance, which then stored zero
    # reviews and set the wrong flag without any error. 2024 and 2025 happen to
    # be regular, so nothing was lost; a future year need not be.
    review_h3s = soup.find_all("h3", id=re.compile(r"^review-\d+$"))

    if not review_h1 or not review_h3s:
        return

    review_h3s.sort(key=lambda h3: int(h3["id"].split("-")[1]))

    reviews = []
    for h3 in review_h3s:
        rev = _parse_single_review(
            h3,
            int(h3["id"].split("-")[1]),
            score_field_sub,
            scale_max,
            score_denom,
            post_rebuttal_has_score,
        )
        reviews.append(rev)

    result["reviews"] = reviews
    result["num_reviews"] = len(reviews)

    raw_scores = [
        r["score_raw"] for r in reviews if r["score_raw"] is not None
    ]
    if raw_scores:
        result["avg_score_raw"] = round(sum(raw_scores) / len(raw_scores), 4)
        norm_scores = [
            r["score_normalized"]
            for r in reviews
            if r["score_normalized"] is not None
        ]
        result["avg_score_normalized"] = round(
            sum(norm_scores) / len(norm_scores), 4
        )
        result["score_range"] = max(raw_scores) - min(raw_scores)


def _parse_author_feedback(soup: BeautifulSoup, result: dict) -> None:
    h1 = soup.find("h1", id="authorFeedback-id")
    if not h1:
        return
    bq = h1.find_next_sibling("blockquote")
    if bq:
        text = bq.get_text(separator="\n", strip=True)
        # The section exists for every paper, so an author who chose not to
        # rebut still gets a blockquote; MICCAI fills it with the literal
        # string "N/A". Treating that as a rebuttal is what this guard exists
        # to stop, and it must match era_2022_2023._parse_author_feedback
        # exactly, including the .upper(), or the two eras disagree about what
        # rebuttal_provided means.
        if text and text.strip().upper() != "N/A":
            result["author_feedback_text"] = text
            result["rebuttal_provided"] = True


def _parse_meta_reviews(soup: BeautifulSoup, result: dict) -> None:
    h1 = soup.find("h1", id="metareview-id")
    if not h1:
        return
    # 2024 early-accepted: h1 is followed by a <p> saying "early accepted paper"
    p = h1.find_next_sibling("p")
    if p and "early accepted" in p.get_text().lower():
        result["early_accepted"] = True
        return
    meta_num = 1
    while True:
        h2 = soup.find("h2", id=f"meta-review-{meta_num}")
        if not h2:
            break
        meta: dict = {
            "meta_reviewer_num": meta_num,
            "recommendation": None,
            "text": None,
        }
        # Same sibling walk as the reviews: a meta-review's fields can be split
        # across more than one <ul>, and taking only the first loses the rest.
        meta_items = _field_items(h2)
        if meta_items:
            pre_rec: str | None = None
            post_rec: str | None = None
            text_parts: list[str] = []
            for li in meta_items:
                strong = li.find("strong")
                bq = li.find("blockquote")
                if not strong or not bq:
                    continue
                label = strong.get_text(strip=True).lower()
                val = bq.get_text(separator="\n", strip=True)
                if "after you have reviewed the rebuttal" in label:
                    post_rec = val
                elif label.startswith("your recommendation"):
                    pre_rec = val
                elif (
                    "please justify your recommendation" in label
                    or "justify your" in label
                ):
                    if val and val.upper() not in ("N/A", ""):
                        text_parts.append(val)
                elif "recommendation" not in label:
                    # Any other non-recommendation field (e.g., general text)
                    if val and val.upper() not in ("N/A", ""):
                        text_parts.append(val)
            meta["recommendation"] = post_rec or pre_rec
            meta["text"] = "\n\n".join(text_parts) if text_parts else None
        result["meta_reviews"].append(meta)
        meta_num += 1


# ---------------------------------------------------------------------------
# Single review block parser
# ---------------------------------------------------------------------------


def _parse_single_review(
    rev_h3,
    rev_num: int,
    score_field_sub: str,
    scale_max: int,
    score_denom: int,
    post_rebuttal_has_score: bool,
) -> dict:
    rev: dict = {
        "reviewer_num": rev_num,
        "score_raw": None,
        "score_max": scale_max,
        "score_normalized": None,
        "recommendation_label": None,
        "post_rebuttal_label": None,
        "post_rebuttal_score_raw": None,
        "score_changed": False,
        "confidence_raw": None,
        "confidence_label": None,
        "clarity_label": None,
        "text_contribution": None,
        "text_strengths": None,
        "text_weaknesses": None,
        "text_detailed_comments": None,
        "text_reproducibility": None,
        "text_post_rebuttal_justification": None,
        "review_stack_rank": None,
        "review_stack_size": None,
    }

    field_items = _field_items(rev_h3)
    if not field_items:
        return rev

    for li in field_items:
        strong = li.find("strong")
        bq = li.find("blockquote")
        if not strong or not bq:
            continue

        label = strong.get_text(strip=True).lower()
        val = bq.get_text(separator="\n", strip=True)
        val_inline = bq.get_text(separator=" ", strip=True)

        if "describe the contribution" in label:
            rev["text_contribution"] = val

        elif "major strengths" in label or "main strengths" in label:
            rev["text_strengths"] = val

        elif "major weaknesses" in label or "main weaknesses" in label:
            rev["text_weaknesses"] = val

        elif "rate the clarity" in label:
            rev["clarity_label"] = val_inline

        # 2024 asks about reproducibility twice: "Please comment on the
        # reproducibility of the paper..." and then "Do you have any additional
        # comments regarding the paper's reproducibility?". Both labels contain
        # "reproducibility", and this branch is tested before "additional
        # comments", so a plain assignment let the follow-up answer replace the
        # real one; 1,352 of 2,623 reviews in 2024 ended up storing just "N/A".
        # Append instead, so both are kept in the order the page shows them.
        # No other year asks twice, so this changes nothing outside 2024.
        elif "reproducibility" in label:
            _append_text(rev, "text_reproducibility", val)

        elif "additional comments" in label:
            _append_text(rev, "text_detailed_comments", val)

        # The main free-text box for the authors. Its label matched no branch
        # at all, so the whole answer was dropped for every review in 2021
        # through 2024 (8,254 reviews). 2025 replaced this question with the
        # "additional comments" one above, which is why 2025 was unaffected.
        # text_detailed_comments feeds Avg Review Length on the overview page,
        # so losing it made every year before 2025 look shorter than it was.
        elif "detailed and constructive comments" in label:
            _append_text(rev, "text_detailed_comments", val)

        elif score_field_sub in label:
            score = _extract_parens_int(val_inline)
            rev["score_raw"] = score
            if score is not None:
                rev["score_normalized"] = round((score - 1) / score_denom, 4)
            label_text = re.sub(r"\(\d+\)", "", val_inline).strip(" ,;:-")
            rev["recommendation_label"] = label_text or val_inline

        elif "[post rebuttal]" in label and (
            "final opinion" in label or "overall opinion" in label
        ):
            rev["post_rebuttal_label"] = val_inline
            if post_rebuttal_has_score:
                rev["post_rebuttal_score_raw"] = _extract_parens_int(
                    val_inline
                )

        # The wording changed: 2025 asks to "justify your final decision from
        # above", 2022 through 2024 just "justify your decision". Matching only
        # "justify your final" dropped the answer for all of 2022 to 2024, so
        # 6,626 reviews carried a post-rebuttal verdict with no reasoning, which
        # read as reviewers declining to explain rather than as a parser miss.
        elif "[post rebuttal]" in label and (
            "justify your final" in label or "justify your decision" in label
        ):
            rev["text_post_rebuttal_justification"] = val

        elif "justify your recommendation" in label:
            _append_text(rev, "text_detailed_comments", val)

        elif "reviewer confidence" in label:
            rev["confidence_raw"] = _extract_parens_int(val_inline)
            conf_label = re.sub(r"\(\d+\)", "", val_inline).strip(" ,;:-")
            rev["confidence_label"] = conf_label or val_inline

    post = rev["post_rebuttal_label"]
    if (
        rev["recommendation_label"]
        and post
        and post.strip().upper() not in ("N/A", "")
    ):
        rev["score_changed"] = _verdict(
            rev["recommendation_label"]
        ) != _verdict(post)

    return rev


def _field_items(heading) -> list:
    """Every <li> field belonging to one review or meta-review heading.

    MICCAI usually puts a section's fields in a single <ul>, but not always.
    On miccai-2024-Paper0405 the third review's twelve fields are split across
    two lists, seven in the first and five in the second, with a run of loose
    <blockquote> siblings between them where a long answer broke the markup.
    Taking only find_next_sibling("ul") silently dropped everything in the
    second list, which cost that reviewer their score, recommendation,
    confidence, and both post-rebuttal fields, with no error and no warning.

    Walk the siblings instead, collecting every <ul> and stopping at the next
    heading or horizontal rule so the following section is never absorbed.
    """
    items = []
    for sib in heading.find_next_siblings():
        if sib.name in ("h1", "h2", "h3", "hr"):
            break
        if sib.name == "ul":
            items.extend(sib.find_all("li", recursive=False))
    return items


def _extract_parens_int(text: str) -> int | None:
    """Extract integer from '(N) Label' or 'Label (N)' format."""
    m = re.search(r"\((\d+)\)", text)
    return int(m.group(1)) if m else None


def _verdict(label: str) -> str:
    """
    Normalize a recommendation label to just the verdict keyword for comparison.
    'Reject - should be rejected, independent of rebuttal' → 'reject'
    'Weak Accept; could be accepted, dependent on rebuttal' → 'weak accept'
    'Accept' → 'accept'
    """
    # Split on em dash, en dash, or hyphen with surrounding spaces.
    # Written as escapes on purpose: these match MICCAI's text, not ours,
    # so a search-and-replace over prose dashes must not touch them.
    return re.split(
        r"\s*[\u2014\u2013-]\s*", label, maxsplit=1
    )[0].strip().lower()


def _append_text(rev: dict, key: str, text: str) -> None:
    if rev[key]:
        rev[key] += "\n\n" + text
    else:
        rev[key] = text
