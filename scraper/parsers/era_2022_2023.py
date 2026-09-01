"""Parser for MICCAI 2022 and 2023 paper pages.

Key differences from era_2024_2025:
  - Section headings are <h2> not <h1>; id values differ between 2022 and 2023
  - Score format: bare integer ("6"), no label text
  - Score scale: 1-8 (denominator 7); confidence label has no numeric suffix
  - No BibTeX section; links section has no explicit PDF (DOI/SharedIt only)
  - 2022: stack-rank/size fields in reviews; post-rebuttal score is bare integer
  - 2023: no stack fields; post-rebuttal is text-label only (or absent)
  - Meta-review recommendation present in 2022, absent in 2023 (embedded in text)
"""

import logging
import re

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Categories page
# ---------------------------------------------------------------------------


def parse_categories_page(html: str, cfg: dict) -> dict[str, list[str]]:
    """Parse categories page: h3[id] + div.posts-list-item siblings."""
    soup = BeautifulSoup(html, "lxml")
    area_map: dict[str, list[str]] = {}
    for h3 in soup.find_all("h3"):
        area = h3.get("id", "").strip()
        if not area:
            continue
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
    """Parse index page: collect NNN-PaperNNNN.html hrefs."""
    soup = BeautifulSoup(html, "lxml")
    year = cfg["_year"]
    base = cfg["paper_url_base"].rstrip("/")
    papers: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"(\d{3,4})-(Paper\d{4})\.html", a["href"])
        if not m:
            continue
        filename = f"{m.group(1)}-{m.group(2)}.html"
        if filename in seen:
            continue
        seen.add(filename)
        paper_num = m.group(2).replace("Paper", "")
        papers.append(
            (f"{base}/{filename}", f"miccai-{year}-Paper{paper_num}")
        )
    return papers


# ---------------------------------------------------------------------------
# Paper page
# ---------------------------------------------------------------------------


def parse_paper_page(
    html: str,
    paper_url: str,
    paper_id: str,
    cfg: dict,
    subject_areas: list[str],
) -> dict:
    """Parse one 2022/2023 paper page."""
    soup = BeautifulSoup(html, "lxml")
    year: int = cfg["_year"]
    scale_max: int = cfg["review_scale_max"]
    score_denom: int = cfg["score_normalization_denominator"]
    post_rebuttal_has_score: bool = cfg.get("post_rebuttal_has_score", False)
    score_field_sub: str = cfg["score_field_substring"].lower()
    score_parse_mode: str = cfg.get("score_parse_mode", "bare_integer")
    extra_fields: list[dict] = cfg.get("extra_review_fields", [])

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
    _parse_reviews(
        soup,
        result,
        score_field_sub,
        scale_max,
        score_denom,
        post_rebuttal_has_score,
        score_parse_mode,
        extra_fields,
    )
    _parse_author_feedback(soup, result)
    _parse_meta_reviews(soup, result)
    _determine_early_accepted(soup, result)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find(soup: BeautifulSoup, *id_candidates) -> object | None:
    """Return first element matching any of the given id values."""
    for hid in id_candidates:
        el = soup.find(id=hid)
        if el:
            return el
    return None


def _field_items(heading) -> list:
    """Every <li> field belonging to one review or meta-review heading.

    MICCAI usually puts a section's fields in a single <ul>, but not always: a
    long answer can break the markup, leaving the rest of the fields in a second
    list. Taking only find_next_sibling("ul") drops those silently, costing that
    reviewer their score, confidence, and post-rebuttal fields with no error.
    A confirmed case exists in 2024 (miccai-2024-Paper0405, review 3); none has
    been seen in 2021 to 2023, so this is defensive here rather than a fix, and
    it keeps both era parsers behaving the same way.

    Walk the siblings, collect every <ul>, and stop at the next heading or rule
    so the following section is never absorbed.
    """
    items = []
    for sib in heading.find_next_siblings():
        if sib.name in ("h1", "h2", "h3", "hr"):
            break
        if sib.name == "ul":
            items.extend(sib.find_all("li", recursive=False))
    return items


def _extract_bare_int(text: str) -> int | None:
    """Extract integer from a string that contains ONLY digits (plus whitespace)."""
    m = re.search(r"^\s*(\d+)\s*$", text.strip())
    return int(m.group(1)) if m else None


def _extract_parens_int(text: str) -> int | None:
    m = re.search(r"\((\d+)\)", text)
    return int(m.group(1)) if m else None


def _extract_score(text: str, mode: str) -> int | None:
    if mode == "bare_integer":
        return _extract_bare_int(text)
    return _extract_parens_int(text)


def _verdict(label: str) -> str:
    # Escapes on purpose: this matches MICCAI's own text, so a prose-level
    # dash sweep must not touch it.
    return re.split(
        r"\s*[\u2014\u2013-]\s*", label, maxsplit=1
    )[0].strip().lower()


def _append_text(rev: dict, key: str, text: str) -> None:
    if rev[key]:
        rev[key] += "\n\n" + text
    else:
        rev[key] = text


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------


def _parse_title(soup: BeautifulSoup, result: dict) -> None:
    # 2024/2025-style wrapper
    title_div = soup.find("div", class_="post-title")
    if title_div:
        b = title_div.find("b")
        if b:
            result["title"] = b.get_text(strip=True)
            return
        h = title_div.find(re.compile(r"^h[1-3]$"))
        if h:
            result["title"] = h.get_text(strip=True)
            return

    # First standalone h1 that isn't a section heading
    section_ids = {
        "abstract",
        "abstract-id",
        "link-to-paper",
        "link-id",
        "reviews",
        "review-id",
        "primary-meta-review",
        "metareview-id",
        "author-feedback",
        "authorFeedback-id",
    }
    for h1 in soup.find_all("h1"):
        if h1.get("id", "") not in section_ids:
            text = h1.get_text(strip=True)
            if text:
                result["title"] = text
                return

    # Fall back to <title> tag, stripping " - MICCAI YYYY" suffix
    title_tag = soup.find("title")
    if title_tag:
        text = re.sub(
            r"\s*[---]\s*MICCAI\s+\d{4}.*$", "", title_tag.get_text(strip=True)
        )
        if text:
            result["title"] = text


def _parse_authors(soup: BeautifulSoup, result: dict) -> None:
    # Same pattern as 2024/2025
    tags_div = soup.find("div", class_="post-tags")
    if tags_div:
        for a in tags_div.find_all("a", class_="post-category"):
            if "/categories#" in a.get("href", ""):
                continue
            name = a.get_text(strip=True)
            if name:
                result["authors"].append(name)
        if result["authors"]:
            return

    # Fallback: anchor tags directly after the "author-id" heading
    author_h = _find(soup, "author-id")
    if author_h:
        node = author_h.find_next_sibling()
        while node and getattr(node, "name", None) not in (
            "h1",
            "h2",
            "h3",
            "hr",
        ):
            for a in (
                node.find_all("a", href=True)
                if hasattr(node, "find_all")
                else []
            ):
                if "/tags#" in a.get("href", "") or "/papers/tags" in a.get(
                    "href", ""
                ):
                    name = a.get_text(strip=True)
                    if name:
                        result["authors"].append(name)
            node = node.find_next_sibling()


def _parse_abstract(soup: BeautifulSoup, result: dict) -> None:
    h = _find(soup, "abstract-id", "abstract")
    if not h:
        return
    p = h.find_next_sibling("p")
    if p:
        text = p.get_text(strip=True)
        text = re.sub(r"\\url\{[^}]*\}", "", text)
        text = re.sub(r"\$[^$]+\$", "", text)
        result["abstract"] = text.strip() or None


def _parse_links(soup: BeautifulSoup, result: dict) -> None:
    # 2023 confirmed: h1[id="link-id"] (same as 2024/2025); fallback to "link-to-paper" just in case
    h = _find(soup, "link-id", "link-to-paper")
    if not h:
        return
    node = h.find_next_sibling()
    while node and getattr(node, "name", None) not in ("h1", "h2", "h3", "hr"):
        if getattr(node, "name", None) == "p":
            text = node.get_text(strip=True)
            text_lower = text.lower()
            a = node.find("a", href=True)
            href = a["href"] if a else None
            if href:
                if href.lower().endswith(".pdf") or "main paper" in text_lower:
                    result["pdf_url"] = href
                elif "rdcu.be" in href or "sharedit" in text_lower:
                    result["shared_it_url"] = href
                elif (
                    "doi.org" in href
                    or "doi" in text_lower
                    or "springer" in text_lower
                ):
                    result["doi"] = href
                elif (
                    "supplementary" in text_lower
                    and "not submitted" not in text_lower
                ):
                    result["supp_url"] = href
        node = node.find_next_sibling()


def _parse_code(soup: BeautifulSoup, result: dict) -> None:
    h = _find(soup, "link-to-code-repository", "code-id")
    if not h:
        return
    p = h.find_next_sibling("p")
    if p:
        a = p.find("a", href=True)
        if a:
            url = a["href"].strip()
            if url and url.upper() not in ("N/A", ""):
                result["code_url"] = url
                result["has_code"] = True


def _parse_datasets(soup: BeautifulSoup, result: dict) -> None:
    h = _find(soup, "link-to-datasets", "dataset-id")
    if not h:
        return
    p = h.find_next_sibling("p")
    if p:
        for a in p.find_all("a", href=True):
            result["dataset_urls"].append(a["href"])


def _parse_reviews(
    soup: BeautifulSoup,
    result: dict,
    score_field_sub: str,
    scale_max: int,
    score_denom: int,
    post_rebuttal_has_score: bool,
    score_parse_mode: str,
    extra_fields: list[dict],
) -> None:
    # Reviews section uses id="reviews" (2023) or id="review-id" (2022)
    review_h = _find(soup, "reviews", "review-id")
    # Review numbering does not always start at 1 (e.g. reviewer reassignment can leave
    # gaps like review-3, review-4, review-5); find all review-N headings, not just a
    # sequential scan starting from 1, or content for renumbered papers is silently lost.
    review_h3s = soup.find_all("h3", id=re.compile(r"^review-\d+$"))

    if not review_h or not review_h3s:
        return

    review_h3s.sort(key=lambda h3: int(h3["id"].split("-")[1]))

    reviews = []
    for h3 in review_h3s:
        rev_num = int(h3["id"].split("-")[1])
        rev = _parse_single_review(
            h3,
            rev_num,
            score_field_sub,
            scale_max,
            score_denom,
            post_rebuttal_has_score,
            score_parse_mode,
            extra_fields,
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
    # 2022 uses same id as 2024/2025; 2023 uses "author-feedback"
    h = _find(soup, "authorFeedback-id", "author-feedback")
    if not h:
        return
    bq = h.find_next_sibling("blockquote")
    if bq:
        text = bq.get_text(separator="\n", strip=True)
        if text and text.strip().upper() != "N/A":
            result["author_feedback_text"] = text
            result["rebuttal_provided"] = True


def _parse_meta_reviews(soup: BeautifulSoup, result: dict) -> None:
    # Meta-reviews are <h2> elements (confirmed 2023; 2022 also uses h2).
    # In 2023 the primary meta-review id is "meta-review--1-primary" (double dash),
    # then post-rebuttal ones are "meta-review-2", "meta-review-3".
    # The regex ^meta-review- matches all of these.
    meta_h3s = soup.find_all(["h2", "h3"], id=re.compile(r"^meta-review-"))
    if not meta_h3s:
        # Some papers store the (single) meta-review directly in a <ul> right after
        # h1#metareview-id, with no <h2> reviewer-block wrapper at all. This <ul> is
        # usually FILLED with real content; confirmed by sampling live HTML, where
        # every such case had a populated assessment blockquote. Only treat it as an
        # early-accepted/unfilled form if every blockquote in that <ul> is empty.
        meta_section = _find(soup, "metareview-id")
        if not meta_section:
            return
        first_el = meta_section.find_next_sibling()
        while first_el and not getattr(first_el, "name", None):
            first_el = first_el.find_next_sibling()
        if not first_el or first_el.name != "ul":
            return

        text_parts: list[str] = []
        has_filled_content = False
        for li in first_el.find_all("li", recursive=False):
            strong = li.find("strong")
            bq = li.find("blockquote")
            if not strong or not bq:
                continue
            label = strong.get_text(strip=True).lower()
            val = bq.get_text(separator="\n", strip=True)
            if val and val.upper() not in ("N/A", ""):
                has_filled_content = True
                if "rank" not in label:
                    text_parts.append(val)

        if not has_filled_content:
            return

        result["meta_reviews"].append(
            {
                "meta_reviewer_num": 1,
                "recommendation": None,
                "text": "\n\n".join(text_parts) if text_parts else None,
            }
        )
        return

    for i, h3 in enumerate(meta_h3s, start=1):
        meta: dict = {
            "meta_reviewer_num": i,
            "recommendation": None,
            "text": None,
        }
        meta_items = _field_items(h3)
        if not meta_items:
            result["meta_reviews"].append(meta)
            continue

        text_parts: list[str] = []
        for li in meta_items:
            strong = li.find("strong")
            bq = li.find("blockquote")
            if not strong or not bq:
                continue
            label = strong.get_text(strip=True).lower()
            val = bq.get_text(separator="\n", strip=True)
            val_inline = bq.get_text(separator=" ", strip=True)

            if "after you have reviewed the rebuttal" in label:
                # Final recommendation field (present in 2022)
                meta["recommendation"] = val_inline
            elif (
                "provide your assessment" in label
                or "justify your recommendation" in label
            ):
                if val and val.upper() not in ("N/A", ""):
                    text_parts.append(val)
            elif "rank" not in label:
                # Include other free-text fields; skip stack-rank fields
                if val and val.upper() not in ("N/A", ""):
                    text_parts.append(val)

        meta["text"] = "\n\n".join(text_parts) if text_parts else None
        result["meta_reviews"].append(meta)


def _determine_early_accepted(soup: BeautifulSoup, result: dict) -> None:
    # Verified against official MICCAI stats: papers lacking a post-rebuttal meta-review
    # round (no <h1 id="postrebuttal-id">) were decided in a single primary meta-review,
    # with no further discussion needed; i.e. accepted before/without rebuttal.
    # Exact match against official early-accept counts: 2022 249/573, 2023 308/730.
    result["early_accepted"] = _find(soup, "postrebuttal-id") is None


# ---------------------------------------------------------------------------
# Single review parser
# ---------------------------------------------------------------------------


def _parse_single_review(
    rev_h3,
    rev_num: int,
    score_field_sub: str,
    scale_max: int,
    score_denom: int,
    post_rebuttal_has_score: bool,
    score_parse_mode: str,
    extra_fields: list[dict],
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

    extra_subs = [
        (f["label_substring"].lower(), f["store_as"]) for f in extra_fields
    ]

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

        elif "main strengths" in label or "major strengths" in label:
            rev["text_strengths"] = val

        elif "main weaknesses" in label or "major weaknesses" in label:
            rev["text_weaknesses"] = val

        elif "rate the clarity" in label:
            rev["clarity_label"] = val_inline

        # Append rather than assign, matching era_2024_2025.py: a year whose
        # form asks about reproducibility twice would otherwise keep only the
        # second answer. 2021 to 2023 ask once, so this changes nothing here;
        # it keeps the two parsers from drifting apart.
        elif "reproducibility" in label:
            _append_text(rev, "text_reproducibility", val)

        elif "additional comments" in label:
            _append_text(rev, "text_detailed_comments", val)

        # The main free-text box for the authors. Its label matched no branch,
        # so the answer was dropped for every review in 2021, 2022, and 2023,
        # as well as 2024. It feeds Avg Review Length on the overview page.
        elif "detailed and constructive comments" in label:
            _append_text(rev, "text_detailed_comments", val)

        elif score_field_sub in label:
            score = _extract_score(val_inline, score_parse_mode)
            rev["score_raw"] = score
            if score is not None:
                rev["score_normalized"] = round((score - 1) / score_denom, 4)
            # For bare-integer format there's no label text; store the raw string
            rev["recommendation_label"] = val_inline.strip()

        elif "[post rebuttal]" in label and (
            "final opinion" in label or "overall opinion" in label
        ):
            rev["post_rebuttal_label"] = val_inline
            if post_rebuttal_has_score:
                rev["post_rebuttal_score_raw"] = _extract_score(
                    val_inline, score_parse_mode
                )

        # 2022 and 2023 ask to "justify your decision"; 2025 asks to "justify
        # your final decision from above". Matching only "justify your final"
        # dropped the answer for every 2022 and 2023 review, so the data showed
        # a post-rebuttal verdict with no reasoning behind it.
        elif "[post rebuttal]" in label and (
            "justify your final" in label or "justify your decision" in label
        ):
            rev["text_post_rebuttal_justification"] = val

        elif "justify your recommendation" in label:
            _append_text(rev, "text_detailed_comments", val)

        elif "reviewer confidence" in label:
            # 2022/2023 confidence is text-only ("Very confident"); no numeric suffix
            rev["confidence_raw"] = _extract_parens_int(
                val_inline
            )  # → None for 2022/2023
            conf_label = re.sub(r"\(\d+\)", "", val_inline).strip(" ,;:-")
            rev["confidence_label"] = conf_label or val_inline

        else:
            # Check extra year-specific fields (stack rank, stack size)
            for sub, store_as in extra_subs:
                if sub in label:
                    rev[store_as] = _extract_bare_int(val_inline)
                    break

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
