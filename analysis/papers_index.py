#!/usr/bin/env python3
"""Build website/papers.json, the data behind the All Papers browse page.

This module reads nothing from disk. build_site.py hands it the paper list it
has already loaded and already patched with presentation types, which does two
things: the 59 MB miccai_all.json is parsed once per build instead of twice,
and the types here cannot disagree with the Orals page or the Top Papers
tables, because they are the same objects.

Standard library only. The deploy workflow installs a short dependency list and
this module must not extend it.
"""

import json
from pathlib import Path

TYPE_CODES = {"poster": 0, "oral": 1, "spotlight": 2}

# (key in the output, key in miccai_all.json). Coverage varies a lot by year:
# every paper has a DOI, roughly two thirds have code, and pdf_url, supp_url
# and lncs_volume exist only for 2024 and 2025. Absent values are left out of
# the record entirely rather than written as null, which is worth about 50 KB
# and lets the page test `if (p.code_url)` with no null handling.
LINK_FIELDS = (
    ("code_url", "code_url"),
    ("pdf_url", "pdf_url"),
    ("doi", "doi"),
    ("sharedit_url", "shared_it_url"),
    ("supp_url", "supp_url"),
    ("lncs", "lncs_volume"),
)


def _clean(value):
    """Return value, or None if it is empty or MICCAI's literal "N/A"."""
    if not value or value == "N/A":
        return None
    return value


def scales_from(papers):
    """Map each year to its review scale maximum, read from the reviews.

    The scale changed twice (1-9 in 2021, 1-8 in 2022 and 2023, 1-6 from 2024),
    and the page needs it to print "5.67/6" and to explain its sort. Reading it
    from score_max rather than hardcoding it means a further change in 2026
    needs no code edit.
    """
    scales = {}
    for paper in papers:
        year = paper["year"]
        if year in scales:
            continue
        for review in paper.get("reviews", []):
            if review.get("score_max"):
                scales[year] = review["score_max"]
                break
    return scales


def build(papers, out_path, built):
    """Write the browse index to out_path and return it.

    papers: the list build_site.py holds, after oral_stats.attach_types().
    built:  ISO date string, shown nowhere but useful when debugging a stale file.
    """
    areas = sorted({a for p in papers for a in p.get("subject_areas", [])})
    area_id = {name: i for i, name in enumerate(areas)}

    records = []
    for paper in papers:
        record = {
            "id": paper["paper_id"],
            "year": paper["year"],
            "title": paper["title"],
            "authors": paper.get("authors", []),
            "areas": [area_id[a] for a in paper.get("subject_areas", [])],
            "url": paper.get("url"),
            "score": paper.get("avg_score_raw"),
            "score_norm": paper.get("avg_score_normalized"),
            # Drop any null score rather than carry it to the page, where it
            # would render as "null". This should never fire: the one case that
            # existed came from a parser bug fixed on 2026-08-22, not from
            # MICCAI publishing a review without a score.
            "reviews": [
                r["score_raw"]
                for r in paper.get("reviews", [])
                if r.get("score_raw") is not None
            ],
            "early": 1 if paper.get("early_accepted") else 0,
            "type": TYPE_CODES.get(paper.get("presentation_type") or "poster", 0),
        }

        session = _clean(paper.get("oral_session"))
        if session:
            record["session"] = session

        for out_key, src_key in LINK_FIELDS:
            value = _clean(paper.get(src_key))
            if value:
                record[out_key] = value

        datasets = [d for d in (paper.get("dataset_urls") or []) if _clean(d)]
        if datasets:
            record["dataset_urls"] = datasets

        records.append(record)

    index = {
        "built": built,
        "areas": areas,
        "scales": {str(y): m for y, m in sorted(scales_from(papers).items())},
        "papers": records,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(index, separators=(",", ":")), encoding="utf-8"
    )
    return index
