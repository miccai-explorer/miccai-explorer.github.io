#!/usr/bin/env python3
"""Find review form questions that the parsers silently drop.

    python scraper/audit_review_fields.py --year 2025 [--sample 25]
    python scraper/audit_review_fields.py --all

Why this exists
---------------
MICCAI changes its review form between years, and the parsers match each
question by a substring of its label. A label that matches no branch is skipped
without an error, so the answer never reaches data/raw/miccai_YYYY.json and
leaves no trace there. The committed data therefore cannot tell you a field is
missing; only the live page can.

On 2026-08-22 this found four such faults at once, three of them present since
the first scrape:

  * "Please provide detailed and constructive comments for the authors"
    matched nothing, dropping the main free-text box for 8,254 reviews across
    2021 to 2024.
  * "[Post rebuttal] Please justify your decision" matched nothing, because the
    chain looked for "justify your final" (2025's wording), dropping 6,626
    post-rebuttal justifications across 2022 to 2024.
  * 2024 asks about reproducibility twice, and both labels contain the word,
    so the follow-up answer overwrote the real one in 1,352 reviews.
  * One 2024 page splits a review's fields across two <ul> blocks, and the
    parser read only the first.

Run this before trusting the field table in CLAUDE.md for a new year.

How it works
------------
It does not re-implement the matching chain; a second copy would drift from the
real one and eventually lie. Instead it feeds the real parser one form question
at a time and watches what comes back:

  * parse a review containing only question X. If every field is still at its
    default, nothing in the parser matched X, so X is DROPPED.
  * if two questions each set the same field, the later one wins and the
    earlier answer is lost: OVERWRITTEN.

It also reports reviews whose fields span more than one <ul> (SPLIT) and
blockquotes stranded outside any list (LOOSE), which is how a split shows up.
"""

import argparse
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scraper.parsers import (  # noqa: E402
    era_2021,
    era_2022_2023,
    era_2024_2025,
)

PARSERS = {
    "2024_2025": era_2024_2025,
    "2022_2023": era_2022_2023,
    "2021": era_2021,
}
# era_2021 delegates its paper parsing to era_2022_2023, so it shares that chain.
CHAIN = {"2021": era_2022_2023, "2022_2023": era_2022_2023, "2024_2025": era_2024_2025}

# Set by _parse_single_review on every review regardless of content, so their
# presence says nothing about whether a question matched.
ALWAYS_SET = {"reviewer_num", "score_max", "score_changed"}


def _single_review_fields(mod, li_html, cfg):
    """Parse a review built from one form question; return the fields it sets.

    Wrapping the question in its own <h3> plus <ul> exercises the real chain,
    including the sibling walk, without needing the rest of the page.
    """
    frag = BeautifulSoup(
        f'<h3 id="review-1">R</h3><ul>{li_html}</ul>', "lxml"
    )
    h3 = frag.find("h3")
    kwargs = dict(
        rev_num=1,
        score_field_sub=cfg["score_field_substring"].lower(),
        scale_max=cfg["review_scale_max"],
        score_denom=cfg["score_normalization_denominator"],
        post_rebuttal_has_score=cfg.get("post_rebuttal_has_score", False),
    )
    if mod is era_2022_2023:
        kwargs["score_parse_mode"] = cfg.get("score_parse_mode", "parens")
        kwargs["extra_fields"] = cfg.get("extra_review_fields", []) or []
    rev = mod._parse_single_review(h3, **kwargs)
    return {
        k for k, v in rev.items()
        if v not in (None, False, "") and k not in ALWAYS_SET
    }


def audit_page(html, mod, cfg):
    """Yield (kind, review_id, detail) for every fault found on one page."""
    soup = BeautifulSoup(html, "lxml")
    heads = sorted(
        soup.find_all("h3", id=re.compile(r"^review-\d+$")),
        key=lambda h: int(h["id"].split("-")[1]),
    )
    for h3 in heads:
        rid = h3["id"]
        uls, loose = [], 0
        for sib in h3.find_next_siblings():
            if sib.name in ("h1", "h2", "h3", "hr"):
                break
            if sib.name == "ul":
                uls.append(sib)
            elif sib.name == "blockquote":
                loose += 1
        if len(uls) > 1:
            yield "SPLIT", rid, f"fields span {len(uls)} <ul> blocks"
        if loose:
            yield "LOOSE", rid, f"{loose} blockquotes outside any <ul>"

        writers = defaultdict(list)
        for ul in uls:
            for li in ul.find_all("li", recursive=False):
                strong, bq = li.find("strong"), li.find("blockquote")
                if not strong or not bq:
                    continue
                label = strong.get_text(strip=True)
                fields = _single_review_fields(mod, str(li), cfg)
                if not fields:
                    yield "DROPPED", rid, label[:110]
                for f in fields:
                    writers[f].append(label[:60])

        for field, labels in writers.items():
            # text_detailed_comments legitimately collects several questions.
            if len(labels) > 1 and field not in (
                "text_detailed_comments", "text_reproducibility"
            ):
                yield "OVERWRITTEN", rid, f"{field} <- " + " || ".join(labels)


def fetch(url, cfg, cache):
    path = cache / (re.sub(r"[^A-Za-z0-9]+", "_", url)[-120:] + ".html")
    if path.exists():
        return path.read_text(encoding="utf-8")
    resp = requests.get(url, headers={"User-Agent": cfg["user_agent"]}, timeout=30)
    resp.raise_for_status()
    # Match base.fetch_html: the server omits a charset and requests would
    # otherwise fall back to Latin-1, turning every curly quote into mojibake.
    resp.encoding = "utf-8"
    path.write_text(resp.text, encoding="utf-8")
    time.sleep(random.uniform(1.0, 1.5))
    return resp.text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, help="year to audit")
    ap.add_argument("--all", action="store_true", help="audit every year")
    ap.add_argument("--sample", type=int, default=25, help="papers per year")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--cache", default=".audit_cache", help="directory for fetched pages"
    )
    args = ap.parse_args()
    if not args.year and not args.all:
        ap.error("give --year YYYY or --all")

    conf = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    base = conf["base_settings"]
    import json

    papers = json.loads(
        (ROOT / "data/processed/miccai_all.json").read_text(encoding="utf-8")
    )

    cache = ROOT / args.cache
    cache.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    years = sorted(conf["years"]) if args.all else [args.year]
    faults = 0
    for year in years:
        cfg = dict(base)
        cfg.update(conf["years"][year])
        mod = CHAIN[cfg["era"]]

        pool = [p for p in papers if p["year"] == year and p.get("url")]
        if not pool:
            print(f"\n=== {year}: no papers in miccai_all.json ===")
            continue
        sample = random.sample(pool, min(args.sample, len(pool)))

        seen = defaultdict(set)
        for paper in sample:
            try:
                html = fetch(paper["url"], cfg, cache)
            except Exception as exc:
                print(f"  {year} fetch failed for {paper['paper_id']}: {exc}")
                continue
            for kind, rid, detail in audit_page(html, mod, cfg):
                seen[(kind, detail)].add(paper["paper_id"])

        print(f"\n=== {year} ({len(sample)} papers sampled) ===")
        if not seen:
            print("   clean: every form question reaches a field")
            continue
        for (kind, detail), ids in sorted(seen.items(), key=lambda kv: -len(kv[1])):
            faults += 1
            print(f"   {kind:12s} {len(ids):3d}/{len(sample)} papers  {detail}")

    print()
    if faults:
        print(f"{faults} distinct faults found. Fix the parser, then re-scrape "
              "the affected years; the committed data will not heal itself.")
        return 1
    print("No faults found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
