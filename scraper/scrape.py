#!/usr/bin/env python3
"""
Scrape MICCAI papers for a given year.

Usage:
    python scraper/scrape.py --year 2025
    python scraper/scrape.py --year 2025 --limit 10
    python scraper/scrape.py --year 2025 --resume
    python scraper/scrape.py --year 2025 --output data/raw/miccai_2025.json
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from tqdm import tqdm

from scraper.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from scraper.parsers.base import fetch_html

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Single config at the repo root: shared with analysis/build_charts.py and
# analysis/build_site.py
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(year: int) -> dict:
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    base = raw["base_settings"]
    if year not in raw["years"]:
        raise ValueError(
            f"Year {year} not found in config.yaml. Available: {list(raw['years'])}"
        )
    year_cfg = raw["years"][year]
    return {**base, **year_cfg, "_year": year}


def get_era_parsers(era: str):
    if era == "2024_2025":
        from scraper.parsers.era_2024_2025 import (
            parse_categories_page,
            parse_index_page,
            parse_paper_page,
        )

        return parse_categories_page, parse_index_page, parse_paper_page
    if era == "2022_2023":
        from scraper.parsers.era_2022_2023 import (
            parse_categories_page,
            parse_index_page,
            parse_paper_page,
        )

        return parse_categories_page, parse_index_page, parse_paper_page
    if era == "2021":
        from scraper.parsers.era_2021 import (
            parse_categories_page,
            parse_index_page,
            parse_paper_page,
        )

        return parse_categories_page, parse_index_page, parse_paper_page
    raise NotImplementedError(f"Era '{era}' not yet implemented")


def build_paper_list(
    index_html: str,
    cats_html: str,
    cfg: dict,
    parse_index,
    parse_categories,
) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    """
    Combine index page + categories page to get a full, ordered paper list
    and the subject-area map.

    Returns:
        papers: ordered list of (paper_url, paper_id)
        area_map: {filename: [subject_areas]}
    """
    year = cfg["_year"]
    base = cfg["paper_url_base"].rstrip("/")

    area_map: dict[str, list[str]] = parse_categories(cats_html, cfg)

    # Primary ordering comes from the index page
    papers: list[tuple[str, str]] = parse_index(index_html, cfg)
    seen: set[str] = {url.split("/")[-1] for url, _ in papers}

    # Add any papers in categories but not in index (should be rare)
    for filename in area_map:
        if filename in seen:
            continue
        m = re.search(r"Paper(\d{4})\.html", filename)
        if not m:
            continue
        paper_num = m.group(1)
        paper_url = f"{base}/{filename}"
        paper_id = f"miccai-{year}-Paper{paper_num}"
        papers.append((paper_url, paper_id))
        seen.add(filename)

    return papers, area_map


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape MICCAI papers for a given year",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--year", type=int, required=True, help="Year to scrape (e.g. 2025)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max papers to scrape (for testing)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON path"
    )
    args = parser.parse_args()

    cfg = load_config(args.year)
    era = cfg["era"]
    parse_cats, parse_index, parse_paper = get_era_parsers(era)

    out_path = (
        Path(args.output)
        if args.output
        else Path(f"data/raw/miccai_{args.year}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== MICCAI {args.year} scraper (era: {era}) ===")

    # 1. Fetch categories page (subject areas + paper list)
    logger.info(f"Fetching categories page: {cfg['categories_url']}")
    cats_html = fetch_html(cfg["categories_url"], cfg)
    time.sleep(
        random.uniform(cfg["request_delay_min"], cfg["request_delay_max"])
    )

    # 2. Fetch index page (ordered paper list)
    logger.info(f"Fetching index page: {cfg['index_url']}")
    index_html = fetch_html(cfg["index_url"], cfg)
    time.sleep(
        random.uniform(cfg["request_delay_min"], cfg["request_delay_max"])
    )

    # 3. Build master paper list + area map
    all_papers, area_map = build_paper_list(
        index_html, cats_html, cfg, parse_index, parse_cats
    )
    logger.info(f"Total papers discovered: {len(all_papers)}")
    logger.info(f"Papers with subject areas: {len(area_map)}")

    # 4. Handle checkpoint / fresh start
    if args.resume:
        saved = load_checkpoint(args.year)
        completed_ids: set[str] = set(saved.get("completed_ids", []))
        results: list[dict] = saved.get("papers", [])
        logger.info(f"Resuming: {len(completed_ids)} papers already done")
    else:
        completed_ids = set()
        results = []

    # 5. Build todo list (skip completed; apply --limit before starting)
    todo = [(url, pid) for url, pid in all_papers if pid not in completed_ids]
    if args.limit is not None:
        todo = todo[: args.limit]

    logger.info(f"Papers to scrape this run: {len(todo)}")
    if not todo:
        logger.info("Nothing to do.")

    error_count = 0
    checkpoint_interval: int = cfg["checkpoint_interval"]

    for i, (paper_url, paper_id) in enumerate(
        tqdm(todo, desc=f"MICCAI {args.year}", unit="paper")
    ):
        try:
            html = fetch_html(paper_url, cfg)
            filename = paper_url.split("/")[-1]
            subject_areas = area_map.get(filename, [])
            paper = parse_paper(html, paper_url, paper_id, cfg, subject_areas)
            results.append(paper)
        except Exception as exc:
            logger.error(f"Failed {paper_url}: {exc}")
            results.append(
                {"paper_id": paper_id, "url": paper_url, "error": str(exc)}
            )
            error_count += 1

        completed_ids.add(paper_id)

        if (i + 1) % checkpoint_interval == 0:
            save_checkpoint(
                args.year,
                {"completed_ids": list(completed_ids), "papers": results},
            )
            logger.info(f"  checkpoint saved ({i + 1} done)")

        time.sleep(
            random.uniform(cfg["request_delay_min"], cfg["request_delay_max"])
        )

    # 6. Write output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(results)
    logger.info("\n=== Done ===")
    logger.info(f"Papers written : {total}")
    logger.info(f"Errors         : {error_count}")
    logger.info(f"Output         : {out_path}")

    if total > 0 and error_count / total > 0.05:
        logger.warning(
            "ERROR RATE > 5%; review failed URLs before running full scrape"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
