#!/usr/bin/env python3
"""
Merge all per-year raw JSONs into data/processed/miccai_all.json.

Usage:
    python analysis/normalize.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/miccai_all.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge raw per-year JSONs into miccai_all.json"
    )
    parser.add_argument("--output", default=str(OUT_PATH), help="Output path")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("miccai_????.json"))
    if not raw_files:
        logger.error(f"No miccai_YYYY.json files found in {RAW_DIR}")
        return 1

    all_papers: list[dict] = []
    for path in raw_files:
        with open(path) as f:
            papers = json.load(f)
        ok = [p for p in papers if "error" not in p]
        errored = [p for p in papers if "error" in p]
        logger.info(
            f"{path.name}: {len(ok)} papers loaded, {len(errored)} skipped (errors)"
        )
        all_papers.extend(ok)

    logger.info(f"Total papers: {len(all_papers)}")

    # Sanity check: no duplicate paper_ids
    ids = [p["paper_id"] for p in all_papers]
    dupes = len(ids) - len(set(ids))
    if dupes:
        logger.warning(f"{dupes} duplicate paper_ids found; check raw data")
    else:
        logger.info("No duplicate paper_ids")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    logger.info(f"Written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
