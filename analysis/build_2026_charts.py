#!/usr/bin/env python3
"""The three charts a 2026 page can honestly draw.

2026 has titles and authors and nothing else, so only the charts that read
those two fields are built. Buzzwords is deliberately absent: it matches over
title plus abstract, and on titles alone it keeps 25% of "Transformers / ViT"
and 29% of "Classification", which would draw a collapse that never happened.

Called from build_charts.main(), the way build_oral_charts.build_all() is, so
one command still builds the whole site. No-ops and clears its own stale output
when the 2026 program is absent, which keeps the feature strictly additive.
"""

import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROGRAM = Path("data/raw/miccai_2026_program.json")
FACTS = Path("data/miccai_2026_facts.yaml")
CHART_DIR = Path("website/charts")
OUTPUTS = ("authors_2026.html", "coauthor_2026.html", "naming_2026.html")


def build_all(bc, network_renderer: str = "d3") -> None:
    """bc is the live build_charts module, passed in rather than imported.

    build_charts runs as __main__, so `import build_charts` here would create a
    SECOND module object with its own empty CHART_HEIGHTS. The charts would
    still be written but their frame heights would land in the wrong dict and
    never reach build_site.py, which is how authors_2026 first came out missing
    from chart_heights.json while 2021-2025 were all present.
    """
    if not (PROGRAM.exists() and FACTS.exists()):
        for name in OUTPUTS:
            stale = CHART_DIR / name
            if stale.exists():
                stale.unlink()
                logger.info(f"  Removed stale {stale.name}")
        return

    papers = json.loads(PROGRAM.read_text())
    facts = yaml.safe_load(FACTS.read_text())
    logger.info(f"2026 charts ({len(papers)} papers)")

    bc.chart_authors(papers, 2026)
    bc.chart_coauthor(papers, 2026, facts["coauthor_min"], network_renderer)
    bc.chart_naming(papers, 2026)
