"""Parser for MICCAI 2021 paper pages.

HTML structure is nearly identical to 2022/2023 (same section ids, same review/meta-review
layout, same postrebuttal-id early_accepted signal). Key differences:
  - Score field: "state your overall opinion of the paper" (not "scale of 1-8")
  - Score format: "Probably accept (7)"; parens mode, not bare integer
  - Score scale: 1-9
  - Index page hrefs include a date path: 2021/09/01/NNN-PaperNNNN.html
  - No BibTeX section
  - No post-rebuttal reviewer scores
All differences are handled via config.yaml; parse_paper_page delegates to era_2022_2023.
"""

import re

from bs4 import BeautifulSoup

from scraper.parsers.era_2022_2023 import (  # noqa: F401  (same structure)  # noqa: F401  (same structure; score mode read from cfg)
    parse_categories_page,
    parse_paper_page,
)

# ---------------------------------------------------------------------------
# Index page: custom because hrefs include a date path segment
# ---------------------------------------------------------------------------


def parse_index_page(html: str, cfg: dict) -> list[tuple[str, str]]:
    """Parse 2021 index page. Hrefs are relative: 2021/09/01/NNN-PaperNNNN.html."""
    soup = BeautifulSoup(html, "lxml")
    year = cfg["_year"]
    base = cfg["paper_url_base"].rstrip("/")

    papers: list[tuple[str, str]] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.search(r"(\d{4}/\d{2}/\d{2}/\d{3,4}-Paper(\d{4})\.html)", href)
        if not m:
            continue
        full_path = m.group(1)  # e.g. 2021/09/01/001-Paper1891.html
        paper_num = m.group(2)  # e.g. 1891
        filename = full_path.split("/")[-1]
        if filename in seen:
            continue
        seen.add(filename)
        paper_url = f"{base}/{full_path}"
        paper_id = f"miccai-{year}-Paper{paper_num}"
        papers.append((paper_url, paper_id))

    return papers
