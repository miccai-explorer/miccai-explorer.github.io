#!/usr/bin/env python3
"""
Generate HTML pages from Jinja2 templates.

Reads  : data/processed/miccai_all.json
         templates/base.html, index.html, year.html
Writes : website/index.html
         website/papers.html + website/papers.json  (All Papers browse page)
         website/about.html
         website/year/YYYY.html  (for each year in data)

Usage:
    python analysis/build_site.py
"""

import json
import logging
import os
import re
import shutil
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import logo_assets
import oral_stats
import papers_index
import yaml
from jinja2 import Environment, FileSystemLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

ALL_JSON = Path("data/processed/miccai_all.json")
CONFIG = Path("config.yaml")
SUBS_YAML = Path("num_papers_submitted.yaml")
TEMPLATES = Path("templates")
WEBSITE = Path("website")
STATIC_SRC = Path("static")  # tracked source for style.css + logos
# The site is deployed as a GitHub organization site, from the repository
# miccai-explorer/miccai-explorer.github.io, which GitHub serves at the ROOT of
# the domain rather than under a repository path. BASE_URL is therefore empty:
# every template writes it as "{{ base_url }}/assets/..." and so on, which
# yields a correct root-relative "/assets/...".
#
# If this ever moves to a project site (served at
# https://<owner>.github.io/<repo>/), set BASE_URL to "/<repo>" with a leading
# slash and no trailing one; nothing else needs to change. A wrong value here
# is silent and total: every stylesheet, chart iframe, and nav link 404s, and
# the site renders as unstyled HTML with no charts.
BASE_URL = ""
SITE_URL = "https://miccai-explorer.github.io"

# Per-year accent colors from each year's conference logo primaries
# (kept in sync with YEAR_PALETTES in build_charts.py; source: logo_colors.yaml)
YEAR_META_RAW = {
    2026: {
        "logo": "miccai_2026.png",
        "city": "Strasbourg, France",
        "has_white_bg": False,
        "color": "#293587",
        "color2": "#00adf0",
    },
    2025: {
        "logo": "miccai_2025.png",
        "city": "Daejeon, Korea",
        "has_white_bg": False,
        "color": "#2b50a3",
        "color2": "#d2242b",
    },
    2024: {
        "logo": "miccai_2024.png",
        "city": "Marrakesh, Morocco",
        "has_white_bg": False,
        "color": "#2d835d",
        "color2": "#ce5258",
    },
    2023: {
        "logo": "miccai_2023.png",
        "city": "Vancouver, Canada",
        "has_white_bg": False,
        "color": "#559e39",
        "color2": "#e81e25",
    },
    2022: {
        "logo": "miccai_2022.png",
        "city": "Singapore",
        "has_white_bg": False,
        "color": "#23408f",
        "color2": "#ed2425",
    },
    2021: {
        "logo": "miccai_2021.png",
        "city": "Strasbourg, France (Virtual)",
        "has_white_bg": False,
        "color": "#27368a",
        "color2": "#00aeef",
    },
}

VERDICTS = {
    "Strong Reject",
    "Reject",
    "Weak Reject",
    "Weak Accept",
    "Accept",
    "Strong Accept",
}
VERDICT_RANK = {
    "Strong Reject": 0,
    "Reject": 1,
    "Weak Reject": 2,
    "Weak Accept": 3,
    "Accept": 4,
    "Strong Accept": 5,
}


def _verdict(label: str | None) -> str | None:
    if not label:
        return None
    return (
        re.split(r"\s*[\u2014\u2013-]\s*", label.strip(), maxsplit=1)[
            0
        ].strip()
        or None
    )


def _ns(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**d)


def _hex_mix(color: str, other: str, t: float) -> str:
    """Blend `color` toward `other` by fraction t (both '#rrggbb')."""
    c1 = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    c2 = [int(other[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(
        f"{round(a + (b - a) * t):02x}" for a, b in zip(c1, c2)
    )


def _short_subject(label: str) -> str:
    if " -> " in label:
        return label.split(" -> ", 1)[1]
    if " - " in label:
        return label.split(" - ", 1)[1]
    return label


def format_int(value: int) -> str:
    return f"{int(value):,}"


def load_submissions() -> dict:
    """{year: number of papers submitted} from num_papers_submitted.yaml."""
    if not SUBS_YAML.exists():
        return {}
    with open(SUBS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


FACTS_2026 = Path("data/miccai_2026_facts.yaml")
PROGRAM_2026 = Path("data/raw/miccai_2026_program.json")


def load_2026() -> tuple:
    """(facts, papers) for the partial 2026 year, or (None, None).

    2026 is published from the program booklet alone, months before its reviews
    exist. It stays out of miccai_all.json so it cannot reach any cross-year
    chart: a title-only embedding puts a third of papers in the wrong cluster,
    and the subject-area filter would empty Trending Topics.
    """
    if not (FACTS_2026.exists() and PROGRAM_2026.exists()):
        return None, None
    with open(FACTS_2026, encoding="utf-8") as f:
        facts = yaml.safe_load(f)
    with open(PROGRAM_2026, encoding="utf-8") as f:
        return facts, json.load(f)


def compute_2026_stats(facts: dict, papers: list, meta: dict) -> SimpleNamespace:
    """The partial year's stat strip. No placeholder tiles: With Code and Avg
    Review Score would both be dead, and a strip half full of "pending" reads
    as a broken page rather than an honest one, so Acceptance Rate and Talks
    take those two slots."""
    authors: set = set()
    for p in papers:
        authors.update(p.get("authors", []))
    n = facts["n_accepted"]
    return _ns(
        {
            "n_papers": n,
            "n_authors": len(authors),
            "n_submitted": facts["n_submitted"],
            "pct_accept": round(n / facts["n_submitted"] * 100, 1),
            "n_orals": facts["n_orals"],
            "n_spotlights": facts["n_spotlights"],
            "n_talks": facts["n_orals"] + facts["n_spotlights"],
            "pct_early_subs": facts["pct_early_subs"],
            "pct_early_subs_approx": facts.get("pct_early_subs_approx", False),
            "city": meta.get("city", ""),
        }
    )


def compute_year_stats(
    papers_yr: list, meta: dict, n_submitted: int | None = None
) -> SimpleNamespace:
    n = len(papers_yr)
    authors: set = set()
    for p in papers_yr:
        authors.update(p.get("authors", []))

    n_code = sum(1 for p in papers_yr if p.get("has_code"))
    n_early = sum(1 for p in papers_yr if p.get("early_accepted"))

    # Raw scale; each year's scale differs (1-6 / 1-8 / 1-9); shown as "x / max"
    avg_scores = [
        p["avg_score_raw"]
        for p in papers_yr
        if p.get("avg_score_raw") is not None
    ]
    avg = round(sum(avg_scores) / len(avg_scores), 2) if avg_scores else 0

    return _ns(
        {
            "n_papers": n,
            "n_authors": len(authors),
            "pct_code": round(n_code / n * 100, 1) if n else 0,
            "avg_score": avg,
            # Two different denominators, both useful and easy to confuse:
            # pct_early is early accepts as a share of ACCEPTED papers (~30%),
            # pct_early_subs as a share of ALL SUBMISSIONS (~8-13%). The
            # rebuttal caption uses the submissions figure so it matches the
            # "Early Accepted" panel on the overview page.
            "pct_early": round(n_early / n * 100, 1) if n else 0,
            "pct_early_subs": round(n_early / n_submitted * 100, 1)
            if n_submitted
            else 0,
            "city": meta.get("city", ""),
        }
    )


def compute_year_title_lists(papers_yr: list) -> SimpleNamespace:
    valid = [
        p for p in papers_yr if p.get("title") and len(p["title"].split()) >= 1
    ]
    by_len = sorted(valid, key=lambda p: len(p["title"].split()), reverse=True)

    def _item(p, *, word_count=None, n_authors=None):
        d = {"title": p["title"], "url": p.get("url", "")}
        if word_count is not None:
            d["word_count"] = word_count
        if n_authors is not None:
            d["n_authors"] = n_authors
        return d

    longest = [
        _item(p, word_count=len(p["title"].split())) for p in by_len[:5]
    ]
    shortest = [
        _item(p, word_count=len(p["title"].split()))
        for p in reversed(by_len[-5:])
    ]

    by_authors = sorted(
        valid, key=lambda p: len(p.get("authors", [])), reverse=True
    )
    most_authors = [
        _item(p, n_authors=len(p.get("authors", []))) for p in by_authors[:5]
    ]

    return _ns(
        {
            "longest": longest,
            "shortest": shortest,
            "most_authors": most_authors,
        }
    )


# "Poster only" is the right phrase on the orals page, where it is being
# contrasted with the other two; in a table column "Poster" is enough.
_PRESENTATION_LABEL = {
    "oral": "Oral",
    "spotlight": "Spotlight",
    "poster": "Poster",
}


def compute_top_papers(
    papers_yr: list, n: int = 20
) -> tuple[list, list, str | None]:
    """Top-N tables for the year page (N = num_top_papers in config.yaml).
    Returns (overall, rebuttal, rebuttal_mode).

    overall  : ranked by mean raw reviewer score (desc). All years.
    rebuttal : depends on what post-rebuttal data the year's reviews carry -
      mode "scores"   (2022/2023/2024): numeric post-rebuttal scores; ranked by
                      mean post-rebuttal score (ties: biggest improvement).
      mode "verdicts" (2025): post-rebuttal verdict labels only (Accept/Reject);
                      ranked by mean verdict rank, then improvement.
      mode None       (2021): no post-rebuttal review data exists at all; the
                      template renders an explanatory note instead of a table.
    In both modes, reviewers who did not respond post-rebuttal keep their
    original score/verdict; otherwise papers where a single enthusiastic
    reviewer responded would outrank uniformly strong papers.
    """
    overall, reb_scores, reb_verdicts = [], [], []
    for p in papers_yr:
        reviews = p.get("reviews", [])
        pre = [
            r["score_raw"] for r in reviews if r.get("score_raw") is not None
        ]
        if not pre:
            continue
        subjects = p.get("subject_areas") or []
        row = {
            "title": p.get("title", ""),
            "url": p.get("url", ""),
            "subject": _short_subject(subjects[0]) if subjects else "n/a",
            "subject_full": "; ".join(subjects),
            "mean": sum(pre) / len(pre),
            "scores": ", ".join(str(s) for s in pre),
            "early": bool(p.get("early_accepted")),
            # Set by oral_stats.attach_types() before this runs; None when
            # data/processed/orals.json is absent, in which case the template
            # omits the column entirely rather than printing a blank one.
            "presentation": _PRESENTATION_LABEL.get(
                p.get("presentation_type")
            ),
        }
        overall.append(row)

        # --- numeric post-rebuttal scores (carry-forward for non-responders) ---
        post, n_updated = [], 0
        for r in reviews:
            ps = r.get("post_rebuttal_score_raw")
            if ps is None:
                # 2023 stores the numeric post-rebuttal score as a bare-integer label
                lbl = (r.get("post_rebuttal_label") or "").strip()
                if re.fullmatch(r"\d+", lbl):
                    ps = int(lbl)
            if ps is not None:
                n_updated += 1
            elif r.get("score_raw") is not None:
                ps = r["score_raw"]
            if ps is not None:
                post.append(ps)
        if n_updated and post:
            reb_scores.append(
                {
                    **row,
                    "mean_post": sum(post) / len(post),
                    "post_scores": ", ".join(str(s) for s in post),
                }
            )
            continue  # numeric mode wins; no need to also collect verdicts

        # --- verdict labels only (2025: Accept/Reject, no numbers) ---
        verdicts, pre_ranks, n_responded = [], [], 0
        for r in reviews:
            pre_v = _verdict(r.get("recommendation_label"))
            post_v = _verdict(r.get("post_rebuttal_label"))
            if post_v in VERDICT_RANK:
                n_responded += 1
                verdicts.append(post_v)
            elif pre_v in VERDICT_RANK:
                verdicts.append(pre_v)
            if pre_v in VERDICT_RANK:
                pre_ranks.append(VERDICT_RANK[pre_v])
        if n_responded and verdicts:
            ranks = [VERDICT_RANK[v] for v in verdicts]
            mean_rank = sum(ranks) / len(ranks)
            mean_pre_rank = (
                sum(pre_ranks) / len(pre_ranks) if pre_ranks else mean_rank
            )
            reb_verdicts.append(
                {
                    **row,
                    "mean_rank": mean_rank,
                    "improvement": mean_rank - mean_pre_rank,
                    "post_verdicts": ", ".join(verdicts),
                }
            )

    overall.sort(key=lambda r: (-r["mean"], r["title"]))
    if reb_scores:
        reb_scores.sort(
            key=lambda r: (
                -r["mean_post"],
                -(r["mean_post"] - r["mean"]),
                r["title"],
            )
        )
        return overall[:n], reb_scores[:n], "scores"
    if reb_verdicts:
        reb_verdicts.sort(
            key=lambda r: (
                -r["mean_rank"],
                -r["improvement"],
                -r["mean"],
                r["title"],
            )
        )
        return overall[:n], reb_verdicts[:n], "verdicts"
    return overall[:n], [], None


def has_rebuttal_verdicts(papers_yr: list) -> bool:
    """True if reviews carry post-rebuttal verdict labels (2024/2025) -
    the pre/post Sankey only exists for those years."""
    for p in papers_yr:
        for r in p.get("reviews", []):
            # _verdict is defined above and does exactly this split; the
            # inline copy that used to be here is how the two drift apart.
            # It returns None rather than "" for an empty label, which behaves
            # identically since neither is ever a member of VERDICTS.
            if _verdict(r.get("post_rebuttal_label")) in VERDICTS:
                return True
    return False


CHART_HEIGHTS_JSON = Path("website/charts/chart_heights.json")

# Frames whose height is fixed and does not depend on how many rows were drawn.
_STATIC_HEIGHTS = {
    "map": 520,
    "scores": 380,
    "controversy": 380,
    "rebuttal": 460,
    "coauthor": 600,
    "naming": 420,
}

# Charts that share a two-column row and must therefore share a height.
_HEIGHT_ROWS = (("subjects", "authors"), ("code", "buzzwords"))


def load_chart_heights() -> dict:
    """Per-chart frame heights written by build_charts.py, or {} if absent."""
    if not CHART_HEIGHTS_JSON.exists():
        return {}
    with open(CHART_HEIGHTS_JSON, encoding="utf-8") as f:
        return json.load(f)


def row_heights(heights: dict, year: int) -> dict:
    """Frame heights for one year page, with paired charts equalized.

    subjects/authors and code/buzzwords each sit in a two-column row. Sizing
    them independently left one chart visibly shorter than the other with a
    band of empty card below it, so each pair takes the taller of the two.
    Falls back to the static defaults when chart_heights.json is missing, which
    keeps build_site.py runnable on its own.
    """
    out = dict(_STATIC_HEIGHTS)
    for pair in _HEIGHT_ROWS:
        vals = [heights.get(f"{name}_{year}") for name in pair]
        vals = [v for v in vals if v]
        shared = max(vals) if vals else 640
        for name in pair:
            out[name] = shared
    return out


def main() -> int:
    with open(ALL_JSON, encoding="utf-8") as f:
        papers = json.load(f)
    logger.info(f"Loaded {len(papers)} papers")

    # data_years are the years with real data and drive every existing loop.
    # 2026 has no reviews, abstracts or subject areas so it is deliberately not
    # in miccai_all.json; it joins only `years`, which base.html's nav and
    # index.html's year grid iterate.
    data_years = sorted(set(p["year"] for p in papers), reverse=True)
    facts_2026, papers_2026 = load_2026()
    partial_years = [2026] if facts_2026 else []
    years = partial_years + data_years
    logger.info(f"Years: {data_years}  partial: {partial_years}")

    # Presentation types, patched in memory (never written back to
    # miccai_all.json) - the same pattern build_charts.py uses. Without
    # data/processed/orals.json every type is None, the Orals page is skipped
    # and its nav entry is hidden, so the rest of the site is unaffected.
    orals_raw = oral_stats.load_orals()
    n_orals = oral_stats.attach_types(papers, orals_raw)
    orals_summary = (
        oral_stats.page_summary(papers, orals_raw) if n_orals else {}
    )
    if n_orals:
        logger.info(
            f"Attached presentation types to {n_orals} papers "
            f"({orals_summary.get('total_oral', 0)} oral, "
            f"{orals_summary.get('total_spotlight', 0)} spotlight)"
        )

    with open(CONFIG, encoding="utf-8") as f:
        root_cfg = yaml.safe_load(f) or {}
    year_cfg = root_cfg.get("years", {})
    site_cfg = root_cfg.get("site_settings") or {}
    # The environment wins over the config file. The measurement id is not a
    # secret (it is visible in the page source of every deployed site), but
    # keeping it out of the repository means a fork that builds this project
    # does not silently report its traffic to someone else's property.
    analytics_id = (
        os.environ.get("MICCAI_GA_ID")
        or site_cfg.get("google_analytics_id")
        or ""
    ).strip()
    if analytics_id:
        logger.info(f"  Google Analytics tag written ({analytics_id})")
    else:
        logger.info("  No Google Analytics tag (none configured)")

    # Links to the source repository and the write-up, shown on the About page;
    # the repository is also in the footer of every page. An empty value writes
    # no link at all, so a fork that has not set its own links back to nobody.
    repo_url = (site_cfg.get("repo_url") or "").strip()
    blog_url = (site_cfg.get("blog_url") or "").strip()

    # ── static assets (style.css + logos) → website/assets ─────
    # Done before year_meta is built, because shrinking the logos renames them
    # from .png to .webp and year_meta carries the filename the templates use.
    WEBSITE.mkdir(parents=True, exist_ok=True)
    logo_names: dict[str, str] = {}
    if STATIC_SRC.exists():
        shutil.copytree(STATIC_SRC, WEBSITE / "assets", dirs_exist_ok=True)
        logger.info("  Copied static/ → website/assets/")
        logo_names = logo_assets.optimize(
            STATIC_SRC / "logos", WEBSITE / "assets" / "logos"
        )

    year_meta = {}
    for yr in years:
        raw = dict(
            YEAR_META_RAW.get(
                yr,
                {
                    "logo": f"miccai_{yr}.png",
                    "city": str(yr),
                    "has_white_bg": False,
                    "color": "#0891b2",
                    "color2": "#22d3ee",
                },
            )
        )
        # .png source, .webp on the site; unchanged when Pillow is absent.
        raw["logo"] = logo_names.get(raw["logo"], raw["logo"])
        # Derived accent variants: light tint for badges, lightened for the dark strip
        raw["accent_bg"] = _hex_mix(raw["color"], "#ffffff", 0.88)
        raw["color_on_dark"] = _hex_mix(raw["color"], "#ffffff", 0.55)
        year_meta[yr] = _ns(raw)

    submissions = load_submissions()
    year_stats: dict[int, SimpleNamespace] = {}
    year_title_lists: dict[int, SimpleNamespace] = {}
    year_top_papers: dict[int, tuple] = {}
    year_has_sankey: dict[int, bool] = {}
    if facts_2026:
        year_stats[2026] = compute_2026_stats(
            facts_2026, papers_2026, YEAR_META_RAW.get(2026, {})
        )
    for yr in data_years:
        papers_yr = [p for p in papers if p["year"] == yr]
        year_stats[yr] = compute_year_stats(
            papers_yr, YEAR_META_RAW.get(yr, {}), submissions.get(yr)
        )
        year_title_lists[yr] = compute_year_title_lists(papers_yr)
        year_top_papers[yr] = compute_top_papers(
            papers_yr, year_cfg.get(yr, {}).get("num_top_papers", 20)
        )
        year_has_sankey[yr] = has_rebuttal_verdicts(papers_yr)

    # Totals for index page
    all_authors: set = set()
    for p in papers:
        all_authors.update(p.get("authors", []))
    total_with_code = sum(1 for p in papers if p.get("has_code"))
    n_clusters = len(set(p["cluster_id"] for p in papers))

    active_model_file = Path("data/processed/active_model.txt")
    active_proj_file = Path("data/processed/active_proj.txt")
    embed_model = (
        active_model_file.read_text().strip()
        if active_model_file.exists()
        else "unknown"
    )
    embed_proj = (
        active_proj_file.read_text().strip()
        if active_proj_file.exists()
        else "umap"
    )

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=False,
    )
    env.filters["format_int"] = format_int

    common_ctx = dict(
        base_url=BASE_URL,
        site_url=SITE_URL,
        build_date=date.today().isoformat(),
        years=years,
        # years with full data. `years` also carries the partial 2026, which
        # belongs in the nav and the year grid but not in any claim about
        # what the cross-year charts and papers.json actually cover.
        data_years=data_years,
        partial_years=partial_years,
        year_meta=year_meta,
        has_orals_page=bool(orals_summary),
        analytics_id=analytics_id,
        repo_url=repo_url,
        blog_url=blog_url,
    )

    # ── papers.json (data for the All Papers browse page) ───────
    # Built here rather than in build_charts.py so it reuses the papers list
    # that already has presentation types attached; the page can then never
    # disagree with the Orals page about which papers got a talk.
    index = papers_index.build(
        papers, WEBSITE / "papers.json", built=common_ctx["build_date"]
    )
    logger.info(
        f"  Saved website/papers.json "
        f"({len(index['papers'])} papers, {len(index['areas'])} subject areas)"
    )

    # ── index.html ──────────────────────────────────────────────
    tmpl = env.get_template("index.html")
    html = tmpl.render(
        **common_ctx,
        active_page="index",
        active_year=None,
        total_papers=len(papers),
        total_authors=len(all_authors),
        total_with_code=total_with_code,
        pct_code=round(total_with_code / len(papers) * 100, 1),
        n_clusters=n_clusters,
        embed_model=embed_model,
        embed_proj=embed_proj,
        year_stats=year_stats,
    )
    (WEBSITE / "index.html").write_text(html, encoding="utf-8")
    logger.info("  Saved website/index.html")

    # ── orals.html ──────────────────────────────────────────────
    orals_out = WEBSITE / "orals.html"
    if orals_summary:
        tmpl = env.get_template("orals.html")
        html = tmpl.render(
            **common_ctx,
            active_page="orals",
            active_year=None,
            orals=orals_summary,
            embed_model=embed_model,
            embed_proj=embed_proj,
        )
        orals_out.write_text(html, encoding="utf-8")
        logger.info("  Saved website/orals.html")
    else:
        # No oral data: remove any page left over from an earlier build so the
        # site never serves stale numbers behind a hidden nav entry.
        orals_out.unlink(missing_ok=True)
        logger.info("  No oral data; skipped website/orals.html")

    # ── papers.html (All Papers browse page) ────────────────────
    # Always written, unlike orals.html: the only data it needs is the paper
    # list itself, so there is no optional input whose absence could make it
    # stale. Without orals.json every paper simply reads as a poster.
    tmpl = env.get_template("papers.html")
    html = tmpl.render(
        **common_ctx,
        active_page="papers",
        active_year=None,
        total_papers=len(papers),
    )
    (WEBSITE / "papers.html").write_text(html, encoding="utf-8")
    logger.info("  Saved website/papers.html")

    # ── about.html ──────────────────────────────────────────────
    # Always written, for the same reason as papers.html: it reads no data, so
    # there is nothing optional that could be missing. The repository and blog
    # links inside it are individually skipped if unset.
    tmpl = env.get_template("about.html")
    html = tmpl.render(
        **common_ctx,
        active_page="about",
        active_year=None,
    )
    (WEBSITE / "about.html").write_text(html, encoding="utf-8")
    logger.info("  Saved website/about.html")

    # ── year/YYYY.html ───────────────────────────────────────────
    chart_heights = load_chart_heights()

    year_dir = WEBSITE / "year"
    year_dir.mkdir(parents=True, exist_ok=True)
    tmpl = env.get_template("year.html")

    for yr in data_years:
        papers_yr = [p for p in papers if p["year"] == yr]
        top_overall, top_rebuttal, rebuttal_mode = year_top_papers[yr]
        cfg = year_cfg.get(yr, {})
        html = tmpl.render(
            **common_ctx,
            active_page=None,
            active_year=yr,
            year=yr,
            meta=year_meta[yr],
            stats=year_stats[yr],
            title_lists=year_title_lists[yr],
            top_overall=top_overall,
            top_rebuttal=top_rebuttal,
            rebuttal_mode=rebuttal_mode,
            top_n=cfg.get("num_top_papers", 20),
            has_sankey=year_has_sankey[yr],
            heights=row_heights(chart_heights, yr),
            scale_max=cfg.get("review_scale_max", 6),
            coauthor_min=cfg.get("coauthorship_network_min_papers", 2),
            n_clusters=n_clusters,
            embed_model=embed_model,
            embed_proj=embed_proj,
        )
        (year_dir / f"{yr}.html").write_text(html, encoding="utf-8")
        logger.info(f"  Saved website/year/{yr}.html")

    # ── year/2026.html (partial year, its own template) ─────────
    page_2026 = year_dir / "2026.html"
    if facts_2026:
        html = env.get_template("year_2026.html").render(
            **common_ctx,
            active_page=None,
            active_year=2026,
            year=2026,
            meta=year_meta[2026],
            stats=year_stats[2026],
            heights=row_heights(chart_heights, 2026),
            coauthor_min=facts_2026["coauthor_min"],
        )
        page_2026.write_text(html, encoding="utf-8")
        logger.info("  Saved website/year/2026.html")
    elif page_2026.exists():
        page_2026.unlink()
        logger.info("  Removed stale website/year/2026.html")

    logger.info("Site build complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
