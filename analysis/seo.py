"""Search-engine metadata: the page table, sitemap.xml and robots.txt.

Everything a crawler reads about this site is derived from ONE list of Page
records built in `collect_pages()`. The canonical URL in a page's <head>, the
<loc> in sitemap.xml and the page build_site.py actually writes all come from
the same record, so the three cannot drift apart. They did drift in every
hand-maintained sitemap this author has ever seen: a page is added, the sitemap
is not, and nothing reports it because a stale sitemap is still valid XML.
`tests/test_seo.py` closes the remaining gap by comparing the sitemap against
the files on disk rather than against this list.

Both output files are written into website/, not into the repository root.
website/ is gitignored and rebuilt from scratch; it is also the directory
deploy.yml publishes to gh-pages, so it is the only place a file can sit and
actually be served at https://<site>/robots.txt. A robots.txt committed at the
repository root would be served by nothing.

On <meta name="keywords">: it is not written anywhere by this module, and that
is deliberate. Google stopped using it as a ranking signal in 2009 and says so
in public; Bing treats stuffing it as a negative signal. The keywords a search
engine does read are the ones in the title, the description, the headings and
the JSON-LD below, so that is where they are.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)

# A description Google will show whole rather than truncate. The limit is
# pixels, not characters, and it varies by device; ~155 characters is the
# usual safe figure and is what `tests/test_seo.py` enforces.
DESC_MAX = 160


@dataclass
class Page:
    """One indexable page.

    `path` is site-relative and always starts with "/", so the canonical URL is
    site_url + path with no join logic and no doubled slash. The home page is
    "/" rather than "/index.html": those are the same resource, and a crawler
    that finds both counts them as duplicates of each other.
    """

    path: str
    title: str
    description: str
    # Written into the sitemap as <lastmod>. Every page is rebuilt on every
    # deploy, so this is the build date for all of them; it is carried per page
    # anyway, because the day one page gets a real modification date the
    # sitemap should not have to be restructured to say so.
    lastmod: str
    # Extra JSON-LD emitted for this page only, on top of the site-wide graph.
    jsonld: list = field(default_factory=list)

    def url(self, site_url: str) -> str:
        return f"{site_url.rstrip('/')}{self.path}"

    @property
    def social_title(self) -> str:
        """The title without the site-name suffix.

        A social card prints og:site_name on its own line, so an og:title that
        also ends in "| MICCAI Explorer" says it twice and spends characters
        the card would otherwise give to the real subject. The browser tab
        still gets the full title, where the suffix is what identifies the tab
        among twenty others.
        """
        return self.title.split(" | ")[0].strip()


def collect_pages(
    *,
    build_date: str,
    data_years: list[int],
    partial_years: list[int],
    year_meta: dict,
    year_stats: dict,
    totals: dict,
    has_orals_page: bool,
) -> list[Page]:
    """Build the page table from the same numbers the pages themselves print.

    The descriptions carry real counts (3,717 papers, 11,359 reviews) rather
    than adjectives, because a search result is read in one glance and a number
    is the part of it that says whether the page is worth opening. They are
    also therefore never stale: they are computed from the data on every build.
    """
    n_papers = totals["n_papers"]
    n_reviews = totals["n_reviews"]
    span = f"{min(data_years)} to {max(data_years)}"

    pages: list[Page] = [
        Page(
            path="/",
            title=f"MICCAI Papers and Peer Reviews, {span} | MICCAI Explorer",
            description=(
                f"Explore {n_papers:,} MICCAI papers and {n_reviews:,} public "
                f"peer reviews from {span}: semantic map, acceptance rates, "
                f"reviewer scores, and subject-area trends."
            ),
            lastmod=build_date,
        ),
        Page(
            path="/papers.html",
            title=f"All {n_papers:,} MICCAI Papers, {span} | MICCAI Explorer",
            description=(
                f"Search and filter all {n_papers:,} accepted MICCAI papers, "
                f"{span}, by year, subject area, reviewer score, and code "
                f"release. Links to PDFs, DOIs, and code."
            ),
            lastmod=build_date,
        ),
        Page(
            path="/about.html",
            title="About the Data and Method | MICCAI Explorer",
            description=(
                "How MICCAI Explorer is built: the proceedings and reviews "
                "it reads, the SPECTER2 embeddings behind the semantic map, "
                "the statistics, and the known limitations."
            ),
            lastmod=build_date,
        ),
    ]

    if has_orals_page:
        n_talks = totals.get("n_talks", 0)
        pages.insert(
            1,
            Page(
                path="/orals.html",
                title=(
                    "Which MICCAI Papers Get a Talk? Orals and Spotlights "
                    "| MICCAI Explorer"
                ),
                description=(
                    f"{n_talks} MICCAI oral and spotlight talks matched to "
                    f"their papers and reviews: score effect sizes, early-"
                    f"acceptance strata, and which areas get picked."
                ),
                lastmod=build_date,
            ),
        )

    for yr in sorted(data_years, reverse=True):
        st = year_stats[yr]
        city = getattr(year_meta[yr], "city", str(yr))
        pages.append(
            Page(
                path=f"/year/{yr}.html",
                title=f"MICCAI {yr} ({city}): Papers and Reviews "
                f"| MICCAI Explorer",
                description=(
                    f"MICCAI {yr} in {city}: {st.n_papers:,} accepted papers "
                    f"with reviewer scores, subject areas, top-rated papers, "
                    f"co-authorship, and code-release rates."
                ),
                lastmod=build_date,
            )
        )

    # Partial years (a program booklet, no reviews yet) are indexable and
    # described as partial, so the snippet does not promise review analysis
    # the page does not carry.
    for yr in sorted(partial_years, reverse=True):
        st = year_stats.get(yr)
        city = getattr(year_meta[yr], "city", str(yr))
        n = getattr(st, "n_papers", 0) if st else 0
        pages.append(
            Page(
                path=f"/year/{yr}.html",
                title=f"MICCAI {yr} ({city}): Accepted Papers "
                f"| MICCAI Explorer",
                description=(
                    f"MICCAI {yr} in {city}: {n:,} accepted papers from the "
                    f"program booklet, with authors and co-authorship. Peer "
                    f"reviews are not published yet."
                ),
                lastmod=build_date,
            )
        )

    _check_descriptions(pages)
    return pages


def _check_descriptions(pages: list[Page]) -> None:
    """Warn, do not raise, on a description a search engine would truncate.

    Raising would fail a deploy over a snippet that is merely cut short, which
    is not proportionate. The build log is enough, and the test suite asserts
    the same rule, so an over-long description cannot reach a release unnoticed
    while still not blocking a local preview.
    """
    seen: dict[str, str] = {}
    for p in pages:
        if len(p.description) > DESC_MAX:
            logger.warning(
                f"  SEO: description for {p.path} is {len(p.description)} "
                f"chars (> {DESC_MAX}); search engines will truncate it"
            )
        if p.description in seen:
            logger.warning(
                f"  SEO: {p.path} and {seen[p.description]} share a "
                f"description; duplicate snippets compete with each other"
            )
        seen[p.description] = p.path


# ── JSON-LD ────────────────────────────────────────────────────

def site_jsonld(
    *,
    site_url: str,
    author_name: str,
    author_url: str,
    repo_url: str,
    data_years: list[int],
    totals: dict,
    build_date: str,
    license_url: str,
) -> str:
    """Site-wide structured data, as one @graph.

    Three types, because this site is three things at once and a crawler is
    told about each separately:

    - WebSite, which is what carries the site name into a result listing.
    - Dataset, which is what Google Dataset Search indexes. This is the entry
      that matters most here: a researcher looking for "MICCAI peer review
      data" is far more likely to run that search than to search for this
      site by name, and no other Dataset entry for this data exists.
    - Person, so the attribution is machine-readable rather than only printed
      in the footer. `creator` on the Dataset points at it.

    Dataset requires `name`, `description` and (in practice) `creator`,
    `license` and `distribution` to be eligible; the optional `temporalCoverage`
    and `keywords` are what make it match a topical query.
    """
    span = f"{min(data_years)}/{max(data_years)}"
    person = {
        "@type": "Person",
        "@id": f"{site_url}/#author",
        "name": author_name,
    }
    if author_url:
        person["url"] = author_url

    dataset = {
        "@type": "Dataset",
        "@id": f"{site_url}/#dataset",
        "name": (
            f"MICCAI papers, peer reviews and presentation program, "
            f"{min(data_years)} to {max(data_years)}"
        ),
        "description": (
            f"{totals['n_papers']:,} accepted MICCAI papers with "
            f"{totals['n_reviews']:,} peer reviews, meta-reviews, rebuttals, "
            f"reviewer scores and confidence ratings, joined to the official "
            f"oral and spotlight program and to SPECTER2 semantic embeddings. "
            f"Covers MICCAI {min(data_years)} to {max(data_years)}."
        ),
        "url": f"{site_url}/",
        "creator": {"@id": f"{site_url}/#author"},
        "temporalCoverage": span,
        "isAccessibleForFree": True,
        "dateModified": build_date,
        "license": license_url,
        "keywords": [
            "MICCAI",
            "peer review",
            "open peer review",
            "reviewer scores",
            "rebuttal",
            "meta-review",
            "medical image computing",
            "computer assisted intervention",
            "medical image analysis",
            "conference acceptance rate",
            "oral presentation",
            "spotlight presentation",
            "bibliometrics",
            "science of science",
            "scholarly data",
        ],
        "measurementTechnique": (
            "Parsed from the MICCAI open-access proceedings and the official "
            "program booklets; embedded with SPECTER2 and clustered with "
            "k-means."
        ),
        "variableMeasured": [
            "reviewer score",
            "reviewer confidence",
            "post-rebuttal recommendation",
            "early acceptance",
            "presentation type",
            "subject area",
            "code availability",
        ],
    }
    if repo_url:
        dataset["distribution"] = [
            {
                "@type": "DataDownload",
                "encodingFormat": "application/json",
                "contentUrl": f"{site_url}/papers.json",
            },
            {
                "@type": "DataDownload",
                "encodingFormat": "text/html",
                "contentUrl": repo_url,
            },
        ]
        dataset["citation"] = repo_url

    graph = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "WebSite",
                "@id": f"{site_url}/#website",
                "url": f"{site_url}/",
                "name": "MICCAI Explorer",
                "description": (
                    "Interactive explorer for MICCAI papers and their public "
                    "peer reviews."
                ),
                "inLanguage": "en",
                "author": {"@id": f"{site_url}/#author"},
                "license": license_url,
            },
            person,
            dataset,
        ],
    }
    # separators: JSON-LD is machine-read only, so the whitespace is waste.
    #
    # "<" is escaped as \u003c, which is ordinary JSON and parses identically.
    # It is there because this string is written between <script> tags in a
    # template with autoescape off: a "</script>" appearing inside any value
    # would end the script element early and spill the rest of the graph into
    # the page as text. No value carries one today; the escape means none can.
    return json.dumps(
        graph, ensure_ascii=False, separators=(",", ":")
    ).replace("<", "\\u003c")


# ── sitemap.xml ────────────────────────────────────────────────

def build_sitemap(pages: list[Page], site_url: str, out: Path) -> str:
    """Write a sitemaps.org 0.9 urlset.

    Only <loc> and <lastmod> are written. <changefreq> and <priority> are part
    of the schema but Google has stated it ignores both, and a priority number
    that every page shares says nothing anyway; writing them would be noise a
    reader of this file would later have to interpret.

    Chart files under /charts/ are deliberately absent. They are iframe
    fragments with no navigation and no context, so a reader who landed on one
    from a search result would have no way back into the site; base.html's
    counterpart to this is the noindex tag those files carry.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for p in pages:
        lines.append("  <url>")
        lines.append(f"    <loc>{escape(p.url(site_url))}</loc>")
        lines.append(f"    <lastmod>{escape(p.lastmod)}</lastmod>")
        lines.append("  </url>")
    lines.append("</urlset>")
    xml = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding="utf-8")
    logger.info(f"  Saved website/sitemap.xml ({len(pages)} URLs)")
    return xml


# ── robots.txt ─────────────────────────────────────────────────

# The crawlers that feed model training sets and the retrieval layer behind
# search-augmented assistants. Every one of them is already covered by the
# wildcard block below, so naming them changes no behaviour. They are named
# because blocking this list has become the common default, and a robots.txt
# that is silent about them leaves a reader unable to tell an allow from an
# oversight. Here it is a decision: this analysis exists to be read and cited,
# by people and by the assistants answering on their behalf.
AI_AGENTS = [
    ("ClaudeBot", "Anthropic, training and retrieval"),
    ("Claude-User", "Anthropic, fetches a page a user asked about"),
    ("Claude-SearchBot", "Anthropic, search index"),
    ("GPTBot", "OpenAI, training"),
    ("OAI-SearchBot", "OpenAI, search index"),
    ("ChatGPT-User", "OpenAI, fetches a page a user asked about"),
    ("PerplexityBot", "Perplexity, search index"),
    ("Google-Extended", "Google, Gemini training and grounding"),
    ("Applebot-Extended", "Apple, training"),
    ("CCBot", "Common Crawl, which many models are trained from"),
    ("Meta-ExternalAgent", "Meta, training"),
    ("Bytespider", "ByteDance, training"),
]


def build_robots(site_url: str, out: Path) -> str:
    """Write a robots.txt that allows everything and names the sitemap.

    "Allow all" is the whole policy, so the substance is two directives plus
    the sitemap pointer. The one thing this must get right is the Sitemap
    line: that is how a crawler that was never submitted the sitemap finds it,
    and it has to be an absolute URL, which is why this takes site_url.

    /charts/ is NOT disallowed. Those files are the content of the iframes on
    every chart page, and a crawler that cannot fetch them sees the pages as
    nearly empty. They are kept out of the index with a noindex tag instead,
    which needs the crawler to be able to read the file in order to obey it;
    disallowing them here would prevent exactly that.
    """
    base = site_url.rstrip("/")
    lines = [
        "# https://www.robotstxt.org/robotstxt.html",
        "# Everything here is public, derived from the MICCAI open-access",
        "# proceedings, and meant to be found. Nothing is disallowed.",
        "#",
        "# Chart files under /charts/ are crawlable on purpose: they are the",
        "# iframe content of the analysis pages, so blocking them would make",
        "# those pages look empty to a renderer. They carry a noindex tag of",
        "# their own so they do not turn up as standalone results, and a",
        "# crawler has to be allowed to fetch a file in order to see that tag.",
        "",
        "User-agent: *",
        "Allow: /",
        "",
        "# The AI crawlers, named explicitly. The wildcard above already",
        "# permits all of them, so these blocks change nothing; they are here",
        "# because blocking this list has become a common default, and silence",
        "# would leave a reader guessing whether that was the intent. It is",
        "# not. See /llms.txt for a map written for these readers.",
    ]
    for agent, why in AI_AGENTS:
        lines += ["", f"# {why}", f"User-agent: {agent}", "Allow: /"]
    lines += ["", f"Sitemap: {base}/sitemap.xml", ""]

    txt = "\n".join(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt, encoding="utf-8")
    logger.info(f"  Saved website/robots.txt ({len(AI_AGENTS)} AI agents named)")
    return txt


# ── llms.txt ───────────────────────────────────────────────────

def build_llms_txt(
    pages: list[Page],
    *,
    site_url: str,
    author_name: str,
    repo_url: str,
    blog_url: str,
    data_years: list[int],
    totals: dict,
    build_date: str,
    out: Path,
) -> str:
    """Write /llms.txt: a plain-Markdown map of the site for language models.

    What this is, stated plainly so nobody later mistakes it for a standard:
    llms.txt is a *convention proposed* at llmstxt.org in 2024, not a
    specification any model provider has committed to reading. No major
    provider has announced that its crawler fetches it. So this file is a
    low-cost bet, not the mechanism that gets this work cited.

    What actually does that is the ordinary web infrastructure next to it:
    robots.txt admitting the AI crawlers, sitemap.xml listing the pages, the
    canonical URLs, and the schema.org Dataset entry. An assistant answering a
    question about MICCAI review scores today reaches this site through a
    normal search index, not through this file.

    The bet is still worth taking, because the file costs one function and
    pays off in a second way regardless of whether a crawler ever asks for it:
    it is the one page that states, in one screen, what the data covers, how
    it was produced, and what it cannot support. That is exactly what a model
    summarising this site gets wrong without it, and exactly what a human
    landing here for the first time wants too.

    The format is the proposed one: an H1, a blockquote summary, free prose,
    then H2 sections of `- [name](url): note` links, with `## Optional`
    reserved for what a short-on-context reader may skip.
    """
    base = site_url.rstrip("/")
    span = f"{min(data_years)} to {max(data_years)}"
    n_papers = totals["n_papers"]
    n_reviews = totals["n_reviews"]

    out_lines = [
        "# MICCAI Explorer",
        "",
        f"> Every accepted MICCAI paper from {span} ({n_papers:,} of them) "
        f"joined to its published peer reviews ({n_reviews:,}), its rebuttal "
        f"and meta-reviews, and the official oral and spotlight program. "
        f"Built and maintained by {author_name}.",
        "",
        "MICCAI (Medical Image Computing and Computer Assisted Intervention)",
        "is one of the few large conferences that publishes the full review",
        "record for every accepted paper: reviewer scores, confidence",
        "ratings, written reviews, author rebuttals, and meta-reviews. It",
        "separately publishes a program booklet naming which papers were",
        "given a talk. This project is the first to join those two sources,",
        "which is what makes questions like \"do oral papers actually score",
        "higher, once early acceptance is controlled for?\" answerable from",
        "public data.",
        "",
        "## What the data supports, and what it does not",
        "",
        "- Coverage is **accepted papers only**. MICCAI does not publish",
        "  reviews for rejected submissions, so nothing here describes what",
        "  gets rejected, and acceptance-rate figures come from MICCAI's own",
        "  reported submission counts rather than from this data.",
        "- Review scores are **not comparable across years without",
        "  normalization**: the scale was 1-9 in 2021, 1-8 in 2022 and 2023,",
        "  and 1-6 from 2024. Cross-year comparisons here are expressed as a",
        "  percentage of that year's scale, or with Cliff's delta, which is",
        "  ordinal and scale-free.",
        "- Subject-area counts are **not comparable across the 2024/2025",
        "  boundary**. MICCAI replaced its taxonomy in 2025, from 31 ad-hoc",
        "  facets to six required ones, so the median paper carries 5 tags",
        "  against 3 the year before. An area can appear to grow fourfold",
        "  with no research having moved.",
        "- Early acceptance is a **confounder, not a footnote**. Orals are",
        "  drawn from accepted papers and early accepts are decided before",
        "  rebuttal, so any oral-versus-poster score gap is reported within",
        "  early and non-early strata as well as pooled.",
        "",
        "## Pages",
        "",
    ]
    for pg in pages:
        out_lines.append(f"- [{pg.social_title}]({base}{pg.path}): "
                         f"{pg.description}")

    out_lines += [
        "",
        "## Data",
        "",
        f"- [papers.json]({base}/papers.json): the full browse index as one "
        f"JSON file. Every paper with its year, authors, subject areas, mean "
        f"reviewer score, individual scores, early-acceptance flag, "
        f"presentation type, and links to PDF, DOI, code, and datasets. "
        f"Subject-area names are listed once under `areas` and referenced by "
        f"position.",
    ]
    if repo_url:
        out_lines.append(
            f"- [Source repository]({repo_url}): the scrapers, the parsers "
            f"for each era of MICCAI's page format, the analysis, and the "
            f"site build. The raw per-year JSON and the committed embeddings "
            f"are in `data/`."
        )
    out_lines += [
        "",
        "## Optional",
        "",
        f"- [How it was built]({base}/about.html): sources, scraping, "
        f"SPECTER2 embeddings, clustering, and the statistics used.",
    ]
    if blog_url:
        out_lines.append(
            f"- [Write-up]({blog_url}): the longer narrative account of the "
            f"project and what it found."
        )
    out_lines += [
        "",
        f"Last built {build_date}. MICCAI papers and reviews remain the "
        f"copyright of their respective authors; this analysis is by "
        f"{author_name}.",
        "",
    ]

    txt = "\n".join(out_lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt, encoding="utf-8")
    logger.info(f"  Saved website/llms.txt ({len(txt)} bytes)")
    return txt
