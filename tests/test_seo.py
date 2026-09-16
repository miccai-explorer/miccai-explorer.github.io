"""Guards on the crawler-facing output: sitemap.xml, robots.txt, llms.txt,
the canonical URLs, and the structured data.

Everything here is invisible. A page with a canonical URL pointing at the wrong
address, a sitemap that has stopped listing a year, or a JSON-LD block with a
syntax error all render identically to the correct thing in a browser; the only
reader who ever sees the difference is a crawler, and it does not report back.
That is the same shape of problem as the cluster-label mismatch of 2026-09:
correct-looking output, wrong content, no signal. So these are checked against the built files on disk rather
than against the page table that produced them, which would only prove the
table agrees with itself.

Needs a built website/. `built_site` builds one into a temp directory if the
repository does not already have one, so the suite stays runnable from a fresh
clone and in CI, where no build has happened yet.
"""

import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import seo  # noqa: E402

SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

# The origin config.yaml declares. Hardcoded rather than read from the config,
# deliberately: reading it would make this test pass for any value, including
# the empty string and a leftover localhost, which are exactly the two ways a
# canonical URL goes wrong. See feedback on measuring independently of the name.
SITE_URL = "https://miccai-explorer.github.io"

# Pages that must be in the sitemap. Written out rather than globbed, for the
# same reason.
EXPECTED_PATHS = {
    "/",
    "/orals.html",
    "/papers.html",
    "/about.html",
    "/year/2021.html",
    "/year/2022.html",
    "/year/2023.html",
    "/year/2024.html",
    "/year/2025.html",
    "/year/2026.html",
}


@pytest.fixture(scope="module")
def site() -> Path:
    """A built website/ directory, built on demand if there is not one."""
    existing = ROOT / "website"
    if (existing / "sitemap.xml").exists():
        return existing
    env = {**os.environ, "PYTHONPATH": str(ROOT / "analysis")}
    r = subprocess.run(
        [sys.executable, "analysis/build_site.py"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    if r.returncode != 0 or not (existing / "sitemap.xml").exists():
        pytest.skip(f"could not build the site: {r.stderr[-400:]}")
    return existing


@pytest.fixture(scope="module")
def pages(site) -> dict[str, str]:
    """Every built HTML page, keyed by its site-relative path."""
    out = {}
    for f in list(site.glob("*.html")) + list(site.glob("year/*.html")):
        rel = "/" + str(f.relative_to(site))
        out["/" if rel == "/index.html" else rel] = f.read_text(encoding="utf-8")
    return out


# ── sitemap ────────────────────────────────────────────────────

def test_sitemap_is_wellformed_and_uses_the_sitemaps_org_namespace(site):
    root = ET.parse(site / "sitemap.xml").getroot()
    assert root.tag == f"{SITEMAP_NS}urlset"


def test_sitemap_lists_exactly_the_pages_that_were_built(site, pages):
    root = ET.parse(site / "sitemap.xml").getroot()
    listed = {
        el.text[len(SITE_URL):] or "/"
        for el in root.iter(f"{SITEMAP_NS}loc")
    }
    assert listed == EXPECTED_PATHS
    # And the built pages agree, so adding a page without a sitemap entry
    # (or the reverse) fails here rather than going unnoticed for months.
    assert listed == set(pages)


def test_every_sitemap_url_is_absolute_and_resolves_to_a_real_file(site):
    root = ET.parse(site / "sitemap.xml").getroot()
    for el in root.iter(f"{SITEMAP_NS}loc"):
        url = el.text
        assert url.startswith(f"{SITE_URL}/"), url
        assert "//" not in url[len("https://"):], f"doubled slash in {url}"
        rel = url[len(SITE_URL) + 1:] or "index.html"
        assert (site / rel).exists(), f"{url} points at nothing"


def test_every_sitemap_entry_has_a_w3c_date_lastmod(site):
    root = ET.parse(site / "sitemap.xml").getroot()
    mods = [el.text for el in root.iter(f"{SITEMAP_NS}lastmod")]
    assert len(mods) == len(EXPECTED_PATHS)
    for m in mods:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", m), m


# ── robots.txt ─────────────────────────────────────────────────

def test_robots_allows_everything_and_disallows_nothing(site):
    txt = (site / "robots.txt").read_text()
    body = [ln.strip() for ln in txt.splitlines()
            if ln.strip() and not ln.startswith("#")]
    assert "User-agent: *" in body
    assert "Allow: /" in body
    assert not [ln for ln in body if ln.lower().startswith("disallow")]


def test_robots_points_at_the_sitemap_with_an_absolute_url(site):
    txt = (site / "robots.txt").read_text()
    assert f"Sitemap: {SITE_URL}/sitemap.xml" in txt


def test_robots_does_not_block_the_chart_files(site):
    """They are the iframe content of every analysis page.

    Blocked, a renderer would see those pages as nearly empty, and the noindex
    tag inside each file could never be read in order to be obeyed.
    """
    directives = [
        ln.strip() for ln in (site / "robots.txt").read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    # Comments are excluded on purpose: robots.txt explains the /charts/
    # decision in prose, and matching that would make this test pass on the
    # explanation rather than on the rule.
    assert not [ln for ln in directives if "/charts" in ln], directives


def test_robots_names_the_ai_crawlers(site):
    txt = (site / "robots.txt").read_text()
    for agent, _ in seo.AI_AGENTS:
        assert f"User-agent: {agent}" in txt, agent


# ── per-page head ──────────────────────────────────────────────

def test_every_page_has_a_canonical_url_matching_its_own_path(pages):
    for path, html in pages.items():
        m = re.search(r'<link rel="canonical" href="([^"]+)"', html)
        assert m, f"{path} has no canonical link"
        assert m.group(1) == f"{SITE_URL}{path}", path


def test_every_page_has_a_unique_non_empty_title_and_description(pages):
    titles, descs = {}, {}
    for path, html in pages.items():
        t = re.search(r"<title>(.*?)</title>", html, re.S)
        d = re.search(r'<meta name="description" content="([^"]*)"', html)
        assert t and t.group(1).strip(), f"{path} has no title"
        assert d and d.group(1).strip(), f"{path} has no description"
        assert t.group(1) not in titles, f"{path} repeats {titles.get(t.group(1))}'s title"
        assert d.group(1) not in descs, f"{path} repeats {descs.get(d.group(1))}'s description"
        titles[t.group(1)] = path
        descs[d.group(1)] = path


def test_descriptions_fit_a_search_snippet(pages):
    for path, html in pages.items():
        d = re.search(r'<meta name="description" content="([^"]*)"', html)
        assert len(d.group(1)) <= seo.DESC_MAX, (
            f"{path}: {len(d.group(1))} chars, over {seo.DESC_MAX}"
        )


def test_social_image_is_an_absolute_url_and_the_file_exists(site, pages):
    for path, html in pages.items():
        m = re.search(r'<meta property="og:image" content="([^"]+)"', html)
        assert m, f"{path} has no og:image"
        url = m.group(1)
        # Relative og:image is silently dropped by every link-preview
        # fetcher, which has no page to resolve it against.
        assert url.startswith("https://"), f"{path}: {url}"
        assert (site / url[len(SITE_URL) + 1:]).exists(), url


def test_every_page_names_the_author(pages):
    for path, html in pages.items():
        assert '<meta name="author" content="Kumar Abhishek">' in html, path


# ── structured data ────────────────────────────────────────────

def _jsonld(html: str) -> dict:
    m = re.search(
        r'<script type="application/ld\+json">(.*?)</script>', html, re.S
    )
    assert m, "no JSON-LD block"
    return json.loads(m.group(1))


def test_json_ld_parses_on_every_page(pages):
    for path, html in pages.items():
        graph = _jsonld(html)
        assert {n["@type"] for n in graph["@graph"]} == {
            "WebSite", "Person", "Dataset"
        }, path


def test_json_ld_cannot_break_out_of_its_script_tag(pages):
    """A raw "<" in the JSON would let a value close the <script> early."""
    for path, html in pages.items():
        m = re.search(
            r'<script type="application/ld\+json">(.*?)</script>', html, re.S
        )
        assert "<" not in m.group(1), path


def test_dataset_entry_carries_what_dataset_search_needs(pages):
    graph = _jsonld(pages["/"])
    ds = next(n for n in graph["@graph"] if n["@type"] == "Dataset")
    for field in ("name", "description", "creator", "license", "keywords",
                  "temporalCoverage", "distribution"):
        assert ds.get(field), f"Dataset is missing {field}"
    assert "MICCAI" in ds["keywords"]
    person = next(n for n in graph["@graph"] if n["@type"] == "Person")
    assert person["name"] == "Kumar Abhishek"
    assert ds["creator"]["@id"] == person["@id"]


# ── llms.txt ───────────────────────────────────────────────────

def test_llms_txt_follows_the_proposed_shape(site):
    txt = (site / "llms.txt").read_text()
    lines = txt.splitlines()
    assert lines[0] == "# MICCAI Explorer"
    assert any(ln.startswith("> ") for ln in lines[:5]), "no summary blockquote"
    assert "## Optional" in txt


def test_llms_txt_links_every_page_with_an_absolute_url(site, pages):
    txt = (site / "llms.txt").read_text()
    linked = set(re.findall(rf"\]\({re.escape(SITE_URL)}([^)]*)\)", txt))
    for path in pages:
        assert path in linked, f"llms.txt does not link {path}"


def test_llms_txt_states_the_limits_not_just_the_contents(site):
    """The reason this file exists: what a model gets wrong without it.

    Each of these is a caveat the site makes in prose and a summariser drops.
    If one is edited out of llms.txt, it should be a deliberate act.
    """
    txt = (site / "llms.txt").read_text().lower()
    for claim in ("accepted papers only", "not comparable", "confounder"):
        assert claim in txt, claim


# ── chart fragments ────────────────────────────────────────────

def test_chart_files_are_not_indexable(site):
    charts = sorted((site / "charts").glob("*.html"))
    if not charts:
        pytest.skip("no charts built; run analysis/build_charts.py")
    missing = [
        f.name for f in charts
        if 'name="robots" content="noindex' not in f.read_text(encoding="utf-8")
    ]
    assert not missing, f"{len(missing)} chart files are indexable: {missing[:5]}"


def test_chart_files_are_not_in_the_sitemap(site):
    root = ET.parse(site / "sitemap.xml").getroot()
    assert not [
        el.text for el in root.iter(f"{SITEMAP_NS}loc") if "/charts/" in el.text
    ]
