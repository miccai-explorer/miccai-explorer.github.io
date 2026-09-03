# MICCAI Explorer: Project Guide

## What This Is

An interactive web explorer for MICCAI (Medical Image Computing and Computer Assisted
Intervention) papers 2021-2025, including their full peer-review data. No tool like this
exists publicly. The pipeline scrapes MICCAI's open-access site, builds semantic embeddings,
runs analysis, and produces a static multi-page website deployed to GitHub Pages.

**Live URL**: `https://miccai-explorer.github.io/`
**Repo**: `https://github.com/miccai-explorer/miccai-explorer.github.io` (public;
the pipeline is developed in a separate private repo and published from it)
**BASE_URL** (prefix on every generated href): `""` (empty). This is a GitHub
**organization site**, served at the root of its domain, so hrefs are plain
root-relative paths such as `/assets/style.css`. Set it to `"/<repo>"` only if the
site ever moves to a project site at `https://<owner>.github.io/<repo>/`.

> **Before building any part of the website** (templates, CSS, chart HTML files, or
> `analysis/build_site.py`), read `DESIGN.md` in full. It specifies every visual
> decision: color tokens, typography, layout classes, stat strip, chart containers,
> logo handling, and Plotly theme. All website code must follow it exactly.

---

## Repository Layout

```
miccai-explorer.github.io/
├── README.md                    ← human entry point: what this is, how to run it
├── CONTRIBUTING.md              ← setup, reproducing each stage, how to contribute
├── CLAUDE.md                    ← this file: full technical reference
├── AGENTS.md                    ← entry point for coding agents; points here
├── DESIGN.md                    ← the visual system (read before any website code)
├── REPRODUCIBILITY.md           ← what is committed vs regenerated, and known gaps
├── requirements.txt
├── .gitignore
├── config.yaml                  ← THE config; single source of truth for the whole
│                                   pipeline. scrape.py, build_charts.py and
│                                   build_site.py all read this one file. It holds
│                                   the per-year blocks (URLs, review_scale_max,
│                                   coauthorship_network_min_papers, parser rules)
│                                   plus the site_settings, chart_settings and
│                                   oral_schedules blocks. A second copy at
│                                   scraper/config.yaml was read by no code and
│                                   had drifted; deleted 2026-08-30.
├── num_papers_submitted.yaml    ← {year: papers submitted}; read by build_charts.py
│                                   (Submissions, Acceptance Rate, Early Accepted
│                                   panels) and build_site.py (pct_early_subs)
├── logo_colors.yaml             ← principal colors of each year's conference logo
│                                   (first 3 per year = primaries; rest are fallbacks)
├── static/                      ← tracked source for website assets;
│   ├── style.css                   build_site.py copies static/ → website/assets/
│   ├── logos/miccai_YYYY.png       (website/ is gitignored, so assets must live here).
│   │                               PNG here, WebP on the site; see "Year logos"
│   ├── papers.css               ← All Papers page only; style.css stays untouched
│   └── papers.js                ← All Papers page behaviour (no framework, no Plotly)
├── scraper/
│   ├── config.yaml              ← authoritative per-year settings; no hardcoded URLs elsewhere
│   ├── scrape.py                ← entry point: python scraper/scrape.py --year YYYY
│   ├── audit_review_fields.py   ← finds form questions the parsers drop; run when
│   │                               adding a year (see 2026-08-22-review-parser-fixes.md)
│   ├── checkpoint.py            ← save/load/resume checkpoint state
│   └── parsers/
│       ├── __init__.py
│       ├── base.py              ← shared fetch + BeautifulSoup helpers
│       ├── era_2024_2025.py     ← parser for 2024 and 2025 (confirmed HTML structure)
│       ├── era_2022_2023.py     ← parser for 2022 and 2023 (verify HTML before coding)
│       └── era_2021.py          ← parser for 2021 (different domain, verify HTML before coding)
├── data/
│   ├── manually_downloaded/     ← collected by hand; not scrapeable
│   │   └── OralSchedules/YYYY.pdf       ← official program books (oral/spotlight
│   │                                       schedule pages); source for extract_orals.py
│   ├── oral_title_overrides.yaml ← committed; hand-written title → paper_id fixes
│   │                                for anything match_orals.py cannot resolve
│   ├── raw/                     ← committed to git; one JSON per year
│   │   ├── miccai_2025.json
│   │   ├── miccai_2024.json
│   │   ├── orals_YYYY.json              ← committed; extracted oral/spotlight program
│   │   └── ...
│   ├── processed/
│   │   ├── miccai_all.json              ← committed; all years merged, normalized, + umap/cluster fields
│   │   ├── orals.json                   ← committed; paper_id → type/session + match report
│   │   ├── embeddings_specter2.npz      ← committed; SPECTER2 embeddings (~30 MB float32)
│   │   ├── embeddings_{model}.npz       ← one per model (embeddings, cluster_ids, paper_ids)
│   │   ├── proj_{model}_{proj}.npy      ← 2D coords per model×projection (~25 KB each)
│   │   ├── cluster_labels.json          ← copy of cluster_labels_{active}.json; used by build scripts
│   │   ├── cluster_labels_{model}.json  ← one per model; human-edited names (20 clusters each)
│   │   ├── active_model.txt             ← active model slug, e.g. "specter2"
│   │   └── active_proj.txt              ← active projection slug, e.g. "umap-tight"
│   └── raw/checkpoints/         ← NOT committed (in .gitignore); scraper resume state
├── analysis/
│   ├── normalize.py             ← merge raw JSONs → miccai_all.json
│   ├── embed.py                 ← multi-model; --model required; saves embeddings_{model}.npz
│   ├── use_model.py             ← switch active model without re-running embeddings
│   ├── build_charts.py          ← all Plotly chart HTML files → website/charts/
│   ├── build_site.py            ← Jinja2 pages → website/index.html + website/year/*.html
│   ├── papers_index.py          ← paper list (in memory) → website/papers.json;
│   │                               stdlib only, called from build_site.py
│   ├── logo_assets.py           ← static/logos/*.png → website/assets/*.webp,
│   │                               resized; called from build_site.py
│   ├── extract_orals.py         ← program-book PDFs → data/raw/orals_YYYY.json
│   ├── match_orals.py           ← join orals to papers by title → data/processed/orals.json
│   ├── oral_stats.py            ← type join + statistics (Cliff's δ, Wilson, bootstrap)
│   └── build_oral_charts.py     ← orals_*.html charts; called by build_charts.py
├── tests/                       ← python -m pytest tests/ -q
│   ├── test_oral_stats.py       ← known-answer tests for the statistics
│   ├── test_orals_data.py       ← golden counts + re-parse of the source PDFs
│   ├── test_papers_index.py     ← golden counts on website/papers.json
│   └── test_logo_assets.py      ← the logo conversion actually ran (see below)
├── templates/
│   ├── base.html                ← shared nav bar + footer
│   ├── index.html               ← cross-year trends page
│   ├── orals.html               ← Orals & Spotlights page (skipped without orals.json)
│   ├── papers.html              ← All Papers browse page (always built)
│   ├── about.html               ← About page (always built; text only, no data)
│   └── year.html                ← per-year analysis page
├── website/                     ← GENERATED: never hand-edit; not committed to main branch
│   ├── index.html
│   ├── orals.html
│   ├── papers.html
│   ├── about.html
│   ├── papers.json              ← browse index, ~2.4 MB (515 KB gzipped)
│   ├── year/
│   │   └── YYYY.html
│   ├── charts/                  ← standalone Plotly HTML files (embedded as iframes)
│   └── assets/                  ← copied from static/ by build_site.py (do not edit here)
│       └── style.css
└── .github/
    └── workflows/
        └── deploy.yml           ← build + push website/ to gh-pages branch
```

---

## Pipeline Commands

Run from repo root. Each step is independently re-runnable.

```bash
# 1. Scrape one year at a time (start with 2025, then 2024)
python scraper/scrape.py --year 2025
python scraper/scrape.py --year 2024
python scraper/scrape.py --year 2023   # implement era_2022_2023.py first; verify HTML
python scraper/scrape.py --year 2022
python scraper/scrape.py --year 2021   # implement era_2021.py first; verify HTML

# 2. Merge + normalize all scraped years
python analysis/normalize.py

# 3. Build embeddings, UMAP, KMeans (SPECTER2 is the active/recommended model)
python analysis/embed.py --model specter2
# Edit data/processed/cluster_labels_specter2.json with human-readable names, then run:
python analysis/use_model.py specter2   # patches miccai_all.json + copies cluster_labels.json

# 3b. (Optional) Switch to a different pre-computed model without re-running embed.py:
python analysis/use_model.py bge-large  # or gte-large, e5-large, all-mpnet

# 3c. Oral / spotlight program (independent of steps 2-3; only needs miccai_all.json).
#     Source PDFs are collected by hand into data/manually_downloaded/OralSchedules/.
python analysis/extract_orals.py          # PDFs → data/raw/orals_YYYY.json
python analysis/match_orals.py            # join by title → data/processed/orals.json
#     match_orals.py exits nonzero if any presentation is unmatched: never ignore it,
#     a dropped talk silently biases every statistic on the orals page. Run it again
#     with --report to write a stub, then fill in data/oral_title_overrides.yaml.

# 4. build_charts.py patches cluster labels in-memory from cluster_labels.json: and,
#    the same way, presentation types from orals.json. No manual miccai_all.json update
#    is needed after editing cluster names or re-running the oral pipeline:
python analysis/build_charts.py

# 5. Generate HTML pages
python analysis/build_site.py

# 6. Tests (statistics + oral extraction golden counts)
python -m pytest tests/ -q

# To preview locally: serve website/ itself, because BASE_URL is empty and every
# generated href is root-relative, so the directory served has to be the site root.
python -m http.server 8000 --directory website
# Open http://localhost:8000/
```

**Adding a future year (e.g., 2026):**
1. Add a block under `years:` in the root `config.yaml`
2. Drop the conference logo into `static/logos/miccai_2026.png` and add a
   `YEAR_META_RAW` entry in `build_site.py` (city, colors). Any size of PNG will
   do; the build resizes and converts it. See "Year logos" below.
3. Run `python scraper/scrape.py --year 2026`
4. Run steps 2-5 above

---

## Data Schema

### Raw JSON: `data/raw/miccai_YYYY.json`

Array of objects. Store EVERYTHING: all review text, all fields. The website uses only
structured fields, but the full text enables future NLP analysis. Do not discard data.

```jsonc
{
  "paper_id": "miccai-2025-Paper0308",    // f"miccai-{year}-Paper{num:04d}"
  "year": 2025,
  "title": "µ2 Tokenizer: ...",
  "authors": ["Li, Siyou", "Qin, Pengyao"],   // "Last, First" order as on site
  "abstract": "Automated radiology report generation...",  // LaTeX stripped if possible
  "subject_areas": ["Machine Learning -> Foundation Models", "Modalities -> CT / X-Ray"],
  "url": "https://papers.miccai.org/miccai-2025/0001-Paper0308.html",
  "pdf_url": "https://papers.miccai.org/miccai-2025/paper/0308_paper.pdf",
  "doi": "https://doi.org/10.1007/978-3-032-04971-1_1",
  "shared_it_url": "https://rdcu.be/eHwS7",
  "supp_url": null,                        // or URL string if supplementary exists
  "code_url": "https://github.com/Siyou-Li/u2Tokenizer",
  "has_code": true,
  "dataset_urls": ["https://huggingface.co/datasets/SiyouLi/CT-RATE-Chinese"],
  "lncs_volume": "LNCS 15964",            // from BibTeX volume field; stored as-is; field already includes "LNCS" prefix, do NOT prepend
  "bibtex_key": "LiSiy_µ2_MICCAI2025",
  "bibtex_raw": "@InProceedings{...}",
  "early_accepted": false,                // detection is year-specific; see "Notes on early_accepted" below; do NOT use review-1 presence as signal
  "rebuttal_provided": true,              // false if the authorFeedback-id block is
                                          // absent, empty, or the literal "N/A".
                                          // Every accepted paper HAS the section, so
                                          // its presence proves nothing; only the text
                                          // does. All eras must agree here: see the
                                          // note under Era 2022-2023 below.
  "reviews": [
    {
      "reviewer_num": 1,
      "score_raw": 2,
      "score_max": 6,                     // from config: review_scale_max
      "score_normalized": 0.20,           // (score_raw - 1) / (score_max - 1)
      "recommendation_label": "Reject; should be rejected, independent of rebuttal",  // full text from score field, parenthetical stripped
      "post_rebuttal_label": "Reject",    // text from post-rebuttal field (short form: "Accept", "Reject", etc.)
      "post_rebuttal_score_raw": null,    // integer if year has score in post-rebuttal, else null
      "score_changed": false,             // true if verdict keyword before em dash differs (e.g. "Weak Accept" → "Accept" is NOT changed; "Reject" → "Accept" IS)
      "confidence_raw": 4,               // number from "Reviewer confidence" field
      "confidence_label": "Very confident",
      "clarity_label": "Poor",           // from "Please rate the clarity" field
      "text_contribution": "...",
      "text_strengths": "...",
      "text_weaknesses": "...",
      "text_detailed_comments": "...",
      "text_reproducibility": "...",
      "text_post_rebuttal_justification": "...",
      // 2022 and 2021 only (confirmed absent in 2023):
      "review_stack_rank": null,
      "review_stack_size": null
    }
  ],
  "num_reviews": 3,
  "avg_score_raw": 3.67,
  "avg_score_normalized": 0.47,          // mean of score_normalized across reviewers
  "score_range": 3,                      // max(score_raw) - min(score_raw): controversy metric
  "meta_reviews": [
    {
      "meta_reviewer_num": 1,
      "recommendation": "Accept",
      "text": "..."
    }
  ],
  "author_feedback_text": "We thank all reviewers..."
}
```

**Reviewer numbering is not reliable in any era.** MICCAI numbers reviewers
from the original assignment, so when a reviewer is replaced the rest keep their
numbers: a paper can carry `review-3`, `review-4`, `review-5` and no `review-1`,
or run `1, 2, 4`. Measured over the committed data: 82 papers in 2022 and 108 in
2023 do not start at `review-1`, and a further 100 and 137 start at 1 with a gap.
2021, 2024 and 2025 happen to be regular, which is exactly why this is easy to
get wrong. **Both** era parsers now collect every `<h3 id="review-N">` with
`soup.find_all("h3", id=re.compile(r'^review-\d+$'))` and sort by the number;
neither counts upward from 1. `era_2024_2025` also used to treat a missing
`review-1` as proof of early acceptance, which would have stored zero reviews
and set the wrong flag with no error; that branch is gone, and early acceptance
is decided only where it should be, in the meta-review section. Fixed
2026-08-30; covered by `tests/test_parsers.py`.

**Notes on `early_accepted`**: ~30% of accepted MICCAI papers are early-accepted (~9-13% of
all submissions). Early-accepted papers still appear on the site with full review sections.
Detection varies by year; do NOT use the presence/absence of `review-1` as the signal:

- **2024**: `<h1 id="metareview-id">` is followed immediately by
  `<p>Meta-review not available, early accepted paper.</p>` (no `<h2>` meta-review blocks).
  **Note**: Official MICCAI 2024 report says 287 early-accepted out of 857 total; we detect
  286 of 856. The 1-paper discrepancy is a withdrawn paper; accepted early but not published,
  absent from papers.miccai.org entirely. Verified by full live-HTML sweep (856/856 pages).
- **2025**: Exactly one `<h2 id="meta-review-1">` block exists, whose recommendation field
  contains `"Provisional Accept"`.

Both year patterns are implemented in `_parse_meta_reviews` in `era_2024_2025.py`.
- **2022/2023**: two signals (see Era 2022-2023 section below). Implemented in `era_2022_2023.py`.
- **2021**: verify actual HTML before implementing.

### Processed JSON: `data/processed/miccai_all.json`

Identical schema plus these fields added by `embed.py`:
```jsonc
"umap_x": 3.14,
"umap_y": -2.71,
"cluster_id": 7,
"cluster_label": "Deformable Image Registration"
```

---

## Per-Year HTML Parsing

### Era 2024-2025 (CONFIRMED from source inspection of both years)

**Individual paper page structure:**
```
<article class="container-post">
  <div class="post-title">
    <h1><span style="color:#1040a7"><b>TITLE TEXT</b></span></h1>
  </div>
  <div class="post-author print-post-author">   ← IGNORE: placeholder "Kitty K. Wong"
  <div class="post-tags">
    <h2>Author(s):</h2>
    <a class="post-category" href="/miccai-YYYY/tags#Last, First">Last, First</a> |
    ...                                         ← REAL author list
  </div>
  <!-- LARGE COMMENTED-OUT BLOCK containing subject areas; IGNORE -->
  <h1 id="abstract-id">Abstract</h1>
  <p>Abstract text. LaTeX may appear as $...$. May end with \url{...}.</p>
  <h1 id="link-id">Links to Paper and Supplementary Materials</h1>
  <p>Main Paper (Open Access Version): <a href="...pdf">...</a></p>
  <p>SharedIt Link: <a href="...">...</a></p>
  <p>SpringerLink (DOI): <a href="...">...</a></p>
  <p>Supplementary Material: [<a href="...">url</a> OR "Not Submitted"]</p>
  <h1 id="code-id">Link to the Code Repository</h1>
  <p><a href="code_url">code_url</a><br/><br/></p>   ← absent if no code
  <h1 id="dataset-id">Link to the Dataset(s)</h1>
  <p>[zero or more <a href="dataset_url"> links]</p>
  <h1 id="bibtex-id">BibTex</h1>
  <pre><code class="language-{verbatim}">@InProceedings{...}</code></pre>
  <hr/>
  <h1 id="review-id">Reviews</h1>
  <h3 id="review-1">Review #1</h3>
  <ul>
    <li><strong>FIELD LABEL</strong>
        <blockquote>ANSWER (can be <p>, <ol>, <ul>)</blockquote></li>
    ...
  </ul>
  <h3 id="review-2">Review #2</h3>  ...
  <h3 id="review-3">Review #3</h3>  ...  (usually 3 reviewers)
  <hr/>
  <h1 id="authorFeedback-id">Author Feedback</h1>
  <blockquote><p>...</p>...</blockquote>
  <hr/>
  <h1 id="metareview-id">Meta-Review</h1>
  <!-- EARLY ACCEPTED (2024): no <h2> blocks; instead: -->
  <!-- <p>Meta-review not available, early accepted paper.</p> -->
  <!-- EARLY ACCEPTED (2025): exactly one <h2> block with "Provisional Accept" -->
  <!-- STANDARD (both years): two or three <h2> blocks -->
  <h2 id="meta-review-1">Meta-review #1</h2>
  <ul>
    <li><strong>Your recommendation</strong><blockquote><p>Invite for Rebuttal</p></blockquote></li>
    <li><strong>If your recommendation is "Provisional Reject"...</strong><blockquote><p>N/A</p></blockquote></li>
    <li><strong>After you have reviewed the rebuttal and updated reviews, please provide your recommendation...</strong>
        <blockquote><p>Accept</p></blockquote></li>   ← THIS is the final recommendation to store
    <li><strong>Please justify your recommendation...</strong><blockquote><p>justification text or N/A</p></blockquote></li>
  </ul>
  <h2 id="meta-review-2">Meta-review #2</h2>
  <ul>...</ul>
</article>

Meta-review field matching; match by substring in `<strong>` text:
- `"after you have reviewed the rebuttal"` → `recommendation` (preferred/final)
- `"your recommendation"` (exact start) → `recommendation` fallback if no post-rebuttal field
- `"please justify your recommendation"` → `text` (skip if value is "N/A")
```

**Field label matching**; match by substring in `<strong>` text:

| Field to extract | Substring to match in `<strong>` |
|---|---|
| Contribution | `"describe the contribution"` |
| Strengths | `"major strengths"` OR `"main strengths"` |
| Weaknesses | `"major weaknesses"` OR `"main weaknesses"` |
| Clarity | `"rate the clarity"` |
| Reproducibility | `"reproducibility"`; **appends, does not assign** (see below) |
| Additional comments | `"additional comments"` |
| Detailed comments | `"detailed and constructive comments"` (2021-2024; absent in 2025) |
| Score | `"scale of 1-6"` (2024/2025) |
| Justification | `"justify your recommendation"` |
| Reviewer confidence | `"reviewer confidence"` (case-insensitive) |
| Post-rebuttal opinion | `"[Post rebuttal]"` and `"final opinion"` OR `"overall opinion"` |
| Post-rebuttal justification | `"[Post rebuttal]"` and (`"justify your final"` OR `"justify your decision"`) |

**Three traps in this table, all found on 2026-08-22 after they had silently
corrupted the data for months.** The forms are not the same every year, and a
label that matches nothing is dropped with no error:

1. **The chain order matters.** `"reproducibility"` is tested before
   `"additional comments"`, and 2024 asks *"Do you have any additional comments
   regarding the paper's reproducibility?"*, whose label contains both. With a
   plain assignment the follow-up answer replaced the real one; 1,352 of 2,623
   reviews in 2024 ended up storing just `"N/A"`. That branch now **appends**,
   so both answers are kept in page order. Do not change it back.
2. **`"detailed and constructive comments"` matched nothing at all** until
   2026-08-22, so the main free-text box for the authors was dropped for every
   review in 2021 through 2024 (8,254 reviews). 2025 replaced that question with
   the "additional comments" one, which is why 2025 was unaffected. This field
   is in `REVIEW_TEXT_FIELDS`, so losing it made Avg Review Length understate
   every year before 2025.
3. **The post-rebuttal justification label changed wording.** 2025 says
   *"justify your final decision from above"*; 2022 through 2024 say *"justify
   your decision"*. Matching only `"justify your final"` dropped it for 6,626
   reviews, which read as reviewers declining to explain rather than as a miss.

Before trusting this table for a new year, run
`python scraper/audit_review_fields.py --all`: it fetches live pages and reports any
form question whose label matches no branch. The committed data cannot tell you
this, because a field that was never captured leaves no trace in it.

**Score extraction**; 2024/2025 format: `"(N) Label text"` OR `"Label text (N)"`:
```python
import re
m = re.search(r'\((\d+)\)', blockquote_text)
score_raw = int(m.group(1)) if m else None
```

**2024 post-rebuttal score**: Present; same format, extract with same regex.
**2025 post-rebuttal score**: Absent; just a text label like `"Accept"` or `"Reject"`.

**Confidence extraction**: Same `\((\d+)\)` regex on text like `"Very confident (4)"`.

**Subject areas**: **NOT in live HTML** (in a `<!-- comment -->` block). Source them
exclusively from the categories page (see below).

**Authors**: Extract from `<a class="post-category">` tags within `<div class="post-tags">`.
Use `.get_text(strip=True)` on each anchor. Filter out any anchor whose href contains
`/categories#` (those are subject area links in the commented section, but since those are
commented out, you won't see them in live parse; include the filter as a safety measure).

**Abstract**: Find `<h1 id="abstract-id">` then take the next `<p>` sibling.
Strip LaTeX-style `\url{...}` patterns and `$...$` math if they appear.

---

### Era 2022-2023 (CONFIRMED: verified live HTML for both years, implemented in `era_2022_2023.py`)

**HTML structure is nearly identical to 2024/2025**; same `<h1>` section headings, same `id=`
values. Implemented and fully scraped (573 papers 2022, 730 papers 2023, 0 errors).

**Confirmed page structure** (both years, verified against live HTML):
```
<h1 id="">TITLE TEXT</h1>          ← plain <h1> with no id; no div.post-title wrapper
<h1 id="author-id">Authors</h1>
  → authors in <a class="post-category"> anchors (same div.post-tags pattern as 2024/2025)
<h1 id="abstract-id">Abstract</h1>
<h1 id="link-id">Link to paper</h1>
  → DOI and SharedIt links; NO explicit PDF link (unlike 2024/2025)
<h1 id="code-id">Link to the code repository</h1>
<h1 id="dataset-id">Link to the dataset(s)</h1>
                                   ← NO BibTeX section (unlike 2024/2025)
<h1 id="review-id">Reviews</h1>
<h3 id="review-1">Review #1</h3>  ← numbering does NOT always start at 1; see warning below
  <ul> ... </ul>
<h3 id="review-2">Review #2</h3>
<h1 id="metareview-id">Primary Meta-Review</h1>
<h2 id="meta-review--1-primary">Meta-review # 1 (Primary)</h2>  ← NOTE: double dash in id
  <ul> ... </ul>            ← OR meta-review content can appear directly in a <ul> here with
                                NO <h2> wrapper at all; see warning below
<h1 id="authorFeedback-id">Author Feedback</h1>
  <blockquote>...</blockquote>     ← "N/A" if author chose not to rebut (does NOT imply early accept)
<h1 id="postrebuttal-id">Post-rebuttal Meta-Reviews</h1>  ← ABSENCE of this h1 is the early_accepted signal
<h2 id="meta-review-2">Meta-review #2</h2>
<h2 id="meta-review-3">Meta-review #3</h2>
```

**Critical parsing pitfalls (found via live-HTML verification after the initial scrape produced
wrong `early_accepted` counts; see detection section below):**

1. **Review numbering has gaps and does not always start at `review-1`.** Reviewer reassignment
   can leave a paper with e.g. only `review-3`, `review-4`, `review-5` (no `review-1`/`review-2`).
   The original parser scanned sequentially from 1 and stopped at the first miss, which silently
   dropped all review content for these papers and miscounted them as having zero reviews. Fix:
   find all `<h3 id="review-N">` via `soup.find_all("h3", id=re.compile(r'^review-\d+$'))`, sort
   by the numeric suffix, and parse whatever is found; never assume `review-1` exists.

2. **Meta-review content can live directly in a `<ul>` with no `<h2>` wrapper.** Many papers have
   `<h1 id="metareview-id">` followed immediately by a `<ul>` containing a single filled
   `<li><strong>...</strong><blockquote>real text</blockquote></li>`; there is no
   `<h2 id="meta-review-*">` at all. The original parser only searched for `<h2>`/`<h3>` elements
   matching `meta-review-`, so it missed this structure entirely and treated these (real, filled)
   meta-reviews as empty. Fix: when no `<h2>` is found, check the `<ul>` immediately after
   `h1#metareview-id` directly; if any blockquote inside has non-empty, non-"N/A" text, parse it
   as a single meta-review (`meta_reviewer_num=1`); only treat it as truly unfilled if every
   blockquote in that `<ul>` is empty.

**Differences from 2024/2025:**

| Item | 2022/2023 actual | 2024/2025 |
|---|---|---|
| Score scale | 1-8 | 1-6 |
| Score format | bare integer `"6"` | `"(6) Label text"` |
| Score field label | `"scale of 1-8"` | `"scale of 1-6"` |
| Confidence format | text-only `"Very confident"` (no number) | `"Very confident (4)"` |
| `confidence_raw` | always `null` (no number to extract) | integer |
| BibTeX | absent | present |
| PDF link | absent (DOI/SharedIt only) | present |
| Meta-review elements | `<h2>` | `<h2>` (same) |
| Meta-review #1 id | `meta-review--1-primary` (double dash) | `meta-review-1` |
| Post-rebuttal section | `<h1 id="postrebuttal-id">` containing extra `<h2>` meta-reviews | none |
| 2022 post-rebuttal score | bare integer in `[Post rebuttal]` field (`post_rebuttal_has_score: true`) |; |
| 2023 post-rebuttal | text label only (`"N/A"` if unchanged) |; |
| 2022 meta-review recommendation | `"After you have reviewed the rebuttal, please provide your final rating..."` → `"Accept"` | same label |
| 2023 meta-review recommendation | absent (decision embedded in free text; `recommendation: null`) | present |
| 2022 extra fields | `review_stack_rank`, `review_stack_size` (bare integer) |; |
| 2023 extra fields | none (stack fields absent) |; |

**`early_accepted` detection for 2022/2023**; verified, exact-match signal:

```python
result["early_accepted"] = soup.find(id="postrebuttal-id") is None
```

A paper is early-accepted if and only if it has **no** `<h1 id="postrebuttal-id">` section -
i.e. it was decided in a single primary meta-review round with no post-rebuttal discussion
needed. This was confirmed by fetching live HTML for the full population of both years and
comparing against MICCAI's official early-accept stats:

| Year | No `postrebuttal-id` (detected) | Official early-accept count | Match |
|------|----------------------------------|------------------------------|-------|
| 2022 | 249 / 573                        | 249                           | exact |
| 2023 | 308 / 730                        | 308                           | exact |

Two earlier heuristics were tried and were both **wrong** (overcounted by ~20%, ~47-61 papers per
year) because they were based on false premises that only became apparent after sampling live
HTML directly:
- "No reviews at all" (absence of `review-1`); wrong because review numbering can have gaps
  (see pitfall #1 above); every paper sampled under this signal actually had reviews.
- "Meta-review `<ul>` with no `<h2>` wrapper = unfilled form"; wrong because that structure is
  usually **filled** with real content (see pitfall #2 above); every paper sampled under this
  signal had real meta-review text, not a blank template.

Do not use review/meta-review presence as an `early_accepted` signal for 2022/2023; use
`postrebuttal-id` absence only.

Both years: the `authorFeedback-id` section exists for all papers (even early-accepted ones).
`author_feedback_text == "N/A"` means the author chose not to submit a rebuttal; it does **not**
imply early acceptance (papers that went through the full post-rebuttal cycle can still have
`"N/A"` feedback if the author opted not to respond). `rebuttal_provided` is set to `False` only
when the blockquote contains exactly `"N/A"`.

**This rule applies to every era, and 2024/2025 got it wrong until 2026-08-30.**
`era_2024_2025._parse_author_feedback` was missing the `!= "N/A"` check its
2022/2023 sibling has, so it recorded 134 papers (71 in 2024, 63 in 2025) as
having rebutted when they had declined to. The field is written by the scrapers
and read by nothing, so no published number was ever wrong, but the defect
survived a full re-scrape of 2024 in 2026-08 because the audit tool checks
review field labels and not this flag. `tests/test_parsers.py` now runs both era
modules over the same fixtures and asserts they agree, so the two cannot drift
apart again silently.

**Score extraction for bare integer format**:
```python
m = re.search(r'^\s*(\d+)\s*$', blockquote_text.strip())
score_raw = int(m.group(1)) if m else None
```

---

### Era 2021 (CONFIRMED: live HTML verified, `era_2021.py` implemented)

Live HTML verified from two sample pages before implementation. Structure is nearly identical
to 2022/2023. `era_2021.py` delegates `parse_categories_page` and `parse_paper_page` to
`era_2022_2023.py` and provides a custom `parse_index_page` for the date-path hrefs.

**Confirmed structure** (both sample pages verified):
- Same `<h1 id="...">` section ids as 2022/2023: `author-id`, `abstract-id`, `link-id`,
  `code-id`, `dataset-id`, `review-id`, `metareview-id`, `authorFeedback-id`, `postrebuttal-id`
- Title as plain `<h1>` with no id (same as 2022/2023)
- Same `postrebuttal-id` early_accepted signal (same as 2022/2023)
- No BibTeX section
- No post-rebuttal reviewer score fields (confirmed absent in reviews)
- Link section: `<p>DOI: <a href="...">...</a></p>` and `<p>SharedIt: <a href="...">...</a></p>`

**Differences from 2022/2023** (all handled via config.yaml):

| Item | 2021 | 2022/2023 |
|---|---|---|
| Score scale | 1-9 | 1-8 |
| Score format | `"Probably accept (7)"`; parens | bare integer `"6"` |
| Score field label | `"state your overall opinion"` | `"scale of 1-8"` |
| Confidence format | text only (no number) | text only (no number) |
| `confidence_raw` | always `null` | always `null` |
| Stack rank/size | same fields, bare integer | same |
| Index hrefs | `2021/09/01/NNN-PaperNNNN.html` | `NNN-PaperNNNN.html` |
| BibTeX | absent | absent |
| Post-rebuttal score | absent | 2022: present; 2023: absent |

**Index page hrefs**; the key difference requiring a custom `parse_index_page`:
hrefs are relative paths like `2021/09/01/001-Paper1891.html`. Full URL = `{paper_url_base}/{full_path}`.
Categories page hrefs use `../2021/09/01/NNN-PaperNNNN.html`; `parse_categories_page`
extracts just `NNN-PaperNNNN.html` as the key (last component), which matches what
`scrape.py` uses for area_map lookup (`paper_url.split("/")[-1]`).

**`early_accepted` detection for 2021**: same signal as 2022/2023:
```python
result["early_accepted"] = soup.find(id="postrebuttal-id") is None
```

**531 papers** on the index page (confirmed by fetching live index).

---

### Categories Page (all years)

The categories page is the **only source for subject areas**. Scrape it first for each year,
before scraping individual papers. Build a dict: `{relative_paper_path: [subject_area, ...]}`.

Structure (2024/2025, confirmed from live HTML inspection):
```html
<h3 id="Applications -> Anomaly Detection">Applications -&gt; Anomaly Detection</h3>
<div class="posts-list-item">
  <li class="posts-list-item-name float-left"><a href="/miccai-2025/0022-Paper4374.html">Paper title...</a></li>
</div>
<div class="posts-list-item">...</div>
<h3 id="Applications -> Brain Network Analysis">...</h3>
<div class="posts-list-item">...</div>
```

**Confirmed structure**: bare `<div class="posts-list-item">` siblings after each `<h3>`,
not a wrapping `<ul>`. Walk next siblings until the next `<h3>`, collecting `<a href>`
from each `<div>`. This is already correctly implemented in `parse_categories_page`.

Parse: find all `<h3>` elements. For each, the `id` attribute IS the subject area name.
Extract the path from each `<a href>` in the following `<div>` siblings and map it
to the subject area. A paper can appear under multiple `<h3>` entries.

---

## Scraper Design

### `scraper/scrape.py`: entry point

```
python scraper/scrape.py --year 2025 [--resume] [--limit N]
```

Execution order:
1. Load `config.yaml` for the given year
2. Scrape categories page → build `subject_area_map` dict
3. Scrape paper index page → build ordered list of `(paper_url, paper_id)` tuples
4. Load checkpoint **only if `--resume` is passed** (`scrape.py:178`). A checkpoint
   file left over from an earlier run is ignored without that flag, so a plain
   re-run always starts fresh. Do not "tidy up" by making it auto-resume: a
   re-scrape after a parser fix would then silently skip every page and report
   success on unchanged data.
5. For each paper not in checkpoint:
   a. Fetch the paper page HTML
   b. Parse all fields using the era-appropriate parser
   c. Look up subject areas from `subject_area_map`
   d. Append result to in-memory list
   e. Every 50 papers: write checkpoint to `data/raw/checkpoints/miccai_YYYY_checkpoint.json`
   f. Sleep `random.uniform(delay_min, delay_max)` seconds
6. Write final output to `data/raw/miccai_YYYY.json`

### Rate limiting
```python
import random, time
time.sleep(random.uniform(1.0, 1.5))
```
User-Agent: `"Mozilla/5.0 (compatible; MICCAI-Explorer-Scraper/1.0; +research)"`

### Error handling
- HTTP errors (4xx, 5xx): log URL and error, write `{"paper_id": ..., "error": "..."}` to
  output, continue. Do NOT crash.
- Parse errors: same; log, record error field, continue.
- After full scrape: print a summary of how many papers had errors.

---

## Embedding

**Script**: `analysis/embed.py`  **Switcher**: `analysis/use_model.py`

**Active model**: `specter2`; `allenai/specter2_base` with the `proximity` adapter via the
`adapters` library. Chosen because it was trained on scientific paper citation graphs
("papers that cite each other are similar"), which directly matches the goal of clustering
papers by research subfield. Five models were evaluated; SPECTER2 produced the most
thematically specific clusters.

### Model registry

`embed.py` supports five models via `--model`:

| Slug | HF ID | Dim | Loader | Notes |
|------|-------|-----|--------|-------|
| `specter2` | `allenai/specter2_base` | 768 | `specter2` (adapters lib) | **Active/recommended** |
| `bge-large` | `BAAI/bge-large-en-v1.5` | 1024 | sentence-transformers | Best general-purpose alternative |
| `gte-large` | `Alibaba-NLP/gte-large-en-v1.5` | 1024 | sentence-transformers | `trust_remote_code=True` |
| `e5-large` | `intfloat/e5-large-v2` | 1024 | sentence-transformers | Requires `"passage: "` prefix |
| `all-mpnet` | `sentence-transformers/all-mpnet-base-v2` | 768 | sentence-transformers | Worst clusters; kept for reference |

### File scheme

Models and projections are fully decoupled; switching either never overwrites other files:

```
data/processed/
  embeddings_{model}.npz           # embeddings, cluster_ids, paper_ids (no coords)
  proj_{model}_{proj}.npy          # 2D coords only, N×2 float32 (~25 KB each)
  cluster_labels_{model}.json      # 20-entry dict {"0": "label", ...}; human-edited
  cluster_labels.json              # copy of cluster_labels_{active_model}.json
  active_model.txt                 # e.g. "specter2"
  active_proj.txt                  # e.g. "umap-tight"
```

`cluster_labels.json` is the single source of truth read by `build_charts.py` (which patches
cluster labels in-memory; no need to touch `miccai_all.json` after editing cluster names).
`build_site.py` reads `active_model.txt` + `active_proj.txt` and passes `embed_model` /
`embed_proj` variables to Jinja2 templates.

### Switching models and projections

```bash
python analysis/use_model.py specter2              # keep current proj (reads active_proj.txt)
python analysis/use_model.py specter2 umap-tight   # switch model + projection
python analysis/use_model.py specter2 pacmap        # try PaCMAP projection
```

`use_model.py` loads the .npz (cluster data) + proj .npy (2D coords), patches
`miccai_all.json`, copies `cluster_labels.json`, writes `active_model.txt` + `active_proj.txt`.
Takes ~5 seconds. No GPU needed.

### SPECTER2 encoding

```python
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
model.load_adapter("allenai/specter2", source="hf", set_active=True)
model.to('cuda')
model.eval()

# Input: title SEP abstract (SPECTER2 convention)
texts = [f"{p['title']} {tokenizer.sep_token} {p.get('abstract', '')}" for p in papers]
# CLS token → 768-dim embedding
```

Sentence-transformers models use: `f"{prefix}{p['title']}. {p.get('abstract', '')}"`
(prefix is `""` for most models, `"passage: "` for e5-large).

### Projection registry

`embed.py` supports four projection methods via `--proj` (default: all four are computed):

| Slug | Method | Params | Notes |
|------|--------|--------|-------|
| `umap` | UMAP | n_neighbors=15, min_dist=0.12, n_epochs=200 | **Active**; original params, best visually |
| `umap-tight` | UMAP | n_neighbors=30, min_dist=0.05, n_epochs=500 | Tighter clusters |
| `pacmap` | PaCMAP | n_neighbors=auto, MN_ratio=0.5, FP_ratio=2.0 | Explicit local+global balance |
| `densmap` | densMAP | n_neighbors=30, min_dist=0.05, n_epochs=500 | Preserves cluster density |

All projections use `metric="cosine"` and `random_state=42`. Projections are saved as
`proj_{model}_{proj}.npy` (N×2 float32). Switching projection takes ~5s via `use_model.py`.

### Projections + KMeans

Encode in batches of 32. Save `embeddings_{model}.npz` with keys:
`embeddings` (float32, N×dim), `cluster_ids` (int32, N), `paper_ids` (str array, N).
Save each projection separately as `proj_{model}_{proj}.npy`.

**Clustering**: `KMeans(n_clusters=20, n_init=10, random_state=42)` on raw embeddings
(not 2D projection coords). After running, edit `cluster_labels_{model}.json` with
human-readable names, then run `use_model.py {model} {proj}` and rebuild charts/site.

### PyTorch CUDA note

`pyproject.toml` pins torch via the `pytorch-cu126` index
(`https://download.pytorch.org/whl/cu126`) so `uv sync` installs `torch 2.12.0+cu126`.
Required because PyPI's default torch build (cu130) requires CUDA 13.0, which fails
on drivers that top out at CUDA 12.8. The cu126 build works on any driver new enough
for CUDA 12.6. If your driver supports CUDA 13.0, the default index is fine.
Do not change the torch index in `pyproject.toml` without verifying driver compatibility.

---

## Oral & Spotlight Program

**Scripts**: `analysis/extract_orals.py` → `analysis/match_orals.py` →
`analysis/oral_stats.py` (join + statistics) → `analysis/build_oral_charts.py`

MICCAI publishes peer reviews for every accepted paper and, separately, a
program book naming which papers got a talk. Joining the two is what this
pipeline is for; no public analysis of MICCAI orals existed before it.

### Source PDFs (CONFIRMED: all five parsed, 377/377 presentations matched)

`data/manually_downloaded/OralSchedules/{year}.pdf`, collected by hand. All five
carry a real text layer; no OCR. Verified structure:

| Year | Columns | Sessions | Presentations | Types | Block delimiter | Title signal |
|------|---------|----------|---------------|-------|-----------------|--------------|
| 2021 | 2 | 12 (6 slots × A/B) | 60 | oral only | bold-after-non-bold | bold |
| 2022 | 1 | 8  | 41  | oral only | `09:00 - 09:15` time prefix | bold |
| 2023 | 1 | 12 | 68  | oral only | `09:00-09:15` time prefix | bold |
| 2024 | 2 | 16 | 96  | oral + spotlight | vertical gap | `Speaker:` marker |
| 2025 | 2 | 14 | 112 | oral + spotlight | vertical gap | `Authors:` marker |

**The two-type split starts in 2024.** 2021-2023 contain zero occurrences of
"spotlight". A spotlight is a shorter talk with no Q&A.

**Type assignment is exact, not heuristic**: sessions headed
`Oral & Spotlight Session N` contain literal `Oral Presentations:` /
`Spotlight Presentations:` subheaders; sessions headed `Oral Session N` or
`Oral N` are entirely oral. Reset the type to `oral` at each session start.

### Parsing pitfalls (all found the hard way: do not re-derive)

1. **Two-column years must be split by x-midpoint before text is reassembled.**
   `pdftotext -layout` reads across both columns and splices unrelated titles
   together, corrupting the exact field the join depends on. Affects 2021, 2024,
   2025.
2. **A session continues in the same column of the next page**, not on the next
   page's first column. Parser state must be kept **per column index**; a single
   global "current session" credits page 2's presentations to the wrong session
   (this produced 6 and 18 presentations for two sessions that each hold 12).
3. **Bold is not a usable title signal in 2024**; that program sets the whole
   page in a bold face. 2024 relies on the `Speaker:` marker plus the doubled
   vertical gap instead.
4. **Markers must be regexes, not literals.** 2023 prints `Speakers:` (plural)
   for a two-person talk; 2025 prints `Authors :` with a space before the colon.
5. **2021's gutter `#N` markers break naive line handling.** They are set in the
   regular face at a smaller size and land on the same line as the title they
   number, so they must be excluded from the bold-majority vote (otherwise a
   two-word line ties and loses, dropping a word from the title) and stripped
   from the title text. Their uneven leading is also why 2021 cannot use the gap
   rule.
6. **Titles wrap with hyphens.** `Multi-` + `phase CTA` must join without a
   space; join directly whenever the previous line ends in `-`.
7. **Not every program entry is a paper.** 2023 schedules one invited talk
   (`Invited Session Speaker:`) and one panel (`Panel Discussion` /
   `Panelists:`). Both are dropped structurally by `extract_orals.py`.
8. **Session-heading-sized lines are not all headings.** 2022 sets its date
   strapline at the heading size; 2021 sets the chairs there. Route by content
   (date-range regex, `Chairs:`) before treating a line as a wrapped title.

Per-year parameters live in `config.yaml` → `oral_schedules:`.
`extract_orals.py --dump YYYY` prints the parse without writing anything.

### Matching

Titles are the only join key; every other field already exists in
`miccai_all.json` at higher fidelity. Speaker and author names are extracted for
provenance and as a tie-break, and drive no analysis.

Cascade, most-certain first: exact on normalized title → `rapidfuzz` ≥ 92 and
clearly ahead of the runner-up → speaker surname present in the candidate's
author list → hand override. **Anything left is a hard error** (nonzero exit):
silently dropping a talk would bias every downstream statistic toward whatever
kind of title fails to parse.

As of 2026-08, **all 377 presentations across 2021-2025 matched exactly**, with
no fuzzy match and no override needed. `data/oral_title_overrides.yaml` is
therefore empty but retained for future years.

### The type join, and the confounder

`oral_stats.attach_types()` sets `presentation_type` ∈ {`oral`, `spotlight`,
`poster`} and `oral_session` **in memory only**, exactly like the
`cluster_labels.json` patch; `miccai_all.json` is never rewritten, so re-running
`normalize.py` cannot drop the types. Every accepted paper gets a poster, so
`poster` means poster-only and is the comparison group, not a leftover bucket.

**Early acceptance is a confounder, not a side analysis.** Early accepts are
decided pre-rebuttal and orals are drawn from accepted papers; 49-85% of orals
were early-accepted against 28-40% of posters. A pooled score gap could
therefore be entirely an early-accept artefact, so the oral-vs-poster effect is
also reported *within* each stratum (`chart_strata`). It survives in both, but
narrows - do not remove that chart or the caveat.

Statistics use Cliff's δ (ordinal scores, comparable across the 1-9 → 1-8 → 1-6
scale changes), Wilson intervals for proportions, and percentile bootstrap for
means. `confidence_label`; not `confidence_raw`, which is null before 2024 -
is ordinal-encoded 1-4, giving a five-year confidence analysis.

---

## Website Architecture

### Pages

| URL | Template | Content |
|-----|----------|---------|
| `/` | `index.html` | Cross-year trends + all-years analysis |
| `/orals.html` | `orals.html` | Oral & spotlight analysis (all years) |
| `/papers.html` | `papers.html` | All 3,717 papers: search, filter, sort |
| `/about.html` | `about.html` | What the site is, how the data was collected, links out |
| `/year/2025.html` | `year.html` | 2025-specific analysis |
| `/year/2024.html` | `year.html` | 2024-specific analysis |
| ... | ... | ... |

Navigation bar (on every page):
```
MICCAI Explorer   Overview & Trends | Orals & Spotlights | 2025 | 2024 | 2023 | 2022 | 2021 | All Papers
```

The Orals & Spotlights entry is rendered only when `has_orals_page` is true
(i.e. `data/processed/orals.json` exists). Without it the page is not written,
any stale copy is deleted, and the nav entry is hidden; the feature is
strictly additive and its absence cannot break a build.

**About is reached from the footer, not the nav.** The nav is a single
`height: 50px` flex row with `white-space: nowrap` on every entry and no
wrapping rule, so an eighth entry pushes it wider than a narrow screen and the
year logos are the first thing squeezed. The footer carries the byline,
`About`, and `GitHub` on every page instead, which also covers mobile, where
`.nav-links` is `display: none` below 700px and the footer is the only
navigation left. The repository and blog URLs come from `site_settings` in the
root `config.yaml` (`repo_url` and `blog_url`); an empty value writes
no link, so a fork that has not set its own does not link back to this one, the
same reasoning as the analytics id beside them.

### Charts on `index.html` (cross-year)

✓ = implemented ·; = pending (add to `build_charts.py` + `templates/index.html`)

Page order (top to bottom): semantic map → cross-year stat strip → Years at a Glance
→ Subject Areas Across Years → Explore by Year.

| Filename | Type | Content | Status |
|----------|------|---------|--------|
| `map_all.html` | Scattergl | All papers, year-color toggle + cluster toggle. `margin` sets `autoexpand=False` with a fixed `r=300` so the plot does not change width when switching between the 5-entry Year legend and the 20-entry Cluster legend | ✓ |
| `trends_overview.html` | 3×3 grid of 9 Plotly divs | Submissions, Papers Accepted, Acceptance Rate, Early Accepted (% of submissions; denominator is `num_papers_submitted.yaml`), Unique Authors, Papers with Code, Avg Review Score, Avg Review Length, Does the Rebuttal Help?. Bars per year in `YEAR_COLORS` + gray connector line along bar tops. **Not a `make_subplots` figure**; it is hand-written HTML (`_TRENDS_GRID_TEMPLATE`) because each panel needs its own y-range headroom and margin, and the CSS grid reflows to 2 columns under 820px | ✓ |
| `subject_lines.html` | line subplots | Top 12 subject areas over time, share of each year's papers. y-axes are independent per panel. Only areas present in **every** year are eligible (`presence[a] == len(years)`); see the note below | ✓ |
| `subject_movers.html` | diverging bar + controls | 8 fastest-growing and 8 fastest-shrinking areas. **Hand-written HTML** (`_MOVERS_TEMPLATE`) with From/To year dropdowns; every `from < to` window is precomputed in Python and embedded as JSON, so switching windows redraws via `Plotly.react` and never recomputes | ✓ |
| `subject_bump.html` | bump chart | Rank of top 15 areas per year. **Built but not shown**; see below | (off) |
| `subject_heatmap.html` | heatmap | Year × subject area (% of that year's papers). **Built but not shown**; see below | (off) |
| `trends_keywords.html` | bar | MICCAI buzzword % per year |; |
| `cooccur_keywords.html` | network | Keyword co-occurrence, all years |; |
| `cooccur_authors.html` | network | Author collaboration, all years (≥5 papers total) |; |

✓ = implemented and on the page · (off) = function kept, call removed ·; = pending
(add to `build_charts.py` + `templates/index.html`)

`chart_subject_bump` and `chart_subject_heatmap` still work but are **not called** from
`main()` and have no index card: they restate the same rise/fall data as Trending Topics
+ Biggest Movers. `main()` deletes their stale output files. To bring either back,
re-add the call in `main()` and the card in `templates/index.html`.

**Trending Topics excludes any area missing from even one year.** The filter was
`presence[a] >= len(years) - 1` until 2026-08-30, which allowed exactly one zero
year, and one area used it: "Image Reconstruction" ran at 10.5, 11.9, 12.5, and
12.6% from 2021 to 2024 and then dropped to a flat zero, because MICCAI 2025
deleted that subject area from its taxonomy. Reconstruction research did not
stop; 14.8% of 2025 papers still mention it (against 14.4% in 2024), they are
simply filed under the 2025 "Deep Learning" catch-all, "Image Synthesis /
Augmentation / Super-Resolution", and "Other". `SUBJECT_ALIASES` handles renames
that map one old name to one new name; a tag that was dissolved into several
cannot be recovered that way, so the only honest options were to exclude it or
to draw a cliff that means nothing. Microscopy took the freed panel. The same
filter is used by `chart_subject_bump`, which is built but not shown, so
bringing that chart back cannot reintroduce the artefact.

**The filter fixes the falling-to-zero case only. The jumping case is still in
the data, and is captioned rather than corrected.** MICCAI 2025 did not just
delete some tags, it replaced the whole taxonomy. 2024 had 31 ad-hoc top-level
facets; 2025 has exactly six (Applications, Machine Learning, Modalities, Body,
Surgery, Special Topic) and papers appear to pick one tag from each, so the
median paper carries 5 tags against 3 in 2024. An area whose name survived can
therefore grow enormously without any research moving:

| Area | 2024 | 2025 | Why |
|---|---|---|---|
| Abdomen | 3.6% (31 papers, `Clinical applications - Abdomen`) | 16.9% (174 papers, `Body -> Abdomen`) | body region became a required facet |
| CT / X-Ray | 14.8% (`Modalities - CT`) | 31.6% (`Modalities -> CT / X-Ray`) | 2024 had no X-Ray tag at all; 2025 merged the two |
| Microscopy | 3.7% | 10.8% | same pattern |

For scale: the biggest riser from 2023 to 2024 is +2.6 points; from 2024 to
2025 it is +16.8, with four areas above +7. Research does not move like that.
This also contaminates `subject_movers.html`, which ranks areas by exactly this
quantity, so any window ending in 2025 ranks the relabelling. Its default window
was changed from `YEARS[length-1]` to `YEARS[Math.max(1, length-2)]`, so it opens
on 2021 to 2024 (risers of +1.3 to +3.0) instead of 2021 to 2025 (CT / X-Ray
+13.8, Abdomen +10.9). Ending in the last year is still reachable from the
dropdown; it is just not what a reader sees first.

Neither chart's data was changed, because the alternatives were worse: dividing
by the year's tag count rather than its paper count only shrinks the effect
(Abdomen still quadruples), and cutting 2025 out entirely discards the year
readers come for. **The explanation lives in exactly one place**, the "Note:
subject-area taxonomy changes over time" card in `templates/index.html`, which
sits above both charts; the two chart captions carry only what is specific to
them and point up at it. It was briefly written out in all three, which is
redundant and means three places to keep in step. Do not "fix" the numbers
without updating that note and this section together. Found 2026-08-31.

Two panels of `trends_overview.html` need care:
- **Avg Review Score** plots `avg_score_normalized × 100` = "% of that year's scale", so
  it stays comparable as the scale changed (1-9 → 1-8 → 1-6). The raw mean on the native
  scale (`3.89/6`) is printed as an annotation pinned to the top of the panel.
- **Avg Review Length** and Avg Review Score both carry ±1 SD whiskers. Their value
  labels are annotations pinned to the panel top, not `textposition="outside"`; on the
  bar, the label collided with the whisker. Those two panels get 1.34× headroom; the
  other bar panels get 1.22×.
- **Does the Rebuttal Help?** is the share of reviewers who crossed the accept/reject
  line after rebuttal. A year is only included if it has ≥ 30 reviews with *both* a
  pre- and a post-rebuttal verdict label, which in practice means 2024 and 2025.
  Earlier years recorded post-rebuttal values as raw scores on different scales, so
  "the score went up" is not comparable across eras; crossing the accept/reject line is.

Note: trend charts work as bar/grouped-bar even with 2 years; do not skip them.

### Charts on `year/YYYY.html` (per-year)

Implemented for all scraped years (2021-2025). All per-year charts use that year's
logo colors (`YEAR_PALETTES` in `build_charts.py`); distribution charts are
histogram + KDE overlays on the RAW score scale; see DESIGN.md.

| Filename | Type | Content |
|----------|------|---------|
| `map_YYYY.html` | Scattergl | Papers that year, colored by cluster |
| `subjects_YYYY.html` | horizontal bar | Subject area paper counts (top 20). Long names wrap to two lines via `_wrap_ticks` (not `_trunc_ticks`) with `automargin=True`; the full name stays in the hover. Frame height is computed (see below) |
| `scores_YYYY.html` | hist+KDE subplots | Reviewer scores + paper avg scores, raw 1-scale_max; early-accepted papers overlaid as a darker narrower series in both panels; averages panel x-clipped to the data range. Both panels use the year MAIN color; `controversy` takes the secondary (1-1-2 pattern across the Review Analysis row) |
| `controversy_YYYY.html` | hist+KDE | Score range (max−min per paper); year SECONDARY color |
| `rebuttal_YYYY.html` | Sankey | Reviewer verdict flow pre → post rebuttal. **2024/2025 only**; earlier years have no post-rebuttal verdict labels; `chart_rebuttal_sankey` returns False and the template omits the card (`has_sankey`). Carries "Pre-rebuttal" / "Post-rebuttal" column headers and a headline stating what share of reject-leaning reviewers ended on the accept side |
| `code_YYYY.html` | horizontal bar | % with code by subject area (min 8 papers). Same `_wrap_ticks` two-line labels as `subjects_YYYY`; the dashed year-average line is labelled with its value (`Year avg 64.0%`). Frame height is computed (see below) |
| `authors_YYYY.html` | horizontal bar | Top 20 authors; x-ticks at multiples of 5 + max, never rotated |
| `coauthor_YYYY.html` | D3 or Plotly network | Co-authorship; min papers per author from `coauthorship_network_min_papers` in root `config.yaml`. Node color = collaboration community found by `nx.community.greedy_modularity_communities`; connected components are laid out separately and shelf-packed so isolated pairs do not sit on top of the main cluster; only the top `LABEL_TOP_N` (25) authors by paper count get a printed name. See "Co-authorship network renderer" below |
| `buzzwords_YYYY.html` | horizontal bar | MICCAI buzzword frequency |
| `naming_YYYY.html` | 1×2 subplots | Title length hist+KDE + most common first words (articles/prepositions filtered via `FIRST_WORD_STOPWORDS`) |

**Year-page frame heights are computed, not hardcoded.** `build_charts.py` records
what each horizontal bar chart drew into `website/charts/chart_heights.json`
(`_bar_height`: one row per bar, one extra line per wrapped label, plus fixed chrome);
`build_site.py` reads it and passes the numbers to `year.html`. Two pairs share a
two-column row and are forced to the taller of the two, so neither sits in a card with
a band of empty space under it: `subjects_YYYY` with `authors_YYYY`, and `code_YYYY`
with `buzzwords_YYYY`. Everything else uses `_STATIC_HEIGHTS` in `build_site.py`.
Hardcoded heights went stale every time the type scale changed; do not reintroduce
them. See DESIGN.md "Type scale".

**Top Papers section** (below Paper Title Patterns, plain HTML tables from
`compute_top_papers` in `build_site.py`, not Plotly):
- *Top 20 Papers (Overall)*; Title (linked), Subject Area (first area; full list in
  tooltip), Mean Reviewer Score (Scores) like `4.67 (4, 5, 5)`, Presentation, Early
  Accept ✓/✗. Ranked by mean raw reviewer score desc. All years.

**The Presentation column** shows `Oral` / `Spotlight` / `Poster`, read from
`presentation_type`, which `oral_stats.attach_types()` sets before
`compute_top_papers()` runs. It is wrapped in `{% if has_orals_page %}`, so a checkout
without `data/processed/orals.json` drops the column rather than printing a blank one.
Oral and Spotlight render as coloured badges using the presentation-type palette from
the Orals page; Poster is plain muted text, because it is the default rather than a
distinction. The badge styling sits on an inner `<span>`, not the `<td>`: a background
on the cell stretches to the full row height.
- *Top 20 Papers (Rebuttal)*; three modes (`rebuttal_mode`), chosen from the data:
  - `"scores"` (2022/2023/2024): numeric post-rebuttal scores exist
    (2022/2024 `post_rebuttal_score_raw`, 2023 bare-integer `post_rebuttal_label`).
    Column: Mean Post-Rebuttal Score (Scores). Ranked by mean post score desc,
    ties by improvement.
  - `"verdicts"` (2025): only Accept/Reject post-rebuttal labels exist; column
    "Post-Rebuttal Verdicts"; ranked by mean verdict rank, then improvement.
  - `None` (2021): no post-rebuttal review data exists at all; the template
    renders a note card explaining why the table is absent (never omit silently).
  In both table modes, reviewers who did not respond post-rebuttal keep their
  original score/verdict (carry-forward), otherwise papers with a single
  enthusiastic respondent would dominate.

### Charts on `orals.html`

Built by `analysis/build_oral_charts.py` (kept out of `build_charts.py`, which is
already long); `build_charts.py` calls `build_all()` at the end of its run, so one
command still builds everything. All nine no-op and delete their stale output when
`orals.json` is absent.

Type colors come from `oral_stats.TYPE_COLORS`, **not** the per-year logo palette:
a reader comparing types across years needs "oral" to look identical in every
panel. Poster-only is a recessive gray; it is the baseline, not a third highlight.

| Filename | Type | Content |
|----------|------|---------|
| `orals_overview.html` | 1×2 bar | Counts by type + share of accepted papers selected |
| `orals_scores.html` | grouped bar | Mean score by type, as % of that year's scale. Follows the overview page's Avg Review Score panel: the raw mean is **pinned to the top of the panel** as `6.54/9` regardless of bar height, hover carries mean ± 1 SD, whiskers are bootstrap CIs. Bar positions are set explicitly with `barmode="overlay"` (see `_group_positions`) because "group" would reserve an empty slot in 2021-2023, which have no spotlight type |
| `orals_effect.html` | forest | **Centerpiece.** Cliff's δ with CIs for oral/spotlight/poster pairs, shaded magnitude bands |
| `orals_early.html` | grouped bar | Early-accept rate by type, Wilson CIs |
| `orals_strata.html` | forest | The confounder control: oral-vs-poster δ pooled and within early / non-early strata |
| `orals_confidence.html` | line + CI | Mean reviewer confidence (1-4) by type; a null result, and shown as one |
| `orals_areas.html` | diverging bar + controls | Subject-area selection rate vs the overall baseline (min 25 papers). **Hand-written HTML** (`_AREAS_TEMPLATE`) with a Year dropdown, same pattern as `subject_movers.html`: every window (all years, plus each year) is precomputed in Python and switched client-side via `Plotly.react`. The baseline is recomputed per window. Each row also records the years its label was actually used, because MICCAI renames its subject areas: no label appears in all five years and 102 of 161 appear in exactly one |
| `orals_map.html` | Scattergl | UMAP map, selected papers over a faint poster field |
| `orals_code.html` | grouped bar | Code-release rate by type, per year, Wilson CIs; counts pinned at the top like the score chart |

Iframe heights: overview `420`, confidence `430`, scores/early/code `470`, effect `560`,
strata `520`, areas `620`, map `620`.

`orals_code.html` replaced a pooled `orals_profile.html` that also plotted authors
per paper and title length. Those two were flat (5.8 / 6.0 / 6.2 authors and 10.5 /
10.6 / 10.6 words across the three types, intervals fully overlapping), so they are
stated in one line of page prose instead of drawn. Code release is charted **per
year** because pooling it hides a direction that changes sign: posters led in 2022,
the types were level in 2023, and orals pulled ahead in 2024 and 2025.

### The All Papers page (`papers.html`)

Every accepted paper in one searchable, filterable list. No Plotly and no charts,
which makes it the lightest page on the site: one HTML file plus a 515 KB
(gzipped) JSON file, against the 364 KB to 1432 KB of plotly.js every chart page
pulls (see "Plotly bundle selection").

Three files, none of which touch anything else:

| File | Role |
|---|---|
| `analysis/papers_index.py` | builds `website/papers.json`. Standard library only. |
| `templates/papers.html` | markup and controls; no data |
| `static/papers.css`, `static/papers.js` | styling and behaviour, loaded only here |

`build_site.py` calls `papers_index.build()` **after** `oral_stats.attach_types()`,
so the page reuses the already-loaded paper list (`miccai_all.json` is 59 MB and is
parsed once) and cannot disagree with the Orals page about which papers got a talk.
`static/style.css` is deliberately untouched; the table reuses its `.papers-table`
class, and everything new lives in `papers.css`.

Unlike `orals.html`, this page is **always** written: its only input is the paper
list itself, so nothing optional can be missing. Without `orals.json` every paper
simply reads as a poster.

**`website/papers.json`** (2.4 MB, 515 KB gzipped). Subject-area names are listed
once in `areas` and referenced by position; every other repeated-string saving was
measured and rejected as not worth the code. Link keys are **omitted when absent**
rather than written as `null`, so the page can test `if (p.code_url)`.

```jsonc
{"built": "2026-08-22",
 "areas": ["Applications -> Anomaly Detection", ...],   // 161
 "scales": {"2021": 9, "2022": 8, "2023": 8, "2024": 6, "2025": 6},
 "papers": [{"id": ..., "year": ..., "title": ..., "authors": [...],
             "areas": [42, 87], "url": ..., "score": 4.67, "score_norm": 0.733,
             "reviews": [4, 5, 5], "early": 0, "type": 1,
             "session": "Surgical Data Science",     // talks only
             "code_url": ..., "pdf_url": ..., "doi": ..., "sharedit_url": ...,
             "supp_url": ..., "dataset_urls": [...], "lncs": "LNCS 15964"}]}
```

`scales` is read from `score_max` on the reviews, not hardcoded, so a scale change
in a future year needs no code edit.

**Three things not to "simplify":**

- **Score sorting is offered only when the Year filter names a single year**
  (`scoreSortAllowed()` / `syncSortOptions()` in `papers.js`). Sorting scores
  across mixed years was tried first and dropped on 2026-08-30, because neither
  form of it is defensible. Raw sorting does not merely skew the order, it
  excludes years: the best possible 2025 paper averages 6.00, below the
  417th-best 2021 paper, so the top 100 by raw score is 99% 2021 papers.
  Normalizing does not rescue it, because 2021 reviewers used the top of their
  9-point scale far more freely (mean 69.3% of scale against 57.8% in 2025), so
  the top 100 by normalized score is still 62/17/13/5/3 across the five years.
  Either way the ranking reports which year a paper was reviewed in. Inside one
  year the scale is fixed and the comparison is real. The two score options are
  **disabled rather than removed**, with `#sort-hint` saying why: a silently
  absent feature reads as a missing feature. Selecting a score sort and then
  clearing the year resets the sort to `year-desc`, so no URL can reach a state
  the rule forbids.
- **The Score column shows `4.67/6` and nothing else.** It used to carry a muted
  `93%`, which existed only to explain the cross-year sort; it went when that
  sort did. `score_norm` is still in `papers.json` and still drives the
  comparator (inside one year it gives the same order as the raw score), so
  bringing a cross-year view back needs no data change.
- **The subject-area dropdown is rebuilt from whatever the other filters leave
  showing**, with a count on each entry. MICCAI renames its areas; 2025 renamed
  nearly all of them, no name appears in all five years, and 102 of 161 appear in
  exactly one. A fixed list of 161 would silently exclude whole years: picking
  "Image Segmentation" would drop every 2025 paper with nothing on screen saying
  why. Counting live makes the renaming visible instead. Every entry is written
  `Name (n=1,234)` using U+1D45B, mathematical italic small n: an `<option>`
  cannot carry markup, so that character is the only way to italicize the n
  inside a native dropdown. `All areas (3,717)` without it read as a count of
  areas rather than of papers.

**Page size and exports.** The "Show" dropdown offers 10, 50, or 100 rows
(`PAGE_SIZES`), defaulting to 50; changing it keeps the first visible row on
screen rather than jumping back to page 1. Three export actions all act on the
whole filtered set, not the visible page: CSV (UTF-8 with a byte order mark, so
Excel does not mangle accented author names), JSON (self-describing: it carries
the filters, the sort, the link that reproduces them, and subject areas written
out by name rather than as indices into `papers.json`), and Copy link. The copy
races `navigator.clipboard.writeText` against a 1.2 s timeout and falls back to
an off-screen textarea, because in some browsers that promise never settles at
all; on failure it says to copy from the address bar and **never opens a
dialog**, since a modal reporting a failed copy is worse than the failed copy.

### Map chart click-to-paper

All seven maps (`map_all.html`, the five `map_YYYY.html`, and `orals_map.html`)
get this appended after the Plotly div. It lives in one place, the module-level
`CLICK_JS` constant in `build_charts.py`, and is injected by passing
`extra_js=CLICK_JS` to `save_chart`. The maps once carried their own copy of it
plus their own `pio.to_html` call, which meant every fix had to be made twice:

```html
<script>
document.addEventListener('DOMContentLoaded', function() {
  var div = document.querySelector('.plotly-graph-div');
  if (!div) return;
  div.on('plotly_click', function(data) {
    var url = data.points[0].customdata[0];   // paper URL stored in customdata[0]
    if (url) window.open(url, '_blank');
  });
});
</script>
```
Hover shows only the paper title (set in `hovertemplate`). `customdata` stores
`[paper_url, title]` per point. The div is found **by class, not by id**, which
is what lets `save_chart` set a stable `div_id` without touching this script.

**The injection is opt-in, and a map that forgets it looks fine.** `orals_map.html`
was built without `extra_js=CLICK_JS` from the day it was written and shipped that
way; it drew correctly, hovered correctly, and carried the right `customdata`, so
nothing looked wrong until someone tried to click a point. Fixed 2026-08-31. Any
new map chart has to pass `extra_js=CLICK_JS` by hand. `save_chart` could instead
switch it on whenever a trace's `customdata` looks like `[url, title]`, but that
is inferring intent from data shape, and a chart is free to store a URL it does
not want opened. Check it by hand: `grep -c plotly_click website/charts/*.html`
should report 1 for every map.

### Chart output is byte-reproducible

`save_chart` passes `div_id="chart-<filename stem>"` rather than letting Plotly
stamp a random UUID per build, and the co-authorship code sorts everything that
feeds a layout. Two builds of unchanged data therefore produce byte-identical
files, so a rebuild diff shows only real changes. Three things must stay sorted
or this silently regresses: the active-author set in `_coauthor_graph`, the
tie-breaks in the component and community sorts, and the per-component graph in
`_coauthor_layout`, which is **constructed** rather than obtained from
`G.subgraph()` (a filtered view iterates its own internal set, so its node order
is not the graph's). See `REPRODUCIBILITY.md` I9 for the full account.

### Iframe embedding

All charts embedded as:
```html
<iframe src="/charts/map_2025.html"
        class="chart-frame" scrolling="no" frameborder="0">
</iframe>
```
CSS in `style.css`: `.chart-frame { width: 100%; height: 600px; border: none; }`

### Plotly chart config

`build_charts.py` uses `_config(interactive)` to select the right Plotly config:

- **Static charts** (bar, histogram, heatmap): `{"responsive": True, "displayModeBar": False}`
- **Interactive charts** (map_YYYY, map_all, coauthor_YYYY): `{"responsive": True, "displayModeBar": "hover", "scrollZoom": True}`; modebar appears on hover; scroll zooms, drag pans (`dragmode="pan"` set in layout)

`hoverlabel` in `base_layout()` explicitly sets `font.color=TEXT` to ensure dark text on the white hover-label background (Plotly's default hover text is white, which is invisible on `bgcolor="#fff"`).

Four charts do **not** go through `save_chart`/`pio.to_html` at all, because their
interactivity is a few lines of plain JavaScript and forcing it through Plotly would be
harder to read, not easier: `trends_overview.html` (`_TRENDS_GRID_TEMPLATE`),
`subject_movers.html` (`_MOVERS_TEMPLATE`), `coauthor_YYYY.html` under the d3 renderer
(`D3_NETWORK_TEMPLATE`), and `orals_areas.html` (`_AREAS_TEMPLATE`, in
`build_oral_charts.py`). Each is a module-level HTML template string with `__TOKEN__`
placeholders.

All four are written by **`save_template(template, filename, tokens)`**, the counterpart
to `save_chart`. `theme_tokens(bundle="basic")` supplies the shared placeholders
(`__PLOTLYSCRIPT__`, `__FONT__`, `__BG__`, `__PLOTBG__`, `__TEXT__`, `__MUTED__`,
`__RULE__`, `__ACCENT__`) and the caller adds its own; a token a template does not
contain is simply not found, so passing the whole set is free. `theme_tokens` is a
function rather than a constant because `__PLOTLYSCRIPT__` is the whole
`<script src="https://cdn.plot.ly/...">` tag, built by `plotly_script_tag()` from the
installed plotly's version rather than frozen at import time. All four templates draw
bar and scatter traces only, so they take the default `basic` bundle. Each of the
four used to carry its own copy of the fill loop plus the same `mkdir`,
`write_text` and log line.

### Plotly bundle selection

Charts do not all load the same plotly.js. `save_chart` rewrites the CDN script
tag plotly wrote to the smallest official bundle that can draw the figure.
Measured from `cdn.plot.ly`, gzipped: `plotly-basic` 364 KB (bar, pie, scatter),
`plotly-cartesian` 462 KB (adds box, contour, heatmap, histogram, image,
violin), `plotly-gl2d` 519 KB (scattergl, splom, parcoords, scatter), full
`plotly` 1432 KB (everything, and the only one with sankey).

`_BUNDLE_TRACES` lists what each bundle draws; those lists were read out of the
downloaded bundles, not off the documentation page, because a trace type a
bundle lacks draws nothing rather than erroring:

```bash
curl -sL https://cdn.plot.ly/plotly-basic-3.7.0.min.js \
  | grep -o 'moduleType:"trace",name:"[a-z0-9]*"' | sort -u
```

**Bundle choice is a property of the page, not of the chart**, because a page
built from iframes pays for the union of what its frames request. Choosing per
chart made `year/2024.html` and `year/2025.html` **57% heavier**: their nine
other charts moved to basic and gl2d while the rebuttal Sankey stayed on full,
so those pages fetched 2372 KB where they used to fetch 1467 KB. `page_bundle_floor(bundle)`
is a context manager holding a module-level floor that every choice inside the
block is widened to cover; `main()` opens it around each year's charts with
`"full"` when that year has a Sankey, so such a page comes out where it started
rather than worse. The floor has to be set before the year's first chart, and
the Sankey is drawn fifth, which is why the flow computation lives in
`_rebuttal_flows()` and is called by both `main()` and `chart_rebuttal_sankey`.

Two things not to "simplify":

- **`_FULL_BUNDLE_TAG` matches plotly's tag rather than rebuilding it**, and
  `save_chart` raises when it fails to match. Plotly's tag carries a Subresource
  Integrity hash computed from the copy of plotly.js the Python package vendors,
  so reconstructing it means reimplementing a private plotly function. If a
  plotly upgrade changes that markup, fix the regex; a silent no-op here undoes
  the whole thing and the site still works, so nothing else would catch it.
- **`_BUNDLE_SRI` pins the partial bundles' hashes per plotly.js version**,
  because plotly vendors only the full bundle and there is nothing local to hash.
  A version missing from the table logs a warning and writes the tag without an
  integrity attribute rather than failing, so a plotly upgrade degrades instead
  of breaking the build. Restore it with
  `curl -sL <url> | openssl dgst -sha256 -binary | openssl base64 -A`.

After any change here, check the assignment with
`grep -ho 'cdn\.plot\.ly/plotly[^"]*' website/charts/*.html | sort | uniq -c`;
today that is 32 basic, 5 gl2d, 18 full, with the five d3 co-authorship files
loading no plotly at all.

**Map coordinates are rounded to three decimals** in `main()`, in memory, before
any chart is built, the same idiom as the cluster-label and presentation-type
patches; `miccai_all.json` is not rewritten. `embed.py` stores the coordinates
as the float64 repr of a float32, so each is seventeen significant digits, and
`map_all.html` writes 14,924 of them (3,717 papers, twice, once per colour
mode). That made it the largest file on the site and it sits above the fold on
the home page, where lazy loading cannot help. The coordinates span about 10.5
units, so 0.001 is 0.086 px across a 900 px frame and stays sub-pixel past 10x
zoom. Nothing but the seven maps reads `umap_x` / `umap_y`.

Together the two changes took the seven chart pages from 10.93 MB to 8.05 MB
gzipped (-26%); -35 to -37% on the five pages with no Sankey, -1% on 2024 and
2025. Lighter charting libraries were considered and rejected: of uPlot,
Chart.js, ApexCharts and Recharts, only uPlot is dramatically smaller, and it
draws line and scatter charts only. The rest mean redrawing every chart on the
site, and `plotly.js/lib/core` with manual trace registration would need a
JavaScript build step this project does not have.

### Year logos

All five year logos sit in the nav, so every desktop page load fetches all of
them. The tracked source is `static/logos/miccai_YYYY.png`; what the site serves
is WebP, resized, written during the build by `analysis/logo_assets.py` from the
step in `build_site.py` that copies `static/` into `website/assets/`. Adding a
year therefore needs no image step: drop the PNG in at whatever size it comes,
and rebuild. `python analysis/logo_assets.py` runs the same conversion on its
own if you want to see the output without a full build.

Measured over the five 2021-2025 logos:

| | total | |
|---|---|---|
| as committed, PNG at 576-768 px wide | 223 KB | |
| WebP at the same pixel size | 92 KB | |
| resized to 160 px tall, still PNG | 119 KB | |
| **resized to 160 px tall, WebP** | **61 KB** | what the build writes |

**Most of the saving is the resize, not the format.** The logos are drawn at
80 px tall on the year pages (`.year-logo-img`) and 28 px in the nav, so
`MAX_HEIGHT = 160` covers a 2x display; the sources were two to four times
larger than they can ever be shown. If that CSS ever grows, raise `MAX_HEIGHT`
with it.

**The setting that matters is `alpha_quality`, not `quality`.** About three
quarters of each logo is fully transparent and libwebp encodes the alpha channel
separately: at the default `alpha_quality=100` the five files come to 94 KB, at
70 they come to 61 KB. Compared at 4x zoom against the nav background there is
no visible fringing on the thin script lettering, which is the worst case in this
set, and going lower buys almost nothing (58 KB at 60).

**Converting at build time rather than in a git hook is deliberate.** A hook that
rewrites tracked files makes a commit differ from what was reviewed, and it would
put generated artefacts in `static/`, which this repository reserves for source.
The build already owns everything under `website/`.

Pillow does the conversion, and it is the only optional dependency in the site
build: if it is not installed, `optimize()` logs a warning, returns an empty
map, and the PNGs are served exactly as they were before this existed. The
`.png` to `.webp` rename travels through the return value into
`year_meta[yr].logo`, which is why the asset copy happens **before** `year_meta`
is built rather than at the end of `main()` where it used to sit.

### Co-authorship network renderer

The root `config.yaml` has a top-level `chart_settings` block, a sibling of
`base_settings` (which is scraper-scoped):

```yaml
chart_settings:
  network_renderer: "d3"     # or "plotly"
```

- **`d3`** (default); individual nodes can be dragged, the background pans, scroll
  zooms. Loads d3 v7 (~280KB) rather than adding to the page's plotly.js bundle.
- **`plotly`**; the older static figure: pan and zoom only, no per-node dragging.

`_coauthor_data()` builds the graph, layout, community colors and node sizes;
`_render_coauthor_d3()` and `_render_coauthor_plotly()` only draw. Switching renderers
cannot change the layout, and both write the same `coauthor_YYYY.html`, so
`templates/year.html` needs no conditional. An unrecognised value logs a warning and
falls back to `d3`. Community detection is wrapped in try/except; a networkx version
mismatch degrades to a single color rather than failing the build.

Known limitation: there are more communities than colors in `CLUSTER_PALETTE`
(2025 has 45 communities, the palette has 20), so colors repeat. Because components are
packed apart spatially this reads acceptably, but color is not a unique identifier.

### Templating

`analysis/build_site.py` uses Jinja2. Passes `years` (sorted desc, from data),
`base_url=""` (see BASE_URL above), `year_meta` (logo filename + city + has_white_bg
+ year colors: `color`, `color2`, derived `accent_bg`, `color_on_dark`),
per-year stats and title-list SimpleNamespaces to the templates. Year pages also get
`scale_max`, `coauthor_min` (from root `config.yaml`), `top_overall`, `top_rebuttal`,
and `has_sankey`. `year.html` emits a `<style>` block overriding the accent CSS
variables with the year color (see DESIGN.md "Per-Year Color Themes"). The
`format_int` Jinja2 filter formats integers with thousands separators.
`build_site.py` also copies `static/` → `website/assets/`.

Per-year stats fields (`year_stats[yr]`): `n_papers`, `n_authors`, `pct_code`,
`avg_score` (raw scale mean, 2 decimals), `pct_early`, `pct_early_subs`, `city`.

**`pct_early` vs `pct_early_subs`; two different denominators, easy to confuse.**
`pct_early` is early accepts as a share of *accepted* papers (~30%); `pct_early_subs`
is the share of *all submissions* (~8-13%), read from `num_papers_submitted.yaml`. The
rebuttal Sankey caption uses `pct_early_subs` so it agrees with the "Early Accepted"
panel on the overview page.

`templates/base.html` ends with a small script that makes any card containing a
`table.papers-table` collapsible: it wraps the card body, turns the card header into a
keyboard-accessible toggle with a chevron and an "N papers" badge, and **starts
collapsed**. It is progressive enhancement; if the script never runs you get the old
always-open tables, and no table markup changed. Styles live in `static/style.css`
under "Collapsible Top Papers tables".

Per-year title extremes (`title_lists`): `longest` (top 5 by word count), `shortest`
(bottom 5 by word count), `most_authors` (top 5 by author count). Each entry has `title`,
`url`, and either `word_count` or `n_authors`. Rendered as a 3-column HTML card in the
"Title & Author Extremes" section of `year.html`; NOT a Plotly chart.

---

## MICCAI Buzzwords

Used in `trends_keywords.html` and `buzzwords_YYYY.html`. Match as document frequency
over `title + " " + abstract`, case-insensitive (`re.IGNORECASE`).

```python
BUZZWORDS = {
    "Segmentation":            [r"\bsegment"],
    "Diffusion models":        [r"diffusion model", r"\bddpm\b", r"score.based generat"],
    "Foundation models":       [r"foundation model"],
    "Transformers / ViT":      [r"\btransformer\b", r"\bvit\b", r"\bswin\b"],
    "Self-supervised / SSL":   [r"self.?supervised", r"\bcontrastive learn"],
    "Semi-supervised":         [r"semi.?supervised"],
    "Domain adaptation":       [r"domain adapt"],
    "Federated learning":      [r"\bfederated\b"],
    "Vision-language / VLM":   [r"vision.?language", r"\bvlm\b", r"\bclip\b"],
    "Large language model":    [r"\bllm\b", r"large language model"],
    "Generative / GAN":        [r"\bgenerative\b", r"\bgan\b"],
    "Image registration":      [r"\bregistrat"],
    "Image reconstruction":    [r"\breconstruct"],
    "Detection":               [r"\bdetect"],
    "Classification":          [r"\bclassif"],
    "Uncertainty":             [r"\buncertain"],
    "Weakly supervised":       [r"weakly.?supervised"],
    "Active learning":         [r"active learn"],
    "Surgical AI":             [r"\bsurgical\b", r"\blaparoscop", r"\bendoscop"],
    "Pathology / WSI":         [r"\bpatholog", r"whole.?slide", r"\bwsi\b"],
    "Explainability":          [r"\bexplainab", r"\binterpretab", r"\bxai\b"],
    "Multimodal":              [r"multi.?modal"],
    "Mamba / SSM":             [r"\bmamba\b", r"state.?space model"],
    "Segment Anything / SAM":  [r"\bsam\b", r"segment anything"],
}
```

---

## Deployment

GitHub Pages from the `gh-pages` branch of
`miccai-explorer/miccai-explorer.github.io`, served at
`https://miccai-explorer.github.io/`. Because the repository is named
`<org>.github.io`, this is an organization site and GitHub serves it at the root of
the domain, which is why `BASE_URL` is empty.

The `website/` directory is NOT committed to `main`. The GitHub Actions workflow
(`deploy.yml`) builds it and pushes to `gh-pages`.

**Google Analytics** is written into every page by `templates/base.html`, but only
when a measurement id is configured, so a local preview and a fresh clone carry no
tracking tag at all. The id comes from `MICCAI_GA_ID` in the environment, falling
back to `site_settings.google_analytics_id` in the root `config.yaml`. The
environment is the right place for the real id: `deploy.yml` passes it from the
repository variable of that name, which keeps it out of the repository so a fork
that builds this project does not report its traffic to someone else's property.
The tag is `async` and sits last in `<head>`, after the stylesheet and the font
preconnects, so it cannot delay rendering on a slow connection.

**Hosting options**: GitHub Pages serves a public repo for free. For a *private*
repo it needs a GitHub Pro or Team plan; Cloudflare Pages is free for private repos and
deploys the same output with no code changes. Note that a `<name>.github.io` address
requires a GitHub organization of that exact name, whereas `<name>.pages.dev` on
Cloudflare is free-form.

---

## Conventions

- Python 3.10+. Run all scripts from repo root.
- All URLs and paths read from the root `config.yaml`. No hardcoded URLs in Python.
- Paper IDs: `f"miccai-{year}-Paper{num:04d}"` where `num` is the 4-digit zero-padded
  paper number from the URL slug (e.g., `Paper0308` → `"miccai-2025-Paper0308"`).
- Author names stored as `"LastName, FirstName"` preserving the site's format.
- Subject areas stored as full strings: `"Machine Learning -> Foundation Models"`.
- Score normalization: `(raw - 1) / (scale_max - 1)` → maps any era to [0.0, 1.0].
- `has_code`: `True` if `code_url` is not None and not the string `"N/A"`.
- All chart HTML files: generated with `plotly.io.to_html(include_plotlyjs="cdn",
  full_html=True, config={"responsive": True, "displayModeBar": False})`, after which
  `save_chart` rewrites the CDN script tag to the smallest bundle that can draw the
  page (see "Plotly bundle selection").
