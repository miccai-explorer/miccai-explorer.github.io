# Agent Guidance

Guidance for coding agents (Claude Code, Codex, Cursor, and similar) working on this
repository. Humans should start with [README.md](README.md) and
[CONTRIBUTING.md](CONTRIBUTING.md); everything in those applies here too.

The project is built and deployed. This file is about working on it as it stands, not
about creating it.

## Read these first

| File | Read it when |
|---|---|
| `CLAUDE.md` | Always, for anything touching the pipeline. It is the full technical reference: data schema, per-year HTML parsing, chart inventory, and the pitfalls behind each. |
| `DESIGN.md` | **In full, before writing any website code**: templates, CSS, chart HTML, or `build_site.py`. It specifies every visual decision. |
| `REPRODUCIBILITY.md` | Before changing anything about how data is generated, and to see which inputs have no generating code. |
| `CONTRIBUTING.md` | For the pipeline commands and how to run each stage. |

## How this codebase expects to be changed

- **Run everything from the repository root.** Scripts resolve paths relative to it.
- **No hardcoded URLs.** Everything lives in the root `config.yaml`, which is the
  single source of truth for the whole pipeline: scraping, analysis, and the site
  build all read it. There used to be a second copy at `scraper/config.yaml`; it was
  read by nothing, drifted out of sync, and was deleted on 2026-08-30.
- **`website/` is generated.** Never hand-edit it. Change the template, the CSS in
  `static/`, or the chart function, and rebuild. That includes the year logos: the
  tracked source is the PNG in `static/logos/`, and the WebP the site serves is
  written during the build by `analysis/logo_assets.py`. Do not commit WebP files
  into `static/`, and do not add a git hook that rewrites images in place; a hook
  that edits tracked files makes a commit differ from what was reviewed.
- **`data/processed/miccai_all.json` is not rewritten by the build.** Cluster labels
  and presentation tiers are patched in memory at build time. Preserve that pattern for
  anything similar you add; it keeps stage ordering from mattering.
- **New features must degrade to a no-op.** The Orals pages are the model: with
  `orals.json` absent, the charts delete their stale output, the page is not written,
  and the nav entry disappears. A missing optional input must never break the build.
- **The All Papers page is the deliberate exception.** It is always built, because
  its only input is the paper list itself; there is nothing optional to be missing.
  It still degrades on the oral data: without `orals.json` every paper reads as a
  poster. It also adds no chart and loads no Plotly, so keep it that way.
- **A chart's plotly.js bundle is chosen by the page, not the chart.** `save_chart`
  swaps in the smallest official bundle that can draw a figure, but a page built from
  iframes pays for the union of what its frames ask for, so one chart needing the full
  bundle puts the whole page on it. Choosing per chart made two year pages 57% heavier.
  Add a new chart type and check the assignment afterwards; see "Plotly bundle
  selection" in `CLAUDE.md`.

## Verify, do not assume

Most of `CLAUDE.md` exists because something looked obvious and was wrong. Before
claiming a parsing behaviour or a data property:

- Sample real pages or records and check. A pattern that holds in three samples is not
  a rule.
- **Never audit the scraper using the scraped data.** A field the parser never
  captured leaves no trace in `miccai_all.json`, so the data looks complete and
  consistent while a third of the review text is missing. That is exactly what
  happened: four parser bugs survived for months because every check started from
  the committed data. Start from a live page instead, or run
  `python scraper/audit_review_fields.py --all`, which feeds the real parser one
  form question at a time and reports any question that sets no field.
- Where MICCAI has published its own statistics (early-accept counts, for example),
  reconcile against them. Two plausible `early_accepted` heuristics were each wrong by
  roughly 20% and looked right until checked against the published totals.
- If you find a discrepancy you cannot resolve, say so plainly rather than picking the
  reading that makes the build pass.

## Ask before

- Running `analysis/embed.py` when the embeddings already exist; it needs a GPU and
  takes minutes.
- Overwriting `data/processed/cluster_labels*.json`. Those names are hand-written, and
  each file's `clustering` fingerprint says which partition they describe. Never stamp a
  file (`describe_clusters.py --stamp`) without reading the clusters first: the stamp
  asserts the names were checked, and every build trusts it.
- Changing `BASE_URL`, the deployment workflow, or anything about how the site is
  published.
- Any `git commit`, `git push`, or branch operation. The maintainer does all of these.
- Deleting or regenerating anything under `data/raw/`.

## Safe to do unattended

- `normalize.py`, `build_charts.py`, and `build_site.py`; all are pure regeneration.
- `extract_orals.py` and `match_orals.py`.
- Running the tests.
- Editing templates, CSS, and chart code, then rebuilding to check.

## Error handling rules

- **Never crash the scraper on a single page.** Log the URL, record an `error` field,
  and continue. Report the error count in the run summary.
- If more than 5% of a year's papers fail to parse, stop and report before continuing.
- If an expected HTML landmark is missing, record a `parse_warning` on that paper and
  carry on.
- `match_orals.py` exiting nonzero on an unmatched title is correct behaviour. Fix the
  match or add an override; never suppress it.

## Code conventions

- Python 3.10 or newer. `ruff check` must pass; CI runs it with a pinned version
  before it will build. The rule set is stated explicitly in `pyproject.toml`
  (`E4, E7, E9, F, W, I`) rather than left to ruff's defaults, so an upgrade
  cannot change what passing means.
- **`ruff format` is deliberately not run on this repository.** At the 79-column
  line length it rewrites 14 files, and a good share of those changes are
  readability regressions: it de-aligns the trailing-comment blocks that
  document this codebase's traps, and wraps assignments and type annotations
  across three lines to make room for a comment. Raising the line length makes
  the diff larger, not smaller (550 changed lines at 88, 868 at 100). Format new
  code to match its neighbours by hand.
- Every script takes `--help` and prints a summary of what it did.
- `scrape.py` supports `--year`, `--resume`, `--limit`, and `--output`.
- `build_charts.py` supports `--years`.
- Comments should explain *why*, especially where the obvious approach fails. Match the
  surrounding density; this codebase comments its traps heavily and its plumbing barely.

## Writing style, for anything a reader sees

- Plain words. Real technical terms where they are the right ones, no invented jargon.
- **No em-dashes or en-dashes in prose.** Use semicolons, commas, and parentheses.
  Use the Oxford comma. This rule is about *our* writing only: several regexes
  split MICCAI's own text on `—` and `–` (recommendation labels like
  `Reject — should be rejected`, and time ranges in the program PDFs). Those are
  written as `\u2014` / `\u2013` escapes so a search-and-replace cannot silently
  break them. Leave them alone.
- Captions should say what a chart does not show as readily as what it does. Overclaim
  is a bug.
- Never present a null result as if it were a finding, and never hide one either.
