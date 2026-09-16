# Contributing

Thanks for taking a look. This project turns MICCAI's open-access proceedings and peer
reviews into a browsable website, and there is plenty left to do.

## Ways to help

- **Add the next year.** The single most useful contribution. See
  [Adding a new year](#adding-a-new-year).
- **Report a wrong number.** If a figure on the site disagrees with something you know
  to be true, open an issue with the page, the chart, and what you expected. Data bugs
  matter more than anything else here.
- **Add or improve a chart.** Several are listed as pending in `CLAUDE.md`.
- **Improve the writing.** Every chart has a caption meant to say what the chart shows
  and what it does not. If one overclaims, that is a bug.
- **Fill a reproducibility gap.** `REPRODUCIBILITY.md` lists seven inputs that were
  produced by hand and have no generating code. Each one is a self-contained task.

## Setting up

Python 3.10 or newer. Any environment manager works; pick one.

```bash
git clone https://github.com/miccai-explorer/miccai-explorer.github.io
cd miccai-explorer.github.io

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`pyproject.toml` pins the same dependencies with lower bounds if you prefer `uv sync`
or Poetry. Run every command from the repository root; the scripts resolve paths
relative to it.

**One GPU caveat.** `pyproject.toml` installs `torch` from the CUDA 12.6 wheel index,
because the default PyPI build needs a newer driver than some machines have. If your
driver is newer, the default build is fine; if you only want to rebuild the website,
you do not need `torch` at all.

## Start here: build the site without regenerating anything

The expensive outputs (scraped JSON, embeddings, 2D projections, cluster assignments,
and the matched oral program) are all committed. You can go straight to a working
site in about a minute, with no GPU and no network access:

```bash
python analysis/build_charts.py   # Plotly charts -> website/charts/
python analysis/build_site.py     # HTML pages    -> website/

python -m http.server 8000 --directory website
# open http://localhost:8000/
```

Serve `website/` itself. Generated links are root-relative (`/assets/style.css`,
`/charts/map_2025.html`) because the site is deployed as a GitHub organization site
at the root of its domain, so the directory you serve has to be the site root.
`website/` is generated output; never edit it by hand, and note that it is not
committed.

If that works, you are set up. Everything below is only needed if you want to
regenerate an earlier stage.

## Reproducing the analysis from scratch

Each stage is independently re-runnable and writes files the next stage reads, so you
can start at whichever stage you actually care about.

### 1. Scrape (network, slow, no GPU)

```bash
python scraper/scrape.py --year 2025          # add --limit 10 to try it first
python scraper/scrape.py --year 2024
# ...and so on; one year at a time
```

Per-year URLs and settings live in the root `config.yaml`. There are no hardcoded URLs
anywhere in the Python.

**Please keep the rate limiting.** The scraper sleeps 1.0 to 1.5 seconds between
requests and identifies itself in the User-Agent. papers.miccai.org is a small
community resource. Do not remove the delay, do not parallelise the requests, and use
`--limit` while developing. Progress is checkpointed every 50 papers, so an
interrupted run resumes with `--resume` rather than starting over.

### 2. Normalize

```bash
python analysis/normalize.py      # merges data/raw/*.json -> data/processed/miccai_all.json
```

### 3. Embeddings, projections, and clusters (GPU recommended)

```bash
python analysis/embed.py --model specter2
```

This writes embeddings, four different 2D projections, and KMeans cluster assignments.
It then writes a **placeholder** `cluster_labels_specter2.json`, because naming twenty
topic clusters is a judgement call a human has to make.

To name them, read what is actually in each cluster:

```bash
python analysis/describe_clusters.py specter2
```

That prints each cluster's most distinctive terms and the twelve titles nearest its
centroid. Write the names into `cluster_labels_specter2.json`, then bind them to the
clustering they describe and activate the model:

```bash
python analysis/describe_clusters.py --stamp specter2
python analysis/use_model.py specter2 umap-tight
```

**The stamp is not optional and it is not a formality.** KMeans cluster ids are an
arbitrary numbering, so recomputing a clustering moves every name onto a different group
while the file keeps looking perfectly reasonable. That is what happened before
2026-09-08: all twenty names were wrong on the published site, and no build step, test or
chart could tell, because a wrong name renders exactly like a right one. The stamp records
which partition you read, and `use_model.py`, `embed.py` and `build_charts.py` all refuse
to run without a match.

Switching between already-computed models or projections takes about five seconds and
needs no GPU.

### 4. Oral and spotlight program

This stage is independent of stages 1 to 3 and needs only `miccai_all.json`.

```bash
python analysis/extract_orals.py     # program PDFs -> data/raw/orals_YYYY.json
python analysis/match_orals.py       # join by title  -> data/processed/orals.json
```

The source PDFs are collected by hand into
`data/manually_downloaded/OralSchedules/{year}.pdf`, because MICCAI does not publish
them at a predictable URL.

**`match_orals.py` exits nonzero if any presentation is unmatched, and that is
deliberate.** A dropped talk silently biases every statistic on the Orals page toward
whatever kind of title failed to parse. If it fails, run it again with `--report` to
write a stub file, and add the missing title to `data/oral_title_overrides.yaml`.

### 5. Build

```bash
python analysis/build_charts.py
python analysis/build_site.py
```

Cluster labels and presentation tiers are patched **in memory** at build time from
`cluster_labels.json` and `orals.json`. `miccai_all.json` is never rewritten, so
re-running `normalize.py` cannot silently drop them, and you do not need to rebuild
anything after renaming a cluster.

`build_site.py` also writes `website/papers.json` (2.4 MB, 515 KB gzipped), the data
behind the All Papers page. It is built from the same in-memory paper list, so it
costs no extra load and cannot disagree with the rest of the site.

One thing that *is* dropped: `normalize.py` rebuilds `miccai_all.json` from the raw
files, which removes the map coordinates and cluster labels. If you re-run it, run
`python analysis/use_model.py $(cat data/processed/active_model.txt) $(cat data/processed/active_proj.txt)`
afterwards to put them back. That takes about five seconds and needs no GPU.

### 6. Test

```bash
python -m pytest tests/ -q
```

The tests cover the statistics with known-answer cases, check the oral extraction
against golden per-year counts by re-parsing the source PDFs, and hold golden counts
for `website/papers.json` (`tests/test_papers_index.py`). The PDF tests skip
themselves if the PDFs are absent.

### Coverage

```bash
python -m pytest tests/ -q --cov --cov-report=term
```

CI runs the same command on every push to main and prints the result in the run
log. The number is deliberately not published as a badge: it is low, and a low
badge invites the wrong fix. What ships instead is the count of passing tests,
which is the honest headline.

It is scoped, and the scope is the point. `pyproject.toml` omits the chart and site
builders and the one-shot operator commands, so what is measured is the code that
decides what the data says: the parsers, the oral extraction and matching, the
statistics, and the browse index. A bug in any of those writes a plausible wrong
number that nothing downstream can catch. The rendering code is verified a different
way, by byte-for-byte reproducibility (`REPRODUCIBILITY.md` I9): two builds of
unchanged data produce identical files, so a rebuild diff shows only real changes.
Line coverage of a Plotly layout dict would tell you the code ran, not that the
chart is right.

The number is currently around 39%, and the two files holding it down are the era
parsers. That is the right place to aim a new test: parser bugs here have historically
been silent, and the committed data agrees with whatever the parser did.

Do not add a module to the omit list to make the number go up. Nothing published
depends on it, so the only thing that edit achieves is hiding untested code.

Separately, when you add a year, run the scraper field audit before trusting the
parsers on it:

```bash
python scraper/audit_review_fields.py --year 2026
```

It compares live pages against what the parser extracts and reports any form
question that is being dropped. The committed data cannot tell you this: a field
that was never captured leaves no trace in it. Four such bugs went unnoticed for
months for exactly that reason.

## Adding a new year

1. Add a block under `years:` in the root `config.yaml`. That one file feeds the
   scraper, the charts, and the site build, so there is nothing to keep in sync. If
   the page structure matches an existing era, reuse that parser; MICCAI has changed
   it three times so far, and `CLAUDE.md` documents the differences era by era.
2. Add the submission count to `num_papers_submitted.yaml`, the logo to
   `static/logos/` as a PNG of any size, its colours to `logo_colors.yaml`, and
   the host city. The site build resizes the logo and converts it to WebP, so
   there is no image step to remember; `python analysis/logo_assets.py` shows
   what it would write. `REPRODUCIBILITY.md` lists all of these as known manual
   inputs.
3. If a program book exists, drop it in `data/manually_downloaded/OralSchedules/` and
   add a block under `oral_schedules:` in the root `config.yaml`.
   `python analysis/extract_orals.py --dump YYYY` prints the parse without writing
   anything, which is the fastest way to tune the parameters.
4. Run stages 2 through 6 above.

**Verify before you trust it.** MICCAI's HTML has changed in ways that are invisible
until you check: review numbering has gaps, meta-reviews sometimes appear without their
usual wrapper, and the `early_accepted` signal is different in every era. Two earlier
heuristics for `early_accepted` looked right and were both wrong by roughly 20%. Sample
real pages before implementing, and check your counts against MICCAI's published
statistics where they exist.

## Conventions

- Follow `DESIGN.md` for anything the reader sees. It specifies colour tokens,
  typography, chart layout, and the Plotly theme. Charts should not set their own
  titles; the card around them carries the title.
- Formatting is enforced by `ruff` as configured in `pyproject.toml`.
- Prefer plain words. Write captions that say what a chart does *not* show as readily
  as what it does.
- Do not use em-dashes or en-dashes in prose. Semicolons, commas, and parentheses do
  the same work. Use the Oxford comma. This applies to prose only: a few regexes
  match dashes inside MICCAI's own text, and are written as `\u2014` / `\u2013`
  escapes so that a bulk replace cannot break them.
- When you learn something about MICCAI's page structure the hard way, write it into
  `CLAUDE.md`. Most of that file exists because someone got it wrong first.

## Pull requests

1. Open an issue first for anything larger than a fix, so we can agree on the approach.
2. Keep `website/` out of your diff; it is generated.
3. Say in the description what you ran to check the change. For a data change, include
   the before and after numbers.
4. If you changed anything about parsing or the pipeline, update the relevant
   documentation in the same pull request.

## A note on the data

These are real reviews of real papers, written by named authors' anonymous peers.
MICCAI published them, so analysing them is fair, but please keep the focus on the
process rather than on individuals. Aggregate statistics about how reviewing works are
the point. Ranking or singling out particular authors, reviewers, or papers as bad is
not, and pull requests doing that will be declined.
