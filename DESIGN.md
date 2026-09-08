# MICCAI Explorer: Design Specification

The visual system every page and chart follows: colour tokens, typography, layout,
the card and stat-strip components, and the Plotly theme. Read it before changing
any template, the CSS, or a chart's appearance.

---

## Fonts

Load from Google Fonts in every HTML page `<head>`:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
```

Apply globally: `font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;`

---

## Year Logos

Each MICCAI year has an official logo tied to its host city. The tracked source
files live in `static/logos/` as PNG files named `miccai_YYYY.png`;
`build_site.py` copies `static/` → `website/assets/` at build time (`website/` is
gitignored, so anything only placed there never reaches the CI deploy).

**File setup**; the human provides these files. Expected names:
```
static/logos/miccai_2021.png   (transparent background; Strasbourg; navy artwork)
static/logos/miccai_2022.png   (transparent background; Singapore)
static/logos/miccai_2023.png   (transparent background; Vancouver; red/black artwork)
static/logos/miccai_2024.png   (transparent background; Marrakesh)
static/logos/miccai_2025.png   (transparent background; Daejeon)
```

All five PNGs verified transparent (alpha channel checked 2026-07-13). The
white-pill wrappers (`.year-logo-pill` / `.nav-logo-pill`) remain in the templates
and CSS behind the per-year `has_white_bg` flag (currently `False` everywhere) in
case a future year ships a logo with a baked-in background.

**Primary placement; year page header (always implemented):**

Each year page (`year/YYYY.html`) shows the logo prominently between the nav bar and
the stat strip. The nav and logo header use a LIGHT background (`--nav-bg: #e9eff7`)
because several logos (2021 navy, 2023 red/black) are invisible on dark navy.

```html
<div class="year-logo-header">
  <img src="/assets/logos/miccai_2025.png"
       alt="MICCAI 2025 Daejeon" class="year-logo-img">
</div>

<!-- 2022 only: wrap in a white pill so the white-bg logo looks intentional -->
<div class="year-logo-header">
  <span class="year-logo-pill">
    <img src="/assets/logos/miccai_2022.png"
         alt="MICCAI 2022 Singapore" class="year-logo-img">
  </span>
</div>
```

```css
.year-logo-header {
  background: var(--nav-bg);   /* light; logos must stay visible */
  padding: 18px 24px 16px;
  display: flex;
  align-items: center;
}
.year-logo-img {
  height: 80px;
  width: auto;
  object-fit: contain;
}
.year-logo-pill {
  background: #ffffff;
  border-radius: 8px;
  padding: 6px 14px;
  display: inline-flex;
  align-items: center;
}
```

Top of each year page: `light nav → light logo header → dark stat strip → page body (light)`.
The stat strip is the only dark element and remains the signature block.

**Secondary placement; logos in the nav bar (implement this too):**

2021/2023/2024/2025 have transparent backgrounds and sit cleanly on the dark nav.
2022 gets the same white-pill treatment in the nav as in the page header.

```html
<!-- 2021, 2023, 2024, 2025; transparent, direct on dark nav -->
<a href="/year/2025.html" class="nav-year-link active"
   aria-label="MICCAI 2025">
  <img src="/assets/logos/miccai_2025.png"
       alt="2025" class="nav-logo">
</a>

<!-- 2022 only; white background, wrap in pill -->
<a href="/year/2022.html" class="nav-year-link"
   aria-label="MICCAI 2022">
  <span class="nav-logo-pill">
    <img src="/assets/logos/miccai_2022.png"
         alt="2022" class="nav-logo">
  </span>
</a>
```

```css
.nav-year-link {
  display: inline-flex;
  align-items: center;
  padding: 4px 6px;
  border-bottom: 2px solid transparent;
  text-decoration: none;
}
.nav-year-link.active { border-bottom-color: var(--nav-active); }
.nav-logo {
  height: 30px;
  width: auto;
  object-fit: contain;
  opacity: 0.65;
  transition: opacity 0.15s;
}
.nav-year-link:hover .nav-logo,
.nav-year-link.active .nav-logo { opacity: 1; }
.nav-logo-pill {
  background: #ffffff;
  border-radius: 4px;
  padding: 2px 6px;
  display: inline-flex;
  align-items: center;
}
```

`build_site.py` should check `has_white_bg` per year and use the pill wrapper
conditionally in the Jinja2 template.

**`build_site.py` must pass per-year metadata to templates:**
```python
YEAR_META = {
    2025: {"logo": "miccai_2025.png", "city": "Daejeon, Korea"},
    2024: {"logo": "miccai_2024.png", "city": "Marrakesh, Morocco"},
    2023: {"logo": "miccai_2023.png", "city": "Vancouver, Canada"},
    2022: {"logo": "miccai_2022.png", "city": "Singapore"},
    2021: {"logo": "miccai_2021.png", "city": "Strasbourg, France"},
}
```

---

## Color Tokens

Define as CSS variables in `static/style.css` (copied to `website/assets/` by
`build_site.py`):

```css
:root {
  --nav-bg:       #e9eff7;   /* light blue-gray; navigation + year logo header */
  --nav-border:   #d4deeb;   /* nav bottom border */
  --nav-text:     #1e293b;
  --nav-muted:    #475569;
  --nav-active:   #0891b2;   /* active year indicator; overridden per year page */

  --accent:       #0891b2;   /* teal default; overridden per year page */
  --accent-vivid: #22d3ee;
  --accent-bg:    #cffafe;   /* light tint of --accent; overridden per year page */
  --accent-on-dark: #22d3ee; /* accent legible on dark strip; overridden per year */

  --page-bg:      #f1f5f9;   /* light blue-gray; page background */
  --card-bg:      #ffffff;   /* white; chart containers */

  --text:         #1e293b;   /* dark slate; primary text */
  --text-muted:   #64748b;   /* medium slate; captions, labels */
  --text-faint:   #94a3b8;   /* light slate; very secondary info */

  --border:       #e2e8f0;   /* light slate; card borders, dividers */
  --border-inner: rgba(255,255,255,0.12);  /* inner borders on dark strip */

  --strip-bg:     #0f2744;   /* deep navy; stat strip only */
  --strip-accent: #0891b2;   /* 3px bottom border; overridden per year page */
}
```

---

## Per-Year Color Themes

Each year page is themed with that year's conference logo colors (source of truth:
`logo_colors.yaml`; first three entries per year are the primaries; the
`additional_avoid_if_possible` entries are only fallbacks). Black / near-white
primaries are skipped as data colors.

| Year | Main (`color`) | Secondary (`color2`) | Cross-year designated color |
|------|----------------|----------------------|------------------------------|
| 2025 | `#2b50a3` blue | `#d2242b` red        | `#692f90` purple |
| 2024 | `#2d835d` green| `#ce5258` rose       | `#2d835d` green  |
| 2023 | `#559e39` green| `#e81e25` red        | `#e81e25` red    |
| 2022 | `#23408f` navy | `#ed2425` red        | `#23408f` navy   |
| 2021 | `#27368a` indigo| `#00aeef` cyan      | `#00aeef` cyan   |

- **Year pages**: `year.html` emits a `<style>` block overriding `--accent`,
  `--accent-bg`, `--accent-on-dark`, `--nav-active`, `--strip-accent` with the year
  main color (defined in `YEAR_META_RAW` in `build_site.py`; derived tints computed
  with `_hex_mix`). Per-year charts use `YEAR_PALETTES` in `build_charts.py` -
  main for single-series charts, secondary for the paper-averages histogram.
- **Cross-year charts** (Overview page + `map_all`): every year keeps its one
  designated color from `YEAR_COLORS` in `build_charts.py` (right column above).
  These five are mutually CVD-distinguishable; bars always carry direct value
  labels since brand colors can't be re-tuned for contrast.

---

## Presentation Type Colors (Orals & Spotlights page)

The Orals page is the one place that deliberately does **not** use the per-year
palettes above. Its subject is the type, not the year, and a reader comparing
types across five years needs "oral" to look identical in every panel. Defined
once in `analysis/oral_stats.py` as `TYPE_COLORS`:

| Type | Color | Rationale |
|------|-------|-----------|
| Oral | `#7c3aed` violet | The primary series |
| Spotlight | `#0891b2` teal | Distinct from oral at a glance and CVD-safe against it |
| Poster only | `#94a3b8` slate | Deliberately recessive; it is the baseline the other two are measured against, not a third highlight |

Two charts on that page keep year identity instead, because the year *is* the
subject of the panel: the "share of accepted papers selected" bars in
`orals_overview` use `YEAR_COLORS`. The stratified forest plot
(`orals_strata`) uses its own three-way series colors (violet / green / amber)
since its series are strata, not types.

Effect-size charts shade the Romano et al. magnitude bands
(`rgba(148,163,184,0.13)` for negligible, `0.08` for small) behind the points, so
"this difference is too small to matter" is readable without consulting a legend.

`.methods-table` (in `style.css`) is visually identical to `.papers-table` but is
a separate class on purpose: `base.html` turns every `.papers-table` into a
collapsed "N papers" card, which is wrong for a small methods table.

---

## Layout

**Page max-width**: `1100px`, centered with `margin: 0 auto; padding: 0 24px;`

**Navigation** (fixed top, 50px tall):
```css
.site-nav {
  background: var(--nav-bg);
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  position: sticky;
  top: 0;
  z-index: 100;
}
```

The nav is light (`--nav-bg`), with `border-bottom: 1px solid var(--nav-border)`.

Logo text: left, `font-weight: 600; font-size: 14px; color: var(--nav-text);`

Year links: right, logo images `28px` tall at `opacity: 0.75` (1.0 on hover/active).
- Active year link: `border-bottom: 2px solid var(--nav-active)`; on year pages
  `--nav-active` is that year's color.
- "Overview & Trends" when active: `background: rgba(15,39,68,0.10); color: var(--nav-text); border-radius: 4px; border-bottom: none;`

**Page body** (below nav): `padding: 28px 24px;`

---

## Stat Strip (year pages only)

Appears between the nav and the page body on year pages. This is the signature element -
make it visually strong.

```css
.stat-strip {
  background: var(--strip-bg);
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border-bottom: 3px solid var(--strip-accent);
}
.stat-cell {
  padding: 18px 24px;
  border-right: 0.5px solid var(--border-inner);
}
.stat-cell:last-child { border-right: none; }
.stat-label {
  font-size: 10px;
  color: var(--nav-muted);
  text-transform: uppercase;
  letter-spacing: 0.07em;
  margin-bottom: 4px;
}
.stat-value {
  font-size: 28px;
  font-weight: 600;
  color: #fff;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
}
.stat-value.accent { color: var(--accent-on-dark); }  /* use for "% with code" */
```

**Four stat cells per year page**: Papers · Authors · % with code (accent) ·
Avg score (raw scale, shown as `x.xx / scale_max` with a muted `.stat-scale` suffix)

---

## Cards (chart containers)

Every chart lives inside a `.card`:

```css
.card {
  background: var(--card-bg);
  border: 0.5px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 14px;
}
.card-head {
  padding: 12px 16px;
  border-bottom: 0.5px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}
.card-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  margin: 0;
}
.card-sub {
  font-size: 11px;
  color: var(--text-muted);
  margin: 2px 0 0;
}
```

---

## Badges (chart toggles)

Small pill buttons in `.card-head` for chart view toggles (e.g., "By cluster" / "By year"):

```css
.badge {
  font-size: 11px;
  padding: 3px 9px;
  border-radius: 20px;
  font-weight: 500;
  cursor: pointer;
}
.badge-active { color: var(--accent); background: var(--accent-bg); }
.badge-muted  { color: var(--text-muted); background: var(--page-bg); }
```

These are cosmetic in the static site; they reflect the active view of the iframe beneath.
If needed for the map chart, the toggle triggers `postMessage` to the iframe.

---

## Chart iframes

All charts embedded as:
```html
<iframe src="/charts/CHARTNAME.html"
        class="chart-frame" scrolling="no" frameborder="0" loading="lazy">
</iframe>
```

```css
.chart-frame {
  width: 100%;
  border: none;
  display: block;
}
```

Frame heights on the **year pages are computed, not hardcoded**. `build_charts.py`
records what each horizontal bar chart actually drew (`_bar_height`: one row per bar,
an extra line per wrapped label, plus fixed chrome) into
`website/charts/chart_heights.json`; `build_site.py` reads that file and passes the
values to `year.html`. Hardcoded numbers went stale every time the type scale changed,
and left paired charts at visibly different heights.

Two pairs share a row and are therefore forced to a common height, the taller of the
two (`row_heights` in `build_site.py`):

- `subjects_YYYY` with `authors_YYYY`
- `code_YYYY` with `buzzwords_YYYY`

The remaining year-page frames are fixed, in `_STATIC_HEIGHTS`: map `520`, score and
controversy histograms `380`, rebuttal Sankey `460`, co-authorship network `600`,
naming subplots `420`. If `chart_heights.json` is missing, `build_site.py` falls back
to these plus a shared `640`, so it still runs standalone.

Heights elsewhere are still inline in the template: semantic map `620` on the overview,
`trends_overview` `660`, `subject_lines` `680`, `subject_movers` `580`; and on the
orals page overview `420`, confidence `430`, scores / early / code `470`, effect `560`,
strata `520`, areas `620`, map `620`.

Most charts fill their frame, but three wrap hand-written HTML with an intrinsic
height and *will* clip if the frame is too short: `trends_overview` (a CSS grid),
`subject_movers`, and `coauthor_YYYY` under the d3 renderer. After changing any font
size, re-measure: load each page and compare
`iframe.contentDocument.documentElement.scrollHeight` against `iframe.clientHeight`;
they should be equal.

---

## Grid layouts

Two-column (use for pairs of related charts):
```html
<div class="grid2">
  <div class="card">...</div>
  <div class="card">...</div>
</div>
```
```css
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 700px) { .grid2 { grid-template-columns: 1fr; } }
```

Two-thirds / one-third row (`.grid-2-1`, `grid-template-columns: 2fr 1fr`); used
for Review Score Distribution (two subplots) beside Reviewer Disagreement (one plot)
so all three plots render at equal width.

Four-column stat grid:
```css
.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}
@media (max-width: 700px) { .stat-grid { grid-template-columns: repeat(2, 1fr); } }
```

---

## Small stat cards (index page only)

```css
.stat-card {
  background: var(--card-bg);
  border: 0.5px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}
.stat-card-label {
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 4px;
}
.stat-card-value {
  font-size: 24px;
  font-weight: 600;
  color: var(--text);
  font-variant-numeric: tabular-nums;
}
.stat-card-note {
  font-size: 11px;
  color: var(--accent);
  margin-top: 4px;
}
.stat-card-note.muted { color: var(--text-muted); }
```

---

## All Papers page components

These live in **`static/papers.css`**, not `style.css`, and are linked only from
`papers.html`. Nothing else on the site uses them, and keeping them separate means
this page cannot change the appearance of any other. The table itself reuses
`.papers-table` from `style.css`, including the presentation-type badges, so the
two read as one scheme.

**This page keeps the default teal accent** rather than taking a year theme,
because it spans all five years at once.

| Component | Class | Specification |
|---|---|---|
| Search field | `.papers-search` | full width, 16px, 10px/14px padding, 6px radius, `--border`. On focus: `--accent` border plus a 3px `--accent-bg` ring, and the default outline removed. |
| Filter row | `.papers-filters` | flex, wraps, 14px gap, bottom-aligned. Each control is a `.papers-filter` column with the label above it. |
| Filter label | `.papers-filter > span` | 13px, 600, uppercase, `0.06em` letter-spacing, `--text-muted`. Matches the stat-card label treatment. |
| Dropdown | `.papers-filter select` | 15px, 7px/10px padding, capped at 260px so a long subject-area name cannot stretch the row; the area one gets 380px via `.papers-filter-wide`. |
| Result count | `.papers-count` | 15px, `--text-muted`, left of the bar; sort and export sit right. |
| Buttons | `.papers-button` | 15px, `--card-bg` on `--border`, 6px radius; on hover the border and text go `--accent`. |
| Score cell | `.score-pct` | the normalized percent in `--text-faint`, 6px after the raw score. Always secondary to the raw number. |
| Expanded row | `.detail-row`, `.paper-detail` | a `<dl>` on `--page-bg`, two columns (`max-content 1fr`), 6px/18px gap. `dt` uses the same uppercase label treatment as the filters. |
| Link chips | `.link-chip` | pill (999px radius), 14px, `--accent` text on `--card-bg`; hover fills with `--accent-bg`. Only drawn for links the paper actually has. |
| Pagination | `.papers-pagination` | centred flex, 38px minimum button width so the row does not jump between single and double digits. The current page is filled with `--accent` and white text. |

**Responsive, single breakpoint at 720px:** the subject-area column is hidden
(`.papers-browse .subject-col { display: none; }`), the expanded row collapses to
one column, and the result bar left-aligns instead of spreading. There is no
second breakpoint; the controls already wrap on their own.

**Row height is deliberately not fixed.** Titles wrap to two or three lines and
subject-area names to two, and forcing a uniform height would truncate them. The
table is capped at 100 rows a page, so the page stays a predictable length anyway.

---

## Type scale

The site was originally set much smaller than this and was hard to read on a normal
display. It was raised twice during 2026-08. **The floor is 14px in CSS and 13px in
charts**; nothing on the site is smaller than that. Everything else is scaled from the
floor, except the two large display numbers, which are lifted less so they keep fitting
their cards.

| Role | Size |
|---|---|
| Body text, card body, nav brand | 18px |
| Card subtitle / caption, table cells | 16px |
| Card title | 17px |
| Section heading | 21px |
| Stat card label, table header | 14px |
| Stat card value | 30px |
| Stat strip value | 34px |
| Year title | 23px |

Chart type is set in Python, not CSS: base 17px, hover labels 15.5px, axis ticks 13px
to 14.5px, legends 14.5px, and value labels 13px to 14px. The one deliberate exception
is the co-authorship network's node labels at 12px, because up to 25 of them are drawn
at once on a crowded canvas and larger text overlaps. In `build_oral_charts.py` the
two sizes tuned against frame height are named constants (`PIN_SIZE` and
`LEGEND_SIZE`) rather than inline numbers.

Larger type makes wrapped tick labels collide before it makes anything else break, so
after any change to these values, rebuild and re-measure the frames (see below).

## Plotly chart styling

All Plotly charts must be styled to match the site. Pass these in `base_layout()` inside
`analysis/build_charts.py`:

```python
BG       = "#ffffff"
PLOT_BG  = "#f8fafc"   # very slightly tinted chart area (not pure white)
TEXT     = "#1e293b"
MUTED    = "#64748b"
RULE     = "#e2e8f0"
ACCENT   = "#0891b2"
FONT     = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"

def base_layout(**over):
    title = over.pop("title", None)
    lay = dict(
        paper_bgcolor=BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT, size=17, color=TEXT),
        margin=dict(l=60, r=30, t=60, b=50),
        hoverlabel=dict(font=dict(family=FONT, size=15.5), bgcolor="#fff", bordercolor=RULE),
    )
    lay.update(over)
    if title:
        lay["title"] = dict(text=title, font=dict(size=19, color=TEXT),
                            x=0.02, xanchor="left", y=0.97, yanchor="top")
    return lay
```

Cluster scatter colors (for both the all-years map and per-year maps); use this
20-color palette, cycling if k > 20:
```python
CLUSTER_PALETTE = [
    "#0891b2", "#7c3aed", "#d97706", "#059669", "#e11d48",
    "#f59e0b", "#0284c7", "#9333ea", "#16a34a", "#dc2626",
    "#0369a1", "#6d28d9", "#b45309", "#047857", "#be123c",
    "#0e7490", "#5b21b6", "#92400e", "#065f46", "#9f1239",
]
```

**No in-chart titles.** Every chart is embedded in a `.card` whose head already
carries the title + subtitle, so chart figures must NOT set `layout.title`
(use a small top margin, `t≈20-40`). Subplot titles (e.g. "Individual Reviewer
Scores") are fine; they are not duplicates.

**Distribution charts** (review scores, paper averages, disagreement, title length)
are histogram bars + a smooth density overlay (seaborn `histplot(kde=True)` look),
built with `_hist_kde_traces()` in `build_charts.py`: numpy-only Gaussian KDE
(Silverman bandwidth with a floor; integer-quantized data otherwise yields a spiky
comb), bars at 0.55 opacity of the series color, 2.5px spline line in the same hue,
KDE scaled to counts (`density × n × binwidth`). Review scores use the RAW per-year
scale (1-6 / 1-8 / 1-9), never normalized.

Review Analysis row color pattern is **1-1-2**: both Review Score Distribution
panels use the year MAIN color, Reviewer Disagreement uses the SECONDARY. Both
score panels overlay early-accepted papers as a second series: same hue darkened
~45% (`_darken`), narrower bars (0.45× width) at 0.9 opacity drawn in front, with
its own KDE; horizontal legend above the subplot titles. The averages panel is
x-clipped to the data range (full 1-max leaves dead space).

**Years at a Glance (index)**: 2×3 bar subplots, one designated `YEAR_COLORS` color
per year, direct value labels, plus a thin gray connector line (+small markers)
along the bar tops of each panel so trend direction reads at a glance
(`rgba(100,116,139,0.65)`, hover disabled).

**Rebuttal Sankey** (2024/2025 only; other years have no post-rebuttal verdict
labels): pre-rebuttal verdicts left, post-rebuttal right, best verdict at top.
Node colors are semantic verdict colors (`VERDICT_COLORS`: reds for reject side,
amber for weak reject, greens for accept side); links tinted by direction -
green improved, red downgraded, gray unchanged; N/A responses excluded.

**Top Papers tables** are plain HTML (`.papers-table`), not Plotly: small text
(12px), uppercase muted headers on `--page-bg`, hover row highlight, green
`✓` / red `✗` for early-accept, means bold with the raw score list muted
in parentheses.

---

## Footer

Simple, minimal, on every page:

```html
<footer class="site-footer">
  <p class="footer-links">
    MICCAI Explorer by <a href="https://kabhishe.com">Kumar Abhishek</a>
    <span class="footer-sep">&middot;</span>
    <a href="/about.html">About</a>
    <span class="footer-sep">&middot;</span>
    <a href="{{ repo_url }}"><svg class="gh-mark">…GitHub mark…</svg>GitHub</a>
  </p>
  <p>MICCAI papers and reviews © respective authors.
     <span class="footer-sep">&middot;</span>
     Data sourced from <a href="https://papers.miccai.org">papers.miccai.org</a>.
     <span class="footer-sep">&middot;</span> Last updated {{ build_date }}.
  </p>
</footer>
```

The links line sits above the attribution and is set slightly stronger than it
(`font-weight: 500`). The GitHub mark is inlined as an SVG rather than loaded
from a CDN, sized in `em` and filled with `currentColor`, so it tracks the link
text and takes the year accent on year pages. About and the repository are in
the footer rather than the nav because
the nav is one fixed-height row that does not wrap and already holds seven
entries. Below 760px the nav entries move into the hamburger panel, so the
footer is no longer the only navigation on a phone, but it stays the only
place About is reachable at any width. The source link is skipped entirely
when `site_settings.repo_url` is unset.

```css
.site-footer {
  background: var(--page-bg);
  border-top: 0.5px solid var(--border);
  padding: 20px 24px;
  margin-top: 40px;
  text-align: center;
  font-size: 12px;
  color: var(--text-muted);
}
.site-footer a { color: var(--accent); text-decoration: none; }
.site-footer a:hover { text-decoration: underline; }
```

---

## Page section headings (on long year pages)

Use a left-accent style to visually separate analysis sections:

```css
.section-heading {
  font-size: 16px;
  font-weight: 600;
  color: var(--text);
  margin: 32px 0 12px;
  padding-left: 12px;
  border-left: 3px solid var(--accent);
}
```

Example usage:
```html
<h2 class="section-heading">Review Analysis</h2>
```

---

## Responsive

The site is desktop-first (researchers use large monitors), but it must be
*correct* at every width, not merely unbroken in principle. It was not: until
2026-09-03 every page overflowed below about 1000px, and on a phone Chrome for
Android scaled the whole site down to 45% to make the overflow fit. The
section above this one used to specify `.nav-links { display: none }` with a
"TODO: add hamburger menu later" beside it. The TODO is done; the numbers
below are measured, not chosen, and should be re-measured rather than rounded.

### The rule that matters

**Nothing may be wider than the viewport unless it sits in a container that
scrolls on purpose.** One element that breaks this does not misplace itself,
it rescales the entire page: the nav was 668px against a 360px viewport, and
that alone is what made the body render at 45% width with a blank half screen.

### Breakpoints

Two thresholds, both derived from what the nav actually requires:

| Width | Nav | Layout | Nav requires |
|---|---|---|---|
| >= 1021px | Full labels, 28px logos | Two-column grids | 1002px |
| 761-1020px | "Overview", "Orals", "Papers"; 22px logos | Grids to one column | 660px |
| <= 760px | Brand + hamburger; entries in a panel | Stat grids to 2 then 1 | ~200px |

The nav requirement at full size is brand 140 + "Overview & Trends" 163 +
"Orals & Spotlights" 154 + five logos 374 + "All Papers" 96 + gaps and padding
76 = 1002. Adding an eighth nav entry means re-measuring this table.

Two-column chart grids collapse at 1020px rather than at the phone
breakpoint, because a `.grid2` cell is only 451px at a 1000px viewport, which
is already too narrow for the horizontal bar charts' label margins.

### Tap targets

Every nav entry fills the full 50px nav height, and panel rows are at least
44px. A 22px logo with 4px padding would be a 30px target, under both Google's
and Apple's guidance.

### Charts

Plotly margins are in pixels, so a chart laid out for a 1050px frame has no
room left in a 294px one. Charts are **not** rebuilt per breakpoint. Each
chart page carries a shared script (`RESPONSIVE_JS` in `build_charts.py`,
injected by both `save_chart` and `save_template`) that reads its own frame
width at runtime and:

- scales the figure's margins by how much narrower the frame is, keeping the
  plotting region at least 55% of the frame;
- moves a vertical legend below the plot under 760px, where there is no room
  for it beside;
- hides the modebar under 760px, since its targets are unhittable by finger
  and it overlaps the charts' own controls;
- reports the page's real height to the parent, which sizes the iframe.

- shortens category tick labels, and hands the margin back to `automargin`;
- thins, shortens or re-steps colliding numeric ticks, per cause;
- drops an axis title that cannot fit at a legible size;
- re-lays a subplot grid into fewer columns and grows the frame to suit.

That last point is why `chart_heights.json` is a starting height rather than
the truth: a chart page can reflow in ways the build cannot predict.

### Two rules the runtime script must keep

Both were broken once, and each broke several charts at a time.

**1. Anything that reads the layout and then writes it must read a snapshot.**
The script runs four to six times per page as Plotly settles. `gridInfo()` read
the subplot grid live: pass one re-laid a 3x4 grid into one column and wrote the
domains, pass two read those back, concluded the figure had one column, and
skipped the grid block and the height it claims, at which point the height set
by pass one was actively cleared. `subject_lines` drew twelve 25px panels in a
660px box under a 2136px frame. The snapshots are `_orig`, `_gridSnap`,
`_annSnap` and `_titleMap`; the same trap already had its own note on `dtick`
and on hidden annotations, which is exactly why the rule is stated here once.

**2. Anything positioned above or below a plot in paper units must be placed in
pixels once the figure's height changes.** Paper units are fractions of the
*plot* height, and the plot height is whatever the margins leave. Growing a
margin to make room therefore shrinks the plot and moves the thing back by most
of what was gained, so it never converges. Three cases, all real:

| What | Built as | Became | Fix |
|---|---|---|---|
| Sankey headline | `y = 1.15` | 35px above the frame's top edge | `yshift`, anchored at `y = 1` |
| `scores_*` legend | `y = 1.35` (91px) | 369px once the panels stack | rescale by old plot height / new |
| map legend, moved below | `y = -0.12` | a 104px channel under the map | compute `y` from a 22px target |

Annotations have `yshift` and legends do not, which is the only reason the two
are handled differently.

### A subplot heading is not a value label

They are indistinguishable in the layout: both are paper-referenced annotations
sharing a y. The rule that hides a row of three or more value labels too wide to
fit (an overprinted smear says less than nothing, and every value is still in
the hover) matched `subject_lines`' twelve panel headings, three per row in its
3-column build, and took nine panel names off the chart. Headings are matched to
their panel once, from the snapshot, and are **never hidden**: they shrink, then
wrap, to their own panel's width. A heading is also exempt from the top-margin
reservation, since for every panel but the first "above the plot" is the middle
of the figure.

### Desktop must be the figure as built

Above the breakpoint the script's job is to change nothing. It has to keep
running, because a window can be dragged across the breakpoint and everything
the narrow path sets must be put back, but every restore takes its value from
the snapshot rather than from the current layout, which holds this script's own
last output. `regrid` used to run even on an unchanged grid, deriving its gaps
from its own constants instead of the figure's `horizontal_spacing` and
`vertical_spacing`; all twelve of `subject_lines`' domains were wrong at 1280px.
An unchanged grid now restores the build's domains verbatim, and that is worth
checking after any change here: the rendered domains should equal the ones in
the figure JSON.

### Tick labels: shorten them, do not fight the margin

Plotly gives two ways to size the space a category axis gets, and on a narrow
frame **both fail, in opposite directions**:

| Setting | What Plotly does | Result |
|---|---|---|
| `automargin: true` | Grows the margin to fit the labels | Bars squeezed to nothing. Measured: `margin.l` set to 66, Plotly used **206** of a 344px frame |
| `automargin: false` | Leaves the margin alone and clips | Labels cut off. `subject_movers` sliced "Semi-/Weakly-/Self-supervised Learning" in half |

So the margin is not the variable to control. **Shorten the label**, then let
`automargin` size the margin around it. The full name stays in the hover, which
is the bargain `_trunc_ticks` already makes on the desktop layout.

Estimating the margin from an average character width was tried and is wrong by
19 to 33px: the budget has to be satisfied by the *widest* label, and an average
is not the widest.

### Colliding ticks have three causes and three fixes

Do not reach for one blanket answer. Clearing `tickvals` was tried and broke
four charts, because several place their ticks deliberately (CLAUDE.md records
the authors chart's "multiples of 5 plus the max"; the orals charts label bar
group centres), and Plotly then auto-ticked a `-0.5 … 4.5` range into
`0.01, 0.02, -0.49`.

| Cause | Fix |
|---|---|
| Year labels too wide (`2021` vs `2022`) | Shorten to `’21`. Never thin: dropping years from a five-year comparison loses the point of the chart |
| Deliberate `tickvals` | Keep every k-th |
| Explicit `dtick` (one tick per score point) | Widen the step, so ticks stay on whole values |
| Automatic ticks | `nticks` |

**Read `dtick` from the user layout, never from `_fullLayout`.** Plotly stores
its own computed step there; restoring that as an explicit setting asked for a
tick every 0.02 across a five-unit range.

### The Sankey is the one exception to "fit the screen"

Its information is horizontal by construction: two labelled columns and the
flows between them. Below 520px it renders at 520px inside its own
`overflow-x` container and scrolls. The check exempts anything inside such a
container, so this is a declared exception rather than a hole in the rule.

Its scroll decision reads the **frame** width, not the plot's own width.
Keying it off the plot oscillated: widening the plot to 520 made the next pass
conclude no scrolling was needed and reset it, 24 times over.

**A chart page must never size itself from the viewport height.**
`subject_movers` and `orals_areas` used `height: calc(100vh - 42px)` against a
control row that wraps to 58px on a phone, so the content was always 16px
taller than the frame; once frames size themselves from reported content, that
is an unbounded growth loop. Use flex (`body{display:flex;flex-direction:
column}` with `flex:1` on the plot) so the content cannot exceed the frame.

### Verification

`tests/check_responsive.py` loads every page at 360, 390, 768 and 1000px and
fails on any element wider than the viewport outside a scroll container, any
chart iframe whose content exceeds its frame, or any plot whose region is
under 45% of its frame. Run it after touching this file, `style.css`, or any
chart layout. Chrome DevTools device mode reproduces the page-scaling
behaviour; it does not reproduce touch scrolling inside an iframe.
