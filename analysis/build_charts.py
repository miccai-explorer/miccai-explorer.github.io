#!/usr/bin/env python3
"""
Generate all Plotly chart HTML files → website/charts/

Per-year charts (for each year in data):
  map_YYYY.html, subjects_YYYY.html, scores_YYYY.html, controversy_YYYY.html,
  rebuttal_YYYY.html, code_YYYY.html, authors_YYYY.html, coauthor_YYYY.html,
  buzzwords_YYYY.html, naming_YYYY.html

Cross-year charts (index page):
  map_all.html, subject_heatmap.html

Settings:
  config.yaml -> chart_settings.network_renderer selects how the co-authorship
  networks are drawn: "d3" (draggable nodes, default) or "plotly" (static figure).

Usage:
    python analysis/build_charts.py
    python analysis/build_charts.py --years 2025 2024
"""

import argparse
import contextlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

ALL_JSON = Path("data/processed/miccai_all.json")
CL_JSON = Path("data/processed/cluster_labels.json")
CONFIG = Path("config.yaml")
SUBS_YAML = Path("num_papers_submitted.yaml")
OUT_DIR = Path("website/charts")


def load_year_config() -> dict:
    """Per-year settings (review_scale_max, coauthorship_network_min_papers, ...)."""
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f).get("years", {})


def load_chart_settings() -> dict:
    """Global chart settings from config.yaml -> chart_settings:.

    network_renderer: "d3" (default) or "plotly"
      d3     - interactive: nodes can be dragged, background pans, scroll zooms.
      plotly - legacy static figure: pan + zoom only, no per-node dragging.
    """
    defaults = {"network_renderer": "d3"}
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    settings = {**defaults, **(cfg.get("chart_settings") or {})}
    if settings["network_renderer"] not in ("d3", "plotly"):
        logger.warning(
            f"  Unknown network_renderer "
            f"{settings['network_renderer']!r}; falling back to 'd3'"
        )
        settings["network_renderer"] = "d3"
    return settings


def load_submissions() -> dict:
    """{year: number of papers submitted} from num_papers_submitted.yaml."""
    if not SUBS_YAML.exists():
        return {}
    with open(SUBS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Design tokens (from DESIGN.md)
# ---------------------------------------------------------------------------
BG = "#ffffff"
PLOT_BG = "#f8fafc"
TEXT = "#1e293b"
MUTED = "#64748b"
RULE = "#e2e8f0"
ACCENT = "#0891b2"
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"

CLUSTER_PALETTE = [
    "#0891b2",
    "#7c3aed",
    "#d97706",
    "#059669",
    "#e11d48",
    "#f59e0b",
    "#0284c7",
    "#9333ea",
    "#16a34a",
    "#dc2626",
    "#0369a1",
    "#6d28d9",
    "#b45309",
    "#047857",
    "#be123c",
    "#0e7490",
    "#5b21b6",
    "#92400e",
    "#065f46",
    "#9f1239",
]

# One distinctive color per year for cross-year charts (drawn from each year's
# conference logo primaries: see logo_colors.yaml). Bars using these always carry
# direct value labels or an axis year label, since brand colors can't be re-tuned.
YEAR_COLORS = {
    2021: "#00aeef",  # Strasbourg cyan
    2022: "#23408f",  # Singapore navy
    2023: "#e81e25",  # Vancouver red
    2024: "#2d835d",  # Marrakesh green
    2025: "#692f90",  # Daejeon purple
}

# Per-year page palette: main + secondary chart colors from that year's logo primaries
# (black / near-white primaries are skipped: unusable as data colors).
YEAR_PALETTES = {
    2021: {"main": "#27368a", "secondary": "#00aeef"},
    2022: {"main": "#23408f", "secondary": "#ed2425"},
    2023: {"main": "#559e39", "secondary": "#e81e25"},
    2024: {"main": "#2d835d", "secondary": "#ce5258"},
    2025: {"main": "#2b50a3", "secondary": "#d2242b"},
}


def year_main(year: int) -> str:
    return YEAR_PALETTES.get(year, {}).get("main", ACCENT)


def year_secondary(year: int) -> str:
    return YEAR_PALETTES.get(year, {}).get("secondary", "#7c3aed")


def _darken(color: str, t: float = 0.4) -> str:
    """Blend a '#rrggbb' color toward black by fraction t."""
    c = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(v * (1 - t)):02x}" for v in c)


BUZZWORDS = {
    "Segmentation": [r"\bsegment"],
    "Diffusion models": [
        r"diffusion model",
        r"\bddpm\b",
        r"score.based generat",
    ],
    "Foundation models": [r"foundation model"],
    "Transformers / ViT": [r"\btransformer\b", r"\bvit\b", r"\bswin\b"],
    "Self-supervised / SSL": [r"self.?supervised", r"\bcontrastive learn"],
    "Semi-supervised": [r"semi.?supervised"],
    "Domain adaptation": [r"domain adapt"],
    "Federated learning": [r"\bfederated\b"],
    "Vision-language / VLM": [r"vision.?language", r"\bvlm\b", r"\bclip\b"],
    "Large language model": [r"\bllm\b", r"large language model"],
    "Generative / GAN": [r"\bgenerative\b", r"\bgan\b"],
    "Image registration": [r"\bregistrat"],
    "Image reconstruction": [r"\breconstruct"],
    "Detection": [r"\bdetect"],
    "Classification": [r"\bclassif"],
    "Uncertainty": [r"\buncertain"],
    "Weakly supervised": [r"weakly.?supervised"],
    "Active learning": [r"active learn"],
    "Surgical AI": [r"\bsurgical\b", r"\blaparoscop", r"\bendoscop"],
    "Pathology / WSI": [r"\bpatholog", r"whole.?slide", r"\bwsi\b"],
    "Explainability": [r"\bexplainab", r"\binterpretab", r"\bxai\b"],
    "Multimodal": [r"multi.?modal"],
    "Mamba / SSM": [r"\bmamba\b", r"state.?space model"],
    "Segment Anything / SAM": [r"\bsam\b", r"segment anything"],
}

VERDICT_ORDER = [
    "Strong Reject",
    "Reject",
    "Weak Reject",
    "Weak Accept",
    "Accept",
    "Strong Accept",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def base_layout(**over):
    title = over.pop("title", None)
    lay = dict(
        paper_bgcolor=BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT, size=17, color=TEXT),
        margin=dict(l=60, r=30, t=60, b=50),
        hoverlabel=dict(
            font=dict(family=FONT, size=15.5, color=TEXT),
            bgcolor="#fff",
            bordercolor=RULE,
        ),
    )
    lay.update(over)
    if title:
        lay["title"] = dict(
            text=title,
            font=dict(size=19, color=TEXT),
            x=0.02,
            xanchor="left",
            y=0.97,
            yanchor="top",
        )
    return lay


def _config(interactive: bool = False) -> dict:
    """Plotly config for save_chart / manual saves.
    interactive=True: modebar appears on hover, scroll-to-zoom enabled.
    interactive=False: modebar hidden (static charts; bar, histogram, heatmap).
    """
    if interactive:
        return {
            "responsive": True,
            "displayModeBar": "hover",
            "scrollZoom": True,
        }
    return {"responsive": True, "displayModeBar": False}


# Clicking a point opens that paper. Used by every map chart; the div is found
# by class rather than by id, so a stable div_id (below) does not affect it.
CLICK_JS = """
<script>
document.addEventListener('DOMContentLoaded', function() {
  var div = document.querySelector('.plotly-graph-div');
  if (!div) return;
  div.on('plotly_click', function(data) {
    var url = data.points[0].customdata[0];
    if (url) window.open(url, '_blank');
  });
});
</script>"""


# ---------------------------------------------------------------------------
# Responsive behaviour, injected into every chart page
# ---------------------------------------------------------------------------
# Plotly margins are in pixels. A chart laid out with margin=dict(l=200) reads
# fine in a 900px frame and has no room left in a 294px one; margin=dict(l=20,
# r=300) on the semantic map produced a plotting area TWO PIXELS wide on a
# phone, with the legend drawn across the whole frame. Measured 2026-09-03.
#
# The fix is at runtime, not at build time: an iframe knows its own width, and
# the build does not. So this scales the margins the figure was built with by
# how much narrower the frame actually is, and moves a vertical legend
# underneath the plot when there is no room beside it.
#
# It is injected by save_chart AND save_template for every file, not passed in
# per call. That is deliberate. CLICK_JS is opt-in and orals_map.html shipped
# for months without it, drawing and hovering perfectly while doing nothing on
# click, because a chart that forgets an opt-in enhancement looks completely
# fine. Anything that every chart needs goes in the default path.
#
# The height message exists because a chart page can reflow in ways the build
# cannot predict: trends_overview.html goes from a 3x3 grid to two columns
# under 820px and needs 1187px against the 660px recorded for it, so three
# panels were being cut off. The page measures itself and tells the parent.
# base.html listens. chart_heights.json stays as the starting height.
#
# Both halves no-op safely: the Plotly half finds no .js-plotly-plot on the d3
# co-authorship pages (which load no Plotly at all) and does nothing; the
# height half runs everywhere.
RESPONSIVE_JS = """
<style>
/* The modebar is a desktop affordance: a row of 20px icons pinned to the top
   right of the plot. On a phone it lands on top of the chart's own controls
   (at 360px it covers the map's "Color by Year / Color by Cluster" buttons),
   its targets are far too small to hit with a finger, and everything it offers
   is available by touch anyway: the maps set scrollZoom and dragmode="pan", so
   pinch and drag already work. Hidden rather than repositioned because there
   is no room on a 360px frame to put it that does not cover something. */
@media (max-width: 760px) {
  .modebar { display: none !important; }
}
</style>
<script>
(function () {
  // Matches the phone tier in static/style.css. Kept in step by hand; there is
  // no shared source for a number CSS and JS both need.
  var NARROW = 760;
  // The width these charts were laid out for: the .page-body max-width of
  // 1100px less its padding, which is what a chart frame gets on a desktop.
  var REF = 1050;
  // Leave the plotting region at least this share of the frame. The check in
  // tests/check_responsive.py fails below 0.45, so this is the same rule
  // expressed as an action instead of an assertion.
  var MIN_PLOT_SHARE = 0.55;
  var ELL = "\u2026";

  function axisNames(fl, letter) {
    var out = [];
    for (var k in fl) {
      if (k.indexOf(letter + "axis") === 0 && fl[k] && fl[k]._id) out.push(k);
    }
    return out;
  }

  // What the build asked for, captured once. Every later pass recomputes from
  // THIS, never from the result of a previous pass, or repeated resizes ratchet
  // margins and font sizes down to nothing.
  // Plotly parks a placeholder in _fullLayout for an axis with no title:
  // _fullLayout.xaxis.title.text reads "Click to enter X axis title" on every
  // one of subject_lines\' twelve untitled axes. It is never drawn outside the
  // chart editor, but reading it back treats an untitled axis as titled, and
  // that has to be undone once rather than guarded at each of the three places
  // that ask. It cost 34px of row gap per stacked panel here, taken straight
  // out of the panels; the wide branch would also have written the placeholder
  // back as an explicit title, which Plotly happens to re-detect and skip.
  function realTitle(gd, fa) {
    var txt = (fa && fa.title && fa.title.text) || "";
    if (!txt) return "";
    var dflt = gd._fullLayout._dfltTitle;
    if (dflt) {
      for (var k in dflt) { if (dflt[k] === txt) return ""; }
    }
    return /^Click to enter .* title$/.test(txt) ? "" : txt;
  }

  function orig(gd) {
    if (gd._orig) return gd._orig;
    var m = gd._fullLayout.margin || {};
    var o = {m: {l: m.l || 0, r: m.r || 0, t: m.t || 0, b: m.b || 0},
             legend: gd.layout.legend ? JSON.parse(JSON.stringify(gd.layout.legend)) : null,
             legendY: (gd._fullLayout.legend && gd._fullLayout.legend.y),
             // The frame height the figure was laid out in, needed to read a
             // paper-unit offset back as the pixel gap it was meant to be.
             h0: gd.getBoundingClientRect().height || 0,
             showlegend: gd.layout.showlegend, ax: {}};
    var names = axisNames(gd._fullLayout, "x").concat(axisNames(gd._fullLayout, "y"));
    for (var i = 0; i < names.length; i++) {
      var n = names[i], fa = gd._fullLayout[n], ua = (gd.layout[n] || {});
      o.ax[n] = {
        tickfont: (fa.tickfont && fa.tickfont.size) || 14,
        titlefont: (fa.title && fa.title.font && fa.title.font.size) || 14,
        titletext: realTitle(gd, fa),
        ticktext: ua.ticktext ? ua.ticktext.slice() : null,
        tickvals: ua.tickvals ? ua.tickvals.slice() : null,
        automargin: ua.automargin !== undefined ? ua.automargin : fa.automargin,
        // From the USER layout, never from _fullLayout. Plotly stores its own
        // computed step there: on the orals charts that is 0.02, and restoring
        // that as an explicit setting asks for a tick every 0.02 across a
        // -0.5 to 4.5 range, i.e. 250 of them on top of each other. Only a
        // dtick the build actually asked for should ever be restored.
        dtick: ua.dtick,
        categories: (fa._categories || []).slice(),
        type: fa.type
      };
    }
    gd._orig = o;
    return o;
  }

  // Width per character on one axis, taken as the WIDEST label's, not the
  // average. The budget has to be satisfied by the longest label, and an
  // average under-truncates it: measured on buzzwords at 278px, the average
  // allowed a label that rendered 109px wide against an 83px budget.
  function tickMetrics(gd, axId) {
    var sel = gd.querySelectorAll("." + axId + "tick text");
    var perChar = 0;
    for (var i = 0; i < sel.length; i++) {
      var el = sel[i], s = (el.textContent || "");
      if (!s.length) continue;
      var w = 0;
      try { w = el.getComputedTextLength(); } catch (e) { w = el.getBBox().width; }
      if (!w) continue;
      var pc = w / s.length;
      if (pc > perChar) perChar = pc;
    }
    return {perChar: perChar || 7};
  }

  function truncate(s, chars) {
    s = String(s).split("<br>").join(" ");
    if (s.length <= chars) return s;
    return s.slice(0, Math.max(1, chars - 1)).replace(/\\s+$/, "") + ELL;
  }

  // Re-lay a subplot grid into fewer columns.
  //
  // subject_lines draws 12 panels in 3 columns. At 344px that is about 100px
  // per panel, and the panel titles ("Computer Aided Diagnosis") overprint each
  // other into a single unreadable line. No font size fixes that; the grid has
  // to get narrower and taller. The column count is baked into the axis domains
  // at build time, so it is rewritten here instead.
  //
  // Panels are paired by axis suffix (xaxis3 with yaxis3) and then sorted into
  // visual reading order from their domains, rather than trusting Plotly's
  // numbering to match the layout.
  // The subplot grid AS THE BUILD WROTE IT, captured once, exactly like orig().
  //
  // This used to read gd._fullLayout on every pass, and that one line caused
  // every regression in screenshotsV2. The first pass re-lays a 3x4 grid into
  // one column and writes the new domains; the second pass then reads those
  // domains back, concludes the figure has one column, and skips the whole
  // grid block, including the height it claims. wantH goes null, the
  // "no claim" branch clears gd.style.height, and the figure snaps back to its
  // desktop height with twelve stacked panels still inside it: subject_lines
  // drew twelve 25px panels in a 660px box under a 2100px frame, which is the
  // squashed strip and the screen of white space below it. scores_* and
  // naming_* are the same failure with two panels.
  //
  // Anything that both reads the layout and writes it has to read a snapshot.
  // Caching null matters as much as caching a grid: a single-panel figure must
  // not be re-examined once its axes have been touched.
  function gridInfo(gd) {
    if (gd._gridSnap !== undefined) return gd._gridSnap;
    var fl = gd._fullLayout;
    var xs = axisNames(fl, "x");
    // Two is enough to be worth stacking: scores_* puts two histograms side by
    // side, and at 344px their axis titles overprinted each other.
    if (xs.length < 2) { gd._gridSnap = null; return null; }
    var panels = [];
    for (var i = 0; i < xs.length; i++) {
      var xn = xs[i], yn = "yaxis" + xn.slice(5);
      if (!fl[yn] || !fl[xn].domain || !fl[yn].domain) {
        gd._gridSnap = null;
        return null;
      }
      panels.push({xn: xn, yn: yn,
                   xd: fl[xn].domain.slice(), yd: fl[yn].domain.slice()});
    }
    panels.sort(function (a, b) {
      var dy = b.yd[1] - a.yd[1];
      return Math.abs(dy) > 0.01 ? dy : a.xd[0] - b.xd[0];
    });
    var starts = {};
    for (var j = 0; j < panels.length; j++) starts[panels[j].xd[0].toFixed(3)] = 1;
    var cols = Object.keys(starts).length;
    gd._gridSnap = {panels: panels, cols: cols,
                    rows: Math.ceil(panels.length / cols)};
    return gd._gridSnap;
  }

  // Which annotations are subplot headings, worked out once from the snapshot
  // positions. Returns a map from annotation index to panel index.
  //
  // It has to be a snapshot for the same reason the grid does: after the first
  // pass the headings have been moved, so matching them against the original
  // panels finds nothing and the mapping silently empties.
  //
  // The distinction earns its keep in the annotation pass below. A subplot
  // heading and a row of bar value labels are both paper-referenced
  // annotations sharing a y, and the rule that hides an unreadable smear of
  // value labels was hiding headings too: subject_lines has twelve, three per
  // row in its 3-column build, so "three or more on one line that do not fit"
  // matched every row of them and took nine of the twelve panel names off the
  // chart.
  function titleMap(gd, info, snap) {
    if (gd._titleMap) return gd._titleMap;
    var map = {};
    if (info) {
      for (var a = 0; a < snap.length; a++) {
        var an = snap[a];
        if (!an.paper || !an.text) continue;
        var best = -1, bestD = 1e9;
        for (var m = 0; m < info.panels.length; m++) {
          var p = info.panels[m];
          var cx = (p.xd[0] + p.xd[1]) / 2;
          var d = Math.abs(an.x - cx) + Math.abs(an.y - p.yd[1]) * 1.5;
          if (d < bestD) { bestD = d; best = m; }
        }
        if (best >= 0 && bestD < 0.14) map[a] = best;
      }
    }
    gd._titleMap = map;
    return map;
  }

  function regrid(gd, up, cols, info, totalH, tmap, snap, needPx) {
    var n = info.panels.length, rows = Math.ceil(n / cols);
    var hgap = cols > 1 ? 0.10 : 0;
    // The gap between rows has to hold three things stacked on top of each
    // other: the upper panel's tick labels, its axis title, and the lower
    // panel's heading. A flat 0.09 was the first attempt and it put
    // "Score (1-6 scale)" straight through "Paper Average Scores". Sized in
    // pixels and converted, because what has to fit is text, not a fraction.
    //
    // The floor is 0.02, not the 0.05 it was. A fraction floor is the wrong
    // shape for this: it is a share of the whole figure, so on twelve stacked
    // rows it reserved 0.05 x 11 = 55% of the height for gaps that only ever
    // need 56px each, leaving subject_lines with 78px panels between 104px
    // troughs. The pixel figure is the real requirement; the fraction is only
    // how Plotly is told about it, so let the pixels decide.
    var vgap = rows > 1
      ? Math.max(0.02, Math.min(0.30, needPx / Math.max(240, totalH)))
      : 0;
    var colW = (1 - hgap * (cols - 1)) / cols;
    var rowH = (1 - vgap * (rows - 1)) / rows;
    var moves = [];
    for (var k = 0; k < n; k++) {
      var c = k % cols, r = Math.floor(k / cols), pnl = info.panels[k];
      var x0 = c * (colW + hgap), y1 = 1 - r * (rowH + vgap);
      var nx = [x0, x0 + colW], ny = [y1 - rowH, y1];
      moves.push({old: pnl, nx: nx, ny: ny});
      up[pnl.xn + ".domain"] = nx;
      up[pnl.yn + ".domain"] = ny;
    }
    // Subplot titles are paper-referenced annotations sitting above each panel,
    // so they do not travel with the domain and have to be carried across by
    // hand. Both the panel they belong to and their offset above it come from
    // the snapshot, never from where they currently sit: after the first pass
    // they sit wherever this function last put them, and re-deriving the offset
    // from that adds it a second time on every pass.
    for (var a in tmap) {
      var mv = moves[tmap[a]];
      if (!mv) continue;
      up["annotations[" + a + "].x"] = (mv.nx[0] + mv.nx[1]) / 2;
      up["annotations[" + a + "].y"] = mv.ny[1] + (snap[a].y - mv.old.yd[1]);
      up["annotations[" + a + "].yanchor"] = "bottom";
    }
    return rows;
  }

  // Break one long line into several at word boundaries. An SVG <text> does
  // not wrap, but Plotly honours <br> inside an annotation, so a headline that
  // will not fit on one line can still be shown in full rather than dropped.
  function wrapText(t, maxChars) {
    if (maxChars < 6) return null;
    var words = String(t).split(" "), lines = [], cur = "";
    for (var i = 0; i < words.length; i++) {
      if (cur && (cur + " " + words[i]).length > maxChars) { lines.push(cur); cur = words[i]; }
      else cur = cur ? cur + " " + words[i] : words[i];
    }
    if (cur) lines.push(cur);
    return {text: lines.join("<br>"), lines: lines.length};
  }

  function apply(gd) {
    if (!window.Plotly || !gd._fullLayout) return;
    var w = gd.getBoundingClientRect().width;
    if (!w) return;
    var o = orig(gd);

    // ---- Sankey ----
    // A Sankey has a genuine minimum width: it draws two labelled columns and
    // the flows between them, and at 344px the node labels sit on top of the
    // ribbons. Nothing about margins, fonts or grids helps, because the
    // information is horizontal by construction. So this is the one chart
    // allowed to be wider than the phone and scroll inside its own card. The
    // check in tests/check_responsive.py exempts anything inside an overflow-x
    // container, which is exactly this case.
    //
    // The decision reads the FRAME width, never the plot's own width. Keying it
    // off the plot was the first version and it oscillated: widening the plot
    // to 520 made the next pass see 520, conclude no scrolling was needed, and
    // reset it, 24 times over.
    // Below this the Sankey gives up and scrolls. It is deliberately low: with
    // the counts dropped from the node labels (see below) the diagram fits a
    // 360px phone, so scrolling is now the fallback for genuinely tiny frames
    // rather than the normal phone case it used to be.
    // 260, not 320. A 360px phone gives the chart frame 294px once the card
    // padding is taken out, so a 320px floor put the commonest phone width back
    // into the scrolling path this work exists to remove.
    var SANKEY_MIN = 260;
    var frameW = window.innerWidth || w;
    var isSankey = (gd.data || []).some(function (t) { return t.type === "sankey"; });
    var sankeyScroll = isSankey && frameW <= NARROW && frameW < SANKEY_MIN;
    if (isSankey) {
      // Most of a Sankey's width goes on its node labels, and here two thirds
      // of each label is a count: "Weak Accept · 609" against "Weak Accept".
      // Dropping the counts on a narrow frame is what lets the diagram fit a
      // phone instead of scrolling. The numbers are not lost; a Sankey's hover
      // reports the value of every node and every link.
      var wantShort = frameW <= NARROW;
      if (gd._sankeyShort !== wantShort) {
        var t0 = gd.data[0] || {};
        if (!gd._sankeyLabels && t0.node && t0.node.label) {
          gd._sankeyLabels = t0.node.label.slice();
        }
        if (gd._sankeyLabels) {
          var lab = wantShort
            ? gd._sankeyLabels.map(function (x) { return String(x).split(" · ")[0]; })
            : gd._sankeyLabels.slice();
          gd._sankeyShort = wantShort;
          gd._respSelf = true;
          try {
            window.Plotly.restyle(gd, {
              "node.label": [lab],
              "textfont.size": wantShort ? 11 : null
            });
          } catch (e) { gd._respSelf = false; }
          // Deliberately does NOT return. Relabelling does not change the
          // element's width, unlike the scroll fallback above, so the rest of
          // this pass is still valid. Returning here cost a whole extra pass,
          // and the headline was still at its original size when the layout
          // settled: measured 538px of text in a 328px frame.
        }
        gd._sankeyShort = wantShort;
      }
      if (sankeyScroll) {
        if (!gd._scrollWrap) {
          var wrap = document.createElement("div");
          wrap.style.cssText = "overflow-x:auto;-webkit-overflow-scrolling:touch;width:100%";
          gd.parentNode.insertBefore(wrap, gd);
          wrap.appendChild(gd);
          gd._scrollWrap = wrap;
        }
        if (gd.style.width !== SANKEY_MIN + "px") {
          gd.style.width = SANKEY_MIN + "px";
          try { window.Plotly.Plots.resize(gd); } catch (e) {}
          return;   // re-enter with the real width on the next pass
        }
        w = gd.getBoundingClientRect().width || SANKEY_MIN;
      } else if (gd._scrollWrap && gd.style.width) {
        gd.style.width = "";
        try { window.Plotly.Plots.resize(gd); } catch (e) {}
        return;
      }
    }

    // A scrolling Sankey is being drawn at a desktop-ish width inside a phone
    // frame, so it wants the desktop layout: shrinking its margins here is what
    // sliced the headline annotation off the top.
    var narrow = sankeyScroll ? false : (w <= NARROW);
    var scale = Math.min(1, w / REF);
    var up = {};
    // A figure's height is claimed by three independent things: a re-gridded
    // subplot layout, a legend moved below the plot, and a category axis that
    // needs a minimum row pitch. They each used to write layout.height
    // directly, so whichever ran last won and the others were silently lost:
    // subject_lines asked for 2040px for twelve stacked panels and then had it
    // overwritten by the legend's estimate, leaving its last panel titles 362px
    // below the bottom of the figure. Collect the claims, take the largest, and
    // write it once.
    var wantH = null;
    function claimH(v) { if (v && (!wantH || v > wantH)) wantH = v; }

    // Margins are only rescaled on a narrow frame. Scaling them proportionally
    // at every width was the first version and it introduced a bug at 1000px:
    // subject_movers asks for l=240 and needs nearly all of it, so shaving it
    // to 213 clipped "Surgical Planning and Simulation". Above the breakpoint
    // the figure gets exactly the margins the build asked for.
    var l = narrow ? Math.round(o.m.l * scale) : o.m.l;
    var r = narrow ? Math.round(o.m.r * scale) : o.m.r;
    var t = narrow ? Math.round(o.m.t * Math.max(0.75, scale)) : o.m.t;
    var b = narrow ? Math.round(o.m.b * Math.max(0.75, scale)) : o.m.b;

    var hasLegend = o.showlegend !== false && gd._fullLayout.showlegend;
    var vertical = !o.legend || o.legend.orientation !== "h";
    var legendBelow = false;
    if (!narrow && hasLegend) {
      // Put the legend back where the build wanted it. The narrow branch below
      // moves a vertical legend underneath the plot and never used to undo it,
      // so a window dragged from phone width back to desktop kept the phone
      // legend for the rest of the session.
      var ol = o.legend || {};
      up["legend.orientation"] = ol.orientation || "v";
      up["legend.x"] = (ol.x !== undefined ? ol.x : 1.02);
      up["legend.xanchor"] = ol.xanchor || "left";
      up["legend.y"] = (ol.y !== undefined ? ol.y : 1);
      up["legend.yanchor"] = ol.yanchor || "auto";
      up["legend.font.size"] = (ol.font && ol.font.size) || null;
      up["legend.itemwidth"] = (ol.itemwidth !== undefined ? ol.itemwidth : 30);
    }
    var legHEst = 0;
    if (narrow && hasLegend && vertical) {
      legendBelow = true;
      r = Math.round(Math.min(r, w * 0.06));
      up["legend.orientation"] = "h";
      up["legend.x"] = 0; up["legend.xanchor"] = "left";
      up["legend.y"] = -0.12; up["legend.yanchor"] = "top";
      var legFont = w < 400 ? 10 : 12;
      up["legend.font.size"] = legFont; up["legend.itemwidth"] = 30;
      b = Math.max(b, 30);

      // A legend moved below the plot needs the frame to grow, or it is simply
      // cut off: the maps carry 20 cluster names, which wrap to a dozen rows on
      // a phone and ran up to 105px past the bottom edge. The row count is
      // estimated from the trace names, not measured from the rendered legend,
      // because measuring what we are about to resize is how the earlier
      // oscillations started. The frame follows via the height message.
      var legNames = [];
      for (var ln = 0; ln < (gd.data || []).length; ln++) {
        var trn = gd.data[ln];
        if (trn && trn.showlegend !== false && trn.name) legNames.push(String(trn.name));
      }
      if (legNames.length > 3) {
        var rowW = 0, legRows = 1;
        for (var lq = 0; lq < legNames.length; lq++) {
          var entW = legNames[lq].length * legFont * 0.52 + 34;
          if (rowW + entW > w - 16) { legRows++; rowW = entW; } else rowW += entW;
        }
        if (!gd._origH) gd._origH = gd.getBoundingClientRect().height;
        var legH = legRows * (legFont + 13) + 40;
        legHEst = legH;
        if (legRows > 1) claimH(Math.round(gd._origH + legH));
      }
    }

    // Snapshot the annotations ONCE, from the user layout, before anything
    // reads or moves them. Reading them back from _fullLayout each pass does
    // not work: Plotly drops the text and position of an annotation once it is
    // hidden, so the next pass sees an empty label, concludes it fits, un-hides
    // it, and the pass after that hides it again. Same trap as reading dtick
    // from _fullLayout, and the same trap as the grid domains above.
    if (!gd._annSnap) {
      var srcAnn = (gd.layout && gd.layout.annotations) ||
                   gd._fullLayout.annotations || [];
      var flAnn = gd._fullLayout.annotations || [];
      gd._annSnap = [];
      for (var q0 = 0; q0 < srcAnn.length; q0++) {
        var a0 = srcAnn[q0], fa0 = flAnn[q0] || a0 || {};
        var raw0 = String((a0 && a0.text) || "");
        gd._annSnap.push({
          text: raw0.replace(/<[^>]*>/g, ""),
          raw: raw0,
          x: (a0 && a0.x !== undefined ? a0.x : fa0.x) || 0,
          y: (a0 && a0.y) || 0,
          // Only a paper-referenced annotation can be a subplot heading; one
          // pinned to data coordinates travels with its own axis already.
          paper: (fa0.xref === "paper" && fa0.yref === "paper"),
          // Whether y ALONE is paper-referenced, which is the question the
          // above-the-plot placement asks. The two are not the same: code_*
          // pins "Year avg 51.0%" to the top of the figure in paper units but
          // to the average line in data units, so it is above the plot without
          // being paper/paper. Testing y > 1 without this reads a bar value of
          // 51.0 on a data-referenced label as "above the plot".
          yPaper: (fa0.yref === "paper"),
          font: (a0 && a0.font && a0.font.size) || 13,
          // Lines it ALREADY has. code_* labels its average line as
          // "Year avg<br>51.0%", which is two lines tall however short the
          // text is, and the top margin has to cover that whether or not this
          // script wraps anything.
          lines: (raw0.match(/<br>/g) || []).length + 1
        });
      }
    }

    var info = gridInfo(gd);
    var tmap = titleMap(gd, info, gd._annSnap);

    // ---- axis text ----
    // Three failures live here, and all three are invisible to a check that
    // only measures the plotting rectangle:
    //   * a category axis with automargin:true expands the margin to fit its
    //     labels and squeezes the bars to nothing (subjects_*, code_*, 206px
    //     of margin in a 344px frame, measured);
    //   * the same axis with automargin:false clips the labels instead
    //     (subject_movers, orals_areas);
    //   * numeric ticks collide into an unreadable smear once the plot is
    //     narrow enough (0 200 400 600 on a 126px axis).
    // Shortening the labels is the only move that helps all three: it lets
    // automargin settle small, removes the need to clip, and frees width. The
    // full label stays in the hover, which is the same bargain _trunc_ticks
    // already makes on the desktop layout.
    var yNames = axisNames(gd._fullLayout, "y");
    for (var yi = 0; yi < yNames.length; yi++) {
      var yn = yNames[yi], oy = o.ax[yn];
      if (!oy) continue;
      var fy = gd._fullLayout[yn];
      if (narrow) {
        var yFont = Math.max(10, Math.round(oy.tickfont * 0.82));
        up[yn + ".tickfont.size"] = yFont;
        // Stacking twelve panels into one column leaves each one short, and a
        // numeric axis keeps drawing the same number of ticks into the smaller
        // space: subject_lines printed "0%" through "10%". Fewer ticks, same
        // as the x axis rule.
        if (oy.type !== "category" && info && info.panels.length > 3) {
          up[yn + ".nticks"] = 3;
        }
        if (oy.type === "category") {
          // Budget for the labels: enough to read, never more than a third of
          // the frame, so the bars keep the majority.
          // 0.26 rather than a third: automargin adds the tick length and its
          // own padding on top of the text, which measured about 40px on a
          // 278px frame, so the text budget has to sit below the margin target.
          var budget = Math.max(52, Math.min(w * 0.26, 115));
          var met = tickMetrics(gd, fy._id || "y");
          // Normalise the measurement to the original font size. Later passes
          // measure text this function already shrank, and feeding that back in
          // would let the labels grow again on every pass.
          var curFont = (fy.tickfont && fy.tickfont.size) || oy.tickfont;
          var perCharAtOrig = met.perChar * (oy.tickfont / Math.max(1, curFont));
          var effPerChar = Math.max(4, perCharAtOrig * (yFont / oy.tickfont));
          var chars = Math.max(6, Math.floor(budget / effPerChar));
          var vals = oy.tickvals || oy.categories;
          var txt = oy.ticktext || oy.categories;
          if (vals && vals.length && txt && txt.length === vals.length) {
            var cut = [];
            for (var ci = 0; ci < txt.length; ci++) cut.push(truncate(txt[ci], chars));
            up[yn + ".tickvals"] = vals;
            up[yn + ".ticktext"] = cut;
            // Once the labels are short, hand the margin back to automargin
            // rather than computing it here. Estimating it from an average
            // character width was tried first and was wrong by 19 to 33px,
            // because an average is not the widest label and the estimate has
            // to be right for the widest one. Plotly measures the real glyphs.
            // automargin only misbehaved in the first place because the labels
            // were long; with them cut to fit, it settles small on its own.
            up[yn + ".automargin"] = true;
          }
        }
      } else {
        up[yn + ".tickfont.size"] = oy.tickfont;
        if (oy.ticktext) { up[yn + ".tickvals"] = oy.tickvals; up[yn + ".ticktext"] = oy.ticktext; }
        else if (oy.type === "category") { up[yn + ".ticktext"] = null; up[yn + ".tickvals"] = null; }
        up[yn + ".automargin"] = oy.automargin;
        up[yn + ".nticks"] = null;
      }
    }

    var xNames = axisNames(gd._fullLayout, "x");
    for (var xi = 0; xi < xNames.length; xi++) {
      var xn = xNames[xi], ox = o.ax[xn];
      if (!ox) continue;
      if (narrow) {
        up[xn + ".tickfont.size"] = Math.max(10, Math.round(ox.tickfont * 0.82));
        // Colliding x ticks have three different causes and three different
        // fixes. Clearing tickvals was tried as one blanket answer and was
        // wrong: several charts place their ticks deliberately (CLAUDE.md
        // records the authors chart's "multiples of 5 plus the max", and the
        // orals charts label explicit bar-group centres), and throwing those
        // away let Plotly auto-tick a tiny range into 0.01, 0.02, -0.49.
        var labels = ox.ticktext || (ox.tickvals ? ox.tickvals.map(String) : null);
        var allYears = labels && labels.length > 1 && labels.every(function (v) {
          return /^(19|20)[0-9][0-9]$/.test(String(v));
        });
        if (allYears) {
          // Years shorten instead of thinning. Dropping every other year from
          // a five-year comparison loses data the chart exists to show; "'21"
          // costs about half the width and reads the same.
          up[xn + ".tickvals"] = ox.tickvals;
          up[xn + ".ticktext"] = labels.map(function (v) {
            return "’" + String(v).slice(2);
          });
        } else if (ox.tickvals && ox.tickvals.length) {
          var maxT = w < 420 ? 4 : 6;
          var k = Math.ceil(ox.tickvals.length / maxT);
          if (k > 1) {
            var tv = [], tt = [];
            for (var q = 0; q < ox.tickvals.length; q += k) {
              tv.push(ox.tickvals[q]);
              if (ox.ticktext) tt.push(ox.ticktext[q]);
            }
            up[xn + ".tickvals"] = tv;
            if (ox.ticktext) up[xn + ".ticktext"] = tt;
          }
        } else if (ox.type !== "category" && typeof ox.dtick === "number") {
          // A fixed step keeps every tick regardless of nticks: scores_2021
          // asks for one per score point, which is nine labels on a 130px
          // panel. Widen the step rather than removing it, so the ticks stay
          // on whole score values.
          up[xn + ".dtick"] = ox.dtick * (w < 420 ? 3 : 2);
        } else if (ox.type !== "category") {
          up[xn + ".nticks"] = w < 420 ? 3 : 4;
        }
      } else {
        up[xn + ".tickfont.size"] = ox.tickfont;
        up[xn + ".nticks"] = null;
        if (ox.tickvals) { up[xn + ".tickvals"] = ox.tickvals; up[xn + ".ticktext"] = ox.ticktext; }
        if (typeof ox.dtick === "number") up[xn + ".dtick"] = ox.dtick;
      }
    }

    // ---- axis titles ----
    // An axis title is one unwrappable line of SVG text. "Change in share of
    // papers, 2021 -> 2024 (percentage points)" cannot be made to fit a 360px
    // frame at any legible size, so below the breakpoint it is dropped rather
    // than sliced in half. Nothing is lost: every chart carrying a long axis
    // title has a card caption above it saying the same thing in prose.
    var allAx = xNames.concat(yNames);
    for (var ai = 0; ai < allAx.length; ai++) {
      var an = allAx[ai], oa = o.ax[an];
      if (!oa || !oa.titletext) continue;
      if (narrow) {
        var plotW = (gd._fullLayout._size && gd._fullLayout._size.w) || w;
        var est = oa.titletext.length * oa.titlefont * 0.52;
        if (est > plotW * 0.98) up[an + ".title.text"] = "";
        else up[an + ".title.font.size"] = Math.max(11, Math.round(oa.titlefont * 0.85));
      } else {
        up[an + ".title.text"] = oa.titletext;
        up[an + ".title.font.size"] = oa.titlefont;
      }
    }

    // ---- subplot grid ----
    if (info && info.cols > 1) {
      if (!gd._origGrid) {
        gd._origGrid = {cols: info.cols, rows: info.rows,
                        h: gd.getBoundingClientRect().height};
      }
      var og = gd._origGrid;
      // Few panels always go to a single column on a narrow frame. Two side by
      // side at 420-760px still collide: orals_overview printed "Presentations
      // by type" through "Share of accepted papers". Many panels keep two
      // columns above 420px, where each one is still wide enough to read.
      var want = narrow
        ? (info.panels.length <= 4 ? 1 : (w < 420 ? 1 : 2))
        : og.cols;
      want = Math.min(want, og.cols);
      gd._respCols = want;
      var wantRows = Math.ceil(info.panels.length / want);
      // Floor the per-row height at 150px rather than trusting og.h/og.rows.
      // og.h is the frame height at the first pass, which can still be Plotly's
      // 450px default if the container has not settled, and dividing that
      // across four rows then multiplying by twelve left subject_lines 218px
      // short of its own last panel. A stacked panel needs its heading, its
      // plot and its tick labels regardless of what the grid used to be.
      var perRow = Math.max(og.h / og.rows, 178);
      var totalH = (want === og.cols && !narrow)
        ? og.h : Math.round(perRow * wantRows);
      // What actually has to fit in the trough between two stacked panels:
      // the upper panel's tick labels, then the lower panel's heading, then
      // the lower panel's own axis title if it has one. Measured rather than
      // assumed, because a flat 100px is right for scores_* (which does carry
      // a per-panel "Score (1-6 scale)") and half again too much for
      // subject_lines (which carries none), and the surplus comes straight out
      // of the panels.
      var hasXTitle = false;
      for (var ht = 0; ht < xNames.length; ht++) {
        var oh = o.ax[xNames[ht]];
        if (oh && oh.titletext && up[xNames[ht] + ".title.text"] !== "") {
          hasXTitle = true;
          break;
        }
      }
      // What actually has to fit under a stacked panel before the next one
      // starts: its tick labels, then its axis title if it has one, then the
      // next panel's heading. 34px was not enough for the title case and left
      // "Score (1-6 scale)" 4px into "Paper Average Scores" at 768px.
      var gapPx2 = 56 + (hasXTitle ? 48 : 0);
      if (want === og.cols && !narrow) {
        // Nothing to re-lay. Put the build's own domains back verbatim instead
        // of recomputing them: regrid derives its gaps from this script's
        // constants, not from the horizontal_spacing and vertical_spacing the
        // figure was built with, so running it on an unchanged grid quietly
        // redesigned the desktop layout it was only meant to restore.
        for (var rk = 0; rk < info.panels.length; rk++) {
          up[info.panels[rk].xn + ".domain"] = info.panels[rk].xd;
          up[info.panels[rk].yn + ".domain"] = info.panels[rk].yd;
        }
        for (var ra in tmap) {
          up["annotations[" + ra + "].x"] = gd._annSnap[ra].x;
          up["annotations[" + ra + "].y"] = gd._annSnap[ra].y;
        }
      } else {
        regrid(gd, up, want, info, totalH, tmap, gd._annSnap, gapPx2);
        // More rows need more height, and the frame only knows to grow because
        // the page reports its own height to the parent. Without setting this
        // the panels would simply get shorter as they got narrower.
        claimH(totalH);
      }
    }

    // Whatever is left, keep the plot itself the majority of the frame.
    var budget2 = w * (1 - MIN_PLOT_SHARE);
    if (l + r > budget2 && (l + r) > 0) {
      var k = budget2 / (l + r);
      l = Math.round(l * k); r = Math.round(r * k);
    }
    up["margin.l"] = l; up["margin.r"] = r;
    up["margin.t"] = t; up["margin.b"] = b;

    // autoexpand=False is set on the semantic map so the plot does not change
    // width when the legend switches between 5 and 20 entries. That reasoning
    // holds beside the plot and not underneath it, so let Plotly expand again
    // once the legend has moved below.
    if (narrow && gd._fullLayout.margin &&
        gd._fullLayout.margin.autoexpand === false && hasLegend && vertical) {
      up["margin.autoexpand"] = true;
    }

    // ---- annotations ----
    // Annotations carry the value labels and the explanatory keys, and nothing
    // above touches them. Three separate failures lived here at 360px: the
    // trends grid ran its per-year figures together into "6.54/96/22/93.1/89/6";
    // the orals charts pinned thirteen raw means across one line and printed
    // them on top of each other; the confidence chart's scale key ("1 Not
    // confident · 2 Somewhat · ...") ran off the right edge of the card.
    //
    // Widths are estimated from the layout rather than measured in the DOM, on
    // purpose. Measuring what is currently drawn and then changing it means the
    // next pass measures the change, which is the oscillation that has bitten
    // this file three times already. The layout is a fixed input.
    // Snapshot the annotations ONCE, from the user layout. Reading them back
    // from _fullLayout each pass does not work: Plotly drops the text and
    // position of an annotation once it is hidden, so the next pass sees an
    // empty label, concludes it fits, un-hides it, and the pass after that
    // hides it again. Same trap as reading dtick from _fullLayout.
    var anns2 = gd._annSnap;
    var plotW2 = (gd._fullLayout._size && gd._fullLayout._size.w) || w;
    var bands = {};
    var above = [];
    for (var q2 = 0; q2 < anns2.length; q2++) {
      var a2 = anns2[q2];
      var newFont = narrow ? Math.max(8, Math.round(a2.font * 0.75)) : a2.font;
      up["annotations[" + q2 + "].font.size"] = newFont;
      if (!narrow) {
        up["annotations[" + q2 + "].visible"] = true;
        // From the snapshot, not from gd.layout: relayout writes our own
        // changes back into gd.layout, so restoring from there restores the
        // wrapped and shrunken narrow text rather than what the build wrote.
        up["annotations[" + q2 + "].text"] = a2.raw;
        up["annotations[" + q2 + "].y"] = a2.y;
        up["annotations[" + q2 + "].yanchor"] = "auto";
        up["annotations[" + q2 + "].yshift"] = 0;
        // A headline too wide for its plot is worth shrinking at any width,
        // not only on a phone. The scrolling Sankey renders at 520px and is
        // "wide" by every other measure, and its headline still did not fit.
        var wEst = a2.text.length * a2.font * 0.52;
        if (wEst > plotW2 && a2.text.length) {
          up["annotations[" + q2 + "].font.size"] =
            Math.max(9, Math.floor(plotW2 / (a2.text.length * 0.52)));
        }
        continue;
      }
      var estW = a2.text.length * newFont * 0.52;
      // Anything sitting above the plot is collected here and placed in pixels
      // further down, once the wrapping below has settled how tall each one is.
      //
      // A subplot heading is exempt: it sits above its own panel, not above the
      // figure, so for every panel but the top one "above the plot" is the
      // middle of the figure and there is nothing for the top margin to do.
      if (a2.yPaper && a2.y > 1 && !(q2 in tmap) && a2.text) {
        above.push({i: q2, y: a2.y, font: newFont, lines: a2.lines,
                    moved: false});
      }
      // Subplot headings are kept out of the bands entirely, and never hidden.
      // A heading and a strip of bar value labels look identical here (both are
      // paper-referenced annotations sharing a y), and the rule below hides a
      // row of three or more that cannot fit, on the grounds that a smear of
      // overprinted numbers says less than nothing and every number is still in
      // the hover. That reasoning does not transfer: a panel heading is the
      // only thing naming what the panel shows, it is not in any hover, and
      // subject_lines builds three per row, so the rule matched every row of
      // headings and took nine of its twelve panel names off the chart.
      // They are sized to their own panel just below instead.
      if (q2 in tmap) continue;
      var band = (Math.round(a2.y * 100) / 100).toFixed(2);
      (bands[band] = bands[band] || []).push({i: q2, w: estW, n: a2.text.length,
                                             f: newFont});
    }

    // Subplot headings: fit each one to the width of the panel it names, by
    // shrinking and then wrapping. Never dropped, never truncated.
    if (narrow && info) {
      var tCols = (gd._respCols || info.cols);
      var panelW = (plotW2 - (tCols > 1 ? plotW2 * 0.10 : 0)) / tCols;
      for (var tq in tmap) {
        var ta = anns2[tq];
        if (!ta.text) continue;
        var tf = Math.max(8, Math.round(ta.font * 0.75));
        var tw = ta.text.length * tf * 0.52;
        if (tw > panelW) {
          // Shrink only as far as 9px, then wrap whatever still does not fit.
          tf = Math.max(9, Math.floor(panelW / (ta.text.length * 0.52)));
          var tWrap = wrapText(ta.text,
                               Math.max(6, Math.floor(panelW / (tf * 0.52))));
          if (tWrap && tWrap.lines > 1) {
            up["annotations[" + tq + "].text"] = tWrap.text;
          }
        }
        up["annotations[" + tq + "].font.size"] = tf;
        up["annotations[" + tq + "].visible"] = true;
      }
    }
    // Wide frames get the same crowding test, but the remedy is a smaller font
    // rather than a hidden row. The orals value labels overlap by 3 to 4px at
    // 768 and 1000px too: adjacent bars in a year sit close together, so this
    // was never only a phone problem. Shrinking fixes it without taking
    // anything off the desktop page, which is not this work's to change.
    if (!narrow) {
      for (var wk in bands) {
        var wg = bands[wk];
        if (wg.length < 3) continue;
        var wSlot = plotW2 / wg.length, wWidest = 0, wIdx = 0;
        for (var wi = 0; wi < wg.length; wi++) {
          if (wg[wi].w > wWidest) { wWidest = wg[wi].w; wIdx = wi; }
        }
        if (wWidest + 6 <= wSlot) continue;
        var fit = Math.floor((wSlot - 6) / (wg[wIdx].n * 0.52));
        var newF = Math.max(9, Math.min(wg[wIdx].f, fit));
        for (var wj = 0; wj < wg.length; wj++) {
          up["annotations[" + wg[wj].i + "].font.size"] = newF;
        }
      }
    }
    if (narrow) {
      for (var bk in bands) {
        var grp = bands[bk];
        var tot = 0, widest = 0;
        for (var gi = 0; gi < grp.length; gi++) {
          tot += grp[gi].w;
          if (grp[gi].w > widest) widest = grp[gi].w;
        }
        // Compare the WIDEST label against the slot it gets, not the total
        // against the plot. Labels are spread evenly, so a total that just
        // fits still overprints wherever one label is longer than its share:
        // orals_code kept overlapping by 3px under the total rule.
        var slot = plotW2 / grp.length;
        // A row of three or more that cannot fit is a strip of value labels.
        // Hide the row: a smear of overprinted numbers says less than nothing,
        // and every one of them is still in the hover. Two or fewer is left
        // alone, which is what keeps subplot headings (one per panel, or one
        // per column) from being swept up by this rule.
        var hideRow = grp.length >= 3 && (widest + 6 > slot || tot > plotW2 * 0.9);
        for (var gj = 0; gj < grp.length; gj++) {
          var one = grp[gj];
          // A lone annotation wider than the plot is a key or a headline, not
          // a value label. Those are worth keeping, so shrink first and only
          // drop it if it still cannot fit at the floor size.
          var drop = hideRow || (grp.length < 3 && one.w > plotW2);
          if (drop && grp.length < 3) {
            // Order matters: wrap, then shrink, and only then give up. These
            // are headlines and scale keys (the Sankey's "60% of reject-leaning
            // reviewers moved to Accept", the confidence chart's "1 Not
            // confident · 2 Somewhat · ..."), i.e. the finding the chart is
            // making. Dropping one to save space loses the point of the chart.
            var snapT = anns2[one.i].text;
            var fSize = one.f;
            var maxCh = Math.floor(plotW2 / (fSize * 0.52));
            var wrapped = wrapText(snapT, maxCh);
            if (wrapped && wrapped.lines > 1 && wrapped.lines <= 3) {
              up["annotations[" + one.i + "].text"] = wrapped.text;
              if (anns2[one.i].yPaper && anns2[one.i].y >= 1) {
                var extraPx = wrapped.lines * Math.round(fSize * 1.4);
                var plotH = (gd._fullLayout._size && gd._fullLayout._size.h) || 220;
                if (typeof o.legendY === "number" && o.legendY >= 1) {
                  // The space above the plot already belongs to the legend, and
                  // there is no room to share it: pushing the legend up just
                  // made it grow back down over the key, and pushing the key up
                  // put it through the legend. Below the plot is empty, so the
                  // key goes there. This is the confidence chart's scale key,
                  // which reads perfectly well under the axis.
                  // Sit it just under the tick labels and reserve its full
                  // height below that. The first attempt offset it by its own
                  // height as well as the gap, which pushed the last line off
                  // the bottom of the frame.
                  var gapPx = 30;
                  up["annotations[" + one.i + "].y"] = -gapPx / plotH;
                  up["annotations[" + one.i + "].yanchor"] = "top";
                  up["annotations[" + one.i + "].yshift"] = 0;
                  b = Math.max(b, gapPx + extraPx + 18);
                  for (var mv2 = 0; mv2 < above.length; mv2++) {
                    if (above[mv2].i === one.i) above[mv2].moved = true;
                  }
                } else {
                  // Stays above the plot; it is now taller, so tell the pixel
                  // placement below how many lines it ended up with.
                  for (var lu = 0; lu < above.length; lu++) {
                    if (above[lu].i === one.i) {
                      above[lu].lines = wrapped.lines;
                      above[lu].font = fSize;
                    }
                  }
                }
              }
              drop = false;
            } else if (snapT.length * 9 * 0.52 <= plotW2) {
              up["annotations[" + one.i + "].font.size"] = 9;
              drop = false;
            }
          }
          up["annotations[" + one.i + "].visible"] = !drop;
        }
      }
    }

    // ---- annotations above the plot ----
    // Stack them upward from the top of the plot in PIXELS, and size the top
    // margin to the stack.
    //
    // The obvious approach, growing margin.t until the text fits, does not
    // converge, and this is the third rewrite of that idea. An annotation
    // parked at paper y = 1.15 sits 0.15 of the PLOT height above the plot, and
    // the plot height is what is left after the margins, so every pixel added
    // to the top margin takes a pixel off the plot and pulls the annotation
    // back down by 0.15 of it. The Sankey headline chased its own reservation
    // out of the frame that way, ending 35px above the top edge with a top
    // margin that had already grown to 54px.
    //
    // yshift breaks the loop: it is a plain pixel offset from the anchor, so
    // anchoring at y = 1 (the plot top, whatever the margin turns out to be)
    // and shifting up by a measured number of pixels removes plot height from
    // the equation altogether. Annotations sharing a y stay on one level, and
    // levels stack in the order the figure put them in, so the Sankey's
    // "Pre-rebuttal" and "Post-rebuttal" column headings stay side by side and
    // the headline sits clear above them.
    if (narrow && above.length) {
      var levels = {};
      for (var av = 0; av < above.length; av++) {
        if (above[av].moved) continue;
        var lk = above[av].y.toFixed(3);
        (levels[lk] = levels[lk] || []).push(above[av]);
      }
      var lKeys = Object.keys(levels).sort(function (p1, p2) {
        return parseFloat(p1) - parseFloat(p2);
      });
      var off = 6;
      for (var lv = 0; lv < lKeys.length; lv++) {
        var row = levels[lKeys[lv]], rowH2 = 0;
        for (var rj = 0; rj < row.length; rj++) {
          rowH2 = Math.max(rowH2, row[rj].lines * row[rj].font * 1.45);
        }
        for (var rp = 0; rp < row.length; rp++) {
          up["annotations[" + row[rp].i + "].y"] = 1;
          up["annotations[" + row[rp].i + "].yanchor"] = "bottom";
          up["annotations[" + row[rp].i + "].yshift"] = off;
        }
        off += Math.ceil(rowH2) + 6;
      }
      if (lKeys.length) t = Math.max(t, off + 4);
    }

    // ---- bar value labels ----
    // The two diverging bar charts (subject_movers, orals_areas) print a value
    // outside the end of every bar. Those labels are trace text, so nothing
    // above touches them, and narrowing the plot pushed the positive ones off
    // the right edge and the negative ones into the category names on the left.
    // Shrinking the text and letting Plotly re-autorange with the smaller text
    // gives the labels room at both ends without moving them inside, where a
    // short bar cannot hold one.
    var barText = (gd.data || []).some(function (t) {
      return t.type === "bar" && t.text && t.text.length;
    });
    if (barText) {
      if (gd._barTextNarrow !== narrow) {
        gd._barTextNarrow = narrow;
        gd._respSelf = true;
        try {
          window.Plotly.restyle(gd, {"textfont.size": narrow ? 9 : null});
        } catch (e) { gd._respSelf = false; }
      }
      {
        if (narrow) r = Math.max(r, 30);
        // Pad the value axis so the end bars do not touch the plot edge, which
        // is what pushed the most-negative bar's label into the category names.
        // Applied at every width, not only on a phone: the same label overlaps
        // its category name by 9px at 1000px, and on the published site, so
        // this is a pre-existing bug the new text checks exposed rather than
        // anything the narrow layout introduced. The padding is smaller on a
        // wide frame, where only a few pixels are missing.
        // The padding is computed from the DATA extents, not from the current
        // range: reading the range back and widening it means the next pass
        // widens the widened range, and the axis creeps outward on every pass.
        var lo = Infinity, hi = -Infinity;
        for (var bi = 0; bi < gd.data.length; bi++) {
          var tr = gd.data[bi];
          if (tr.type !== "bar" || !tr.x) continue;
          for (var vi = 0; vi < tr.x.length; vi++) {
            var v = tr.x[vi];
            if (typeof v === "number" && isFinite(v)) {
              if (v < lo) lo = v;
              if (v > hi) hi = v;
            }
          }
        }
        if (isFinite(lo) && isFinite(hi) && hi > lo) {
          var padv = (hi - lo) * (narrow ? 0.20 : 0.08);
          for (var bx = 0; bx < xNames.length; bx++) {
            up[xNames[bx] + ".range"] = [lo - padv, hi + padv];
            up[xNames[bx] + ".autorange"] = false;
          }
        }
      }
    }

    // ---- category axis height ----
    // A horizontal bar chart needs a minimum height per row or its category
    // labels overprint each other vertically: orals_areas ran "Machine Learning
    // - ..." into "Clinical applications ...". Only applied where the figure is
    // not already a re-gridded subplot, which sets its own height above.
    if (narrow && !info) {
      for (var cy = 0; cy < yNames.length; cy++) {
        var cyAx = gd._fullLayout[yNames[cy]];
        if (!cyAx || cyAx.type !== "category") continue;
        var nCat = (cyAx._categories || []).length;
        if (nCat < 6) continue;
        if (!gd._origH) gd._origH = gd.getBoundingClientRect().height;
        var needH = nCat * 26 + t + b + 30;
        if (needH > gd._origH) claimH(Math.round(needH));
      }
    }

    // ---- a legend parked above the plot, when the figure got taller ----
    // Same paper-unit trap as the annotations, one level up. scores_* asks for
    // legend.y = 1.35, which is 0.35 of the PLOT height above the plot: 91px in
    // the 380px frame the figure was designed in. Stacking its two panels makes
    // the figure 760px, the plot area grows with it, and that same 0.35 becomes
    // 369px. The legend does not overlap anything and nothing is cut off, so
    // every check passed on a chart with a third of a screen of white space
    // between its key and its first panel.
    //
    // A legend has no pixel offset to set (no yshift), so the paper figure is
    // rescaled instead, by the ratio of the old plot height to the new one.
    // Both come from snapshots and from the height this pass has already
    // decided, never from the current rendering, so it lands in one pass and
    // does not drift on the next.
    if (narrow && hasLegend && !legendBelow && wantH && o.h0 > 0 &&
        typeof o.legendY === "number" && o.legendY > 1) {
      var pOld = Math.max(80, o.h0 - o.m.t - o.m.b);
      var pNew = Math.max(80, wantH - o.m.t - o.m.b);
      if (pNew > pOld) up["legend.y"] = 1 + (o.legendY - 1) * (pOld / pNew);
    }
    // The legend this script moves below the plot has the same problem, and it
    // is this script that put it there. -0.12 is 0.12 of the plot height below
    // the plot, so on the semantic map, whose frame grows by about 250px to
    // hold twenty wrapped cluster names, it opened a 104px channel between the
    // map and its own key. Ask for a fixed 22px instead.
    if (narrow && legendBelow) {
      var hNow = wantH || o.h0 || gd.getBoundingClientRect().height || 400;
      var pBelow = Math.max(80, hNow - o.m.t - o.m.b - legHEst);
      up["legend.y"] = -(22 / pBelow);
    }

    // Re-write the top margin: the annotation pass above may have wrapped a
    // headline onto extra lines and needs room for them, and it runs after the
    // margins were first computed.
    up["margin.t"] = t;
    up["margin.r"] = r;
    up["margin.b"] = b;

    // In-plot controls (the maps\' "Color by Year / Color by Cluster" toggle) are
    // laid out by Plotly in pixels from a font size, so they do not shrink with
    // the frame. Measured at 360px: the pair ended at 327px inside a 328px div,
    // fitting by one pixel. That is not a margin worth trusting across font
    // stacks, so step the label size down on a narrow frame.
    if (narrow && gd._fullLayout.updatemenus && gd._fullLayout.updatemenus.length) {
      var fs = w < 380 ? 12 : 13;
      for (var u = 0; u < gd._fullLayout.updatemenus.length; u++) {
        up["updatemenus[" + u + "].font.size"] = fs;
      }
    }

    // Apply the height to the ELEMENT, not to layout.height. Every chart here
    // is built with config responsive:true, and Plotly's responsive handler
    // re-applies autosize on resize, which overwrites layout.height with the
    // container's height. Setting layout.height therefore looked like it worked
    // and silently reverted: subject_lines kept reporting 660px after asking
    // for 1980, leaving four of its twelve panels below the visible box. The
    // element's own height wins, and the frame follows via the height message.
    if (wantH && Math.abs((gd._respH || 0) - wantH) > 2) {
      gd._respH = wantH;
      gd.style.height = wantH + "px";
      gd._respSelf = true;
      try { window.Plotly.Plots.resize(gd); } catch (e) { gd._respSelf = false; }
    } else if (!wantH && gd._respH) {
      gd._respH = null;
      gd.style.height = "";
      gd._respSelf = true;
      try { window.Plotly.Plots.resize(gd); } catch (e) { gd._respSelf = false; }
    }

    // Skip a no-op relayout. This is what stops the plotly_afterplot hook below
    // from feeding itself: relayout fires afterplot, afterplot calls apply, and
    // apply must decide to do nothing the second time.
    var key = JSON.stringify(up);
    if (gd._respKey === key) return;
    // Backstop. Nothing should reach this, but a relayout loop on a page with
    // nine chart frames is a hung tab, not a cosmetic bug, so it is capped.
    gd._respCount = (gd._respCount || 0) + 1;
    if (gd._respCount > 40) return;
    gd._respKey = key;
    gd._respSelf = true;      // the afterplot this triggers is ours, not a redraw
    try { window.Plotly.relayout(gd, up); } catch (e) { gd._respSelf = false; }
  }

  // Charts with their own controls redraw through Plotly.react when the reader
  // changes a dropdown (subject_movers, orals_areas), which resets the layout
  // to what the build wrote. Without this hook the chart would look correct
  // until the first interaction and then silently revert.
  function hook(gd) {
    if (gd._respHooked || !gd.on) return;
    gd._respHooked = true;
    gd.on("plotly_afterplot", function () {
      // Every relayout we perform fires this too. Reapplying on our own event
      // and clearing the no-op guard while doing it is an infinite loop, and it
      // hangs the tab rather than merely looking wrong: relayout -> afterplot ->
      // relayout. So our own redraw is consumed here and goes no further.
      if (gd._respSelf) { gd._respSelf = false; return; }
      gd._respKey = null;       // a real redraw discarded what we last applied
      setTimeout(function () { apply(gd); }, 0);
    });
  }

  var lastSent = -1;
  var grew = 0;
  function reportHeight() {
    if (window.parent === window) return;
    var h = Math.max(
      document.documentElement.scrollHeight,
      document.body ? document.body.scrollHeight : 0
    );
    // Only speak when it changed. Plotly charts fill 100% of the frame, so
    // their reported height always equals whatever the parent already set;
    // without this they would post on every resize forever.
    if (Math.abs(h - lastSent) < 4) return;

    // Runaway guard. A page whose own height depends on the frame height can
    // ask for more room, get it, and then ask for more again: subject_movers
    // and orals_areas did exactly that, growing 16px per message, because they
    // sized the plot as calc(100vh - 42px) against a control row that wraps to
    // 58px on a phone. That is fixed at the source, but the guard stays: this
    // message can only ever make a frame taller, so a bug of this shape is
    // unbounded, and the failure (a chart card growing without limit as you
    // watch) is far worse than a chart that is 30px short.
    if (h > lastSent && lastSent > 0 && ++grew > 3) return;

    lastSent = h;
    try {
      window.parent.postMessage({kind: "miccai-chart-height", height: h}, "*");
    } catch (e) {}
  }

  function pass() {
    var plots = document.querySelectorAll(".js-plotly-plot");
    for (var i = 0; i < plots.length; i++) { hook(plots[i]); apply(plots[i]); }
    reportHeight();
  }

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }
  ready(function () {
    pass();
    // Plotly finishes drawing after DOMContentLoaded on a cold cache, and the
    // tick measurements above need real rendered text to read.
    setTimeout(pass, 150);
    setTimeout(pass, 600);
    setTimeout(pass, 1400);   // cold cache, and charts that settle in stages
  });

  var tmr = null;
  window.addEventListener("resize", function () {
    clearTimeout(tmr);
    tmr = setTimeout(function () {
      var plots = document.querySelectorAll(".js-plotly-plot");
      for (var i = 0; i < plots.length; i++) plots[i]._respKey = null;
      pass();
    }, 120);
  });
})();
</script>"""


# ---------------------------------------------------------------------------
# plotly.js bundle selection
# ---------------------------------------------------------------------------
# The full plotly.js bundle is 4.85 MB (1.47 MB over the wire) and carries 49
# trace types. This site draws five of them. Every chart is its own document in
# its own iframe, so a year page parses that bundle once per frame, and 46 of
# the 55 Plotly charts here are bar and scatter only.
#
# plotly.js publishes partial bundles at the same CDN and version. Measured for
# 3.7.0, compressed as served: basic 373 KB, cartesian 473 KB, gl2d 532 KB,
# full 1467 KB. save_chart picks the smallest one that covers what the figure
# actually draws.
#
# Trace lists are the registered trace modules in each published bundle, read
# out of the bundles themselves (grep for moduleType:"trace"), not from the
# docs. A type missing from this table falls through to the full bundle, so a
# future chart using a trace nobody listed here renders correctly and merely
# misses the saving.
_BUNDLE_TRACES: dict[str, frozenset[str]] = {
    "basic": frozenset({"bar", "pie", "scatter"}),
    "cartesian": frozenset({
        "bar", "box", "contour", "heatmap", "histogram", "histogram2d",
        "histogram2dcontour", "image", "pie", "scatter", "scatterternary",
        "violin",
    }),
    "gl2d": frozenset({"parcoords", "scatter", "scattergl", "splom"}),
}

# Smallest first; the first bundle that covers the figure wins.
_BUNDLE_ORDER = ("basic", "cartesian", "gl2d")

# Subresource integrity hashes for the partial bundles, keyed by plotly.js
# version. Plotly computes the full bundle's hash from the copy it vendors, but
# it does not vendor the partial ones, so there is nothing local to hash. These
# were taken from the published files with:
#
#   curl -sL https://cdn.plot.ly/plotly-basic-3.7.0.min.js \
#     | openssl dgst -sha256 -binary | openssl base64 -A
#
# The same command against the full bundle reproduces the hash plotly itself
# emits, which is how the method was checked. A version missing from this table
# is not an error: the script tag is written without an integrity attribute and
# a warning is logged, so a plotly upgrade degrades rather than breaking the
# build. Add the new hashes and the attribute comes back.
_BUNDLE_SRI: dict[str, dict[str, str]] = {
    "3.7.0": {
        "basic": "sha256-wjsDWRpr2tC9D0egxcUjBaVRmBK3POJUvmXNhBVTycI=",
        "cartesian": "sha256-fFk7ntoOdKHQczXPicv3pV/8EUkJmAw3Ka+DVFO9sCo=",
        "gl2d": "sha256-05bA1Z4oRKFn3ErKhGn+FlMGDLEF9NfL1Hs8d4ix51A=",
    },
}


# The tag plotly writes for include_plotlyjs="cdn". Matched rather than
# reconstructed, because its integrity attribute is a hash of the copy of
# plotly.js that the installed plotly package vendors, and reproducing that
# here would mean reimplementing a private plotly function.
_FULL_BUNDLE_TAG = re.compile(
    r'<script charset="utf-8" src="https://cdn\.plot\.ly/plotly-'
    r'[0-9][^"]*\.min\.js"[^>]*></script>'
)


def _bundle_for(fig: go.Figure) -> str:
    """Name the smallest plotly.js bundle that can draw every trace in fig."""
    types = {t.type for t in fig.data if t.type}
    for bundle in _BUNDLE_ORDER:
        if types <= _BUNDLE_TRACES[bundle]:
            return bundle
    return "full"


def _combine_bundles(a: str, b: str) -> str:
    """The smallest bundle that can draw everything a and b can."""
    if "full" in (a, b):
        return "full"
    need = _BUNDLE_TRACES[a] | _BUNDLE_TRACES[b]
    for bundle in _BUNDLE_ORDER:
        if need <= _BUNDLE_TRACES[bundle]:
            return bundle
    return "full"


# Which bundle to load is a property of the page, not of the chart, because a
# page pays for the union of what its frames ask for. Splitting is only a win
# while every frame can share one small bundle: the 2025 year page draws nine
# charts basic can handle plus one map needing gl2d, which is 905 KB against
# 1467 KB, but its Sankey exists in no partial bundle, so splitting there means
# fetching basic AND gl2d AND full, 2372 KB, and the "optimization" makes that
# page 57% heavier. Measured, not predicted; the Sankey sits sixth of ten, so
# readers reach it.
#
# So a page carrying a chart no partial bundle covers puts all of its charts on
# one bundle, and that page comes out exactly where it started rather than
# worse. Set for the duration of a page's charts by page_bundle_floor().
_PAGE_BUNDLE_FLOOR: str | None = None


@contextlib.contextmanager
def page_bundle_floor(bundle: str | None):
    """Widen every bundle chosen inside this block to cover `bundle` too."""
    global _PAGE_BUNDLE_FLOOR
    previous = _PAGE_BUNDLE_FLOOR
    _PAGE_BUNDLE_FLOOR = bundle
    try:
        yield
    finally:
        _PAGE_BUNDLE_FLOOR = previous


def plotly_script_tag(bundle: str = "basic") -> str:
    """The <script> tag loading one plotly.js bundle from the CDN.

    Mirrors the tag plotly writes for include_plotlyjs="cdn", integrity
    attribute included, so swapping one for the other changes only the URL.
    """
    version = get_plotlyjs_version()
    name = "plotly" if bundle == "full" else f"plotly-{bundle}"
    url = f"https://cdn.plot.ly/{name}-{version}.min.js"
    sri = _BUNDLE_SRI.get(version, {}).get(bundle)
    if sri is None and bundle != "full":
        logger.warning(
            f"  No SRI hash for {name}-{version}; writing the tag without one. "
            f"Add it to _BUNDLE_SRI to restore the integrity attribute."
        )
    integrity = f' integrity="{sri}" crossorigin="anonymous"' if sri else ""
    return f'<script charset="utf-8" src="{url}"{integrity}></script>'


def save_chart(
    fig: go.Figure,
    filename: str,
    out_dir: Path = OUT_DIR,
    interactive: bool = False,
    extra_js: str = "",
) -> None:
    """Write one figure to out_dir/filename, optionally with extra script.

    div_id is set explicitly rather than left to Plotly, which otherwise stamps
    a fresh random UUID into every file on every build. That made all 60 chart
    files differ byte-for-byte between two builds of identical data, which in
    turn made "did my change alter any output?" impossible to answer with a
    diff. Filenames are unique within the directory and each file holds exactly
    one figure, so the stem is a safe id; the "chart-" prefix guarantees it
    starts with a letter, as an HTML id must.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=True,
        config=_config(interactive),
        div_id="chart-" + filename.removesuffix(".html"),
    )
    # Downgrade the full bundle to the smallest one that can draw this figure.
    # Done by rewriting plotly's own tag rather than by passing a URL to
    # include_plotlyjs, because that route drops the integrity attribute.
    # A miss here is loud on purpose: silently leaving the 1.47 MB bundle in
    # place is exactly the bug this function exists to prevent, and it looks
    # like a working chart.
    bundle = _bundle_for(fig)
    if _PAGE_BUNDLE_FLOOR is not None:
        bundle = _combine_bundles(bundle, _PAGE_BUNDLE_FLOOR)
    if bundle != "full":
        html, n = _FULL_BUNDLE_TAG.subn(plotly_script_tag(bundle), html, count=1)
        if n != 1:
            raise RuntimeError(
                f"{filename}: could not find plotly's CDN script tag to "
                f"replace. Plotly's HTML output has changed; update "
                f"_FULL_BUNDLE_TAG to match it."
            )
    # RESPONSIVE_JS is added here rather than passed by each caller, so that a
    # new chart cannot be written without it. See the comment on RESPONSIVE_JS.
    html = html.replace("</body>", RESPONSIVE_JS + (extra_js or "") + "\n</body>")
    (out_dir / filename).write_text(html, encoding="utf-8")
    logger.info(f"  Saved {filename}")


def theme_tokens(bundle: str = "basic") -> dict:
    """The design tokens every hand-written chart template needs.

    A function rather than a constant because __PLOTLYSCRIPT__ has to track the
    installed plotly, and freezing it at import time is exactly the kind of
    drift this project keeps getting bitten by. Tokens a given template does
    not contain are simply not found, so passing the whole set costs nothing.

    bundle defaults to "basic" because all three Plotly templates draw bars and
    lines and nothing else. save_chart reads the bundle off the figure, which a
    template does not have, so this one is declared by the caller; a template
    that grows a trace basic cannot draw has to say so here.
    """
    return {
        "__PLOTLYSCRIPT__": plotly_script_tag(bundle),
        "__FONT__": FONT,
        "__BG__": BG,
        "__PLOTBG__": PLOT_BG,
        "__TEXT__": TEXT,
        "__MUTED__": MUTED,
        "__RULE__": RULE,
        "__ACCENT__": ACCENT,
    }


def save_template(
    template: str,
    filename: str,
    tokens: dict,
    out_dir: Path = OUT_DIR,
    note: str = "",
) -> None:
    """Write one hand-written chart page, filling its __TOKEN__ placeholders.

    The counterpart to save_chart, for the four charts whose interactivity is a
    few lines of plain JavaScript and would be harder to read forced through
    Plotly: trends_overview, subject_movers, orals_areas, and the d3
    co-authorship network. Each used to carry its own copy of this loop plus
    the same mkdir, write and log underneath it.
    """
    html = template
    for token, value in tokens.items():
        html = html.replace(token, value)
    # Same automatic injection as save_chart, for the same reason. Three of the
    # four templates draw Plotly figures and take the full treatment; the d3
    # co-authorship page loads no Plotly, so the Plotly half finds nothing and
    # only the height report runs. Its own resize handling is separate work.
    if "</body>" not in html:
        raise RuntimeError(
            f"{filename}: template has no </body> to inject the responsive "
            f"script before. Every chart page needs it; add the tag."
        )
    html = html.replace("</body>", RESPONSIVE_JS + "\n</body>")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / filename).write_text(html, encoding="utf-8")
    logger.info(f"  Saved {filename}{note}")


def _verdict(label: str | None) -> str | None:
    if not label:
        return None
    parts = re.split(r"\s*[\u2014\u2013-]\s*", label.strip(), maxsplit=1)
    return parts[0].strip() if parts else label.strip()


def _short_subject(label: str) -> str:
    """Strip 'Applications -> ', 'Machine Learning -> ', etc. prefixes."""
    if " -> " in label:
        return label.split(" -> ", 1)[1]
    if " - " in label:
        return label.split(" - ", 1)[1]
    return label


def _trunc(label: str, n: int = 32) -> str:
    """Shorten long axis labels; the full name stays available in hover text."""
    return label if len(label) <= n else label[: n - 1].rstrip() + "…"


def _trunc_ticks(labels: list, n: int = 32) -> dict:
    """yaxis kwargs mapping full category values to truncated display text."""
    return dict(
        tickmode="array",
        tickvals=labels,
        ticktext=[_trunc(x, n) for x in labels],
    )


def _wrap(label: str, n: int = 26) -> str:
    """Wrap a long axis label onto at most two lines (Plotly reads <br>).

    Preferred over _trunc for the subject/code bar charts: full subject names stay
    readable instead of being cut at 32 chars. Overflow past two lines is elided.
    """
    if len(label) <= n:
        return label
    lines: list[str] = []
    line = ""
    for word in label.split():
        candidate = f"{line} {word}".strip()
        if len(candidate) > n and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    if len(lines) > 2:
        lines = lines[:2]
        lines[1] = lines[1][: n - 1].rstrip() + "…"
    return "<br>".join(lines)


def _wrap_ticks(labels: list, n: int = 26) -> dict:
    """yaxis kwargs mapping full category values to two-line display text."""
    return dict(
        tickmode="array",
        tickvals=labels,
        ticktext=[_wrap(x, n) for x in labels],
        automargin=True,
    )


# Frame heights for the horizontal bar charts are computed from what was
# actually drawn, then written to chart_heights.json for build_site.py to read.
# They used to be hardcoded in the templates, which went stale every time the
# type scale changed, and left paired charts in a two-column row at visibly
# different heights. A chart knows its own row count; the template does not.
CHART_HEIGHTS: dict = {}
HEIGHTS_JSON = OUT_DIR / "chart_heights.json"


def _bar_height(
    display_labels: list, tick_size: float, chrome: int = 120
) -> int:
    """Frame height for a horizontal bar chart.

    One row per bar, an extra line for each label that wraps to two, plus fixed
    chrome for the axis title, tick labels and margins. `display_labels` must be
    the post-wrap text, so pass the output of `_wrap`, not the raw names.
    """
    row = 2.15 * tick_size
    extra = 1.25 * tick_size
    wrapped = sum(1 for x in display_labels if "<br>" in str(x))
    return int(round(chrome + len(display_labels) * row + wrapped * extra))


def _kde(values: list, x_grid: np.ndarray, min_bw: float = 0.0) -> np.ndarray:
    """Gaussian KDE (Silverman bandwidth); plain numpy, no scipy dependency.

    min_bw floors the bandwidth: review scores are integers (and paper averages
    quantized to 1/3 steps), where Silverman's rule yields a spiky comb instead
    of a smooth density.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    std = v.std(ddof=1) if n > 1 else 0.0
    q75, q25 = np.percentile(v, [75, 25])
    iqr = q75 - q25
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    if sigma <= 0:
        sigma = max(std, 0.5)
    bw = max(0.9 * sigma * n ** (-0.2), min_bw)
    diff = (x_grid[:, None] - v[None, :]) / bw
    return np.exp(-0.5 * diff**2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))


def _hist_kde_traces(
    values: list,
    color: str,
    *,
    discrete: bool = False,
    nbins: int = 24,
    bin_edges: np.ndarray | None = None,
    width_frac: float = 0.92,
    opacity: float = 0.55,
    name: str | None = None,
    showlegend: bool = False,
    hover_label: str = "Value",
) -> tuple:
    """Histogram bars + smooth KDE curve scaled to counts (seaborn histplot kde=True look).

    discrete=True bins at each integer (review scores, word counts); otherwise
    nbins equal-width bins over the data range, or explicit bin_edges (pass the
    same edges to overlay two series on aligned bins).
    Returns (bar_trace, line_trace); line_trace is None when the sample is too
    small for a meaningful KDE.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if discrete:
        lo, hi = int(v.min()), int(v.max())
        centers = np.arange(lo, hi + 1)
        counts = np.array([(v == c).sum() for c in centers])
        binwidth = 1.0
        grid = np.linspace(lo - 0.75, hi + 0.75, 240)
        min_bw = 0.45  # integer-spaced data needs at least this to be smooth
    else:
        edges = (
            bin_edges
            if bin_edges is not None
            else np.histogram_bin_edges(v, bins=nbins)
        )
        counts, _ = np.histogram(v, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2
        binwidth = float(edges[1] - edges[0])
        pad = (edges[-1] - edges[0]) * 0.06
        grid = np.linspace(edges[0] - pad, edges[-1] + pad, 240)
        min_bw = 0.6 * binwidth

    extra = f"<extra>{name}</extra>" if name else "<extra></extra>"
    bar = go.Bar(
        x=centers,
        y=counts,
        width=binwidth * width_frac,
        marker=dict(color=color, opacity=opacity, line=dict(width=0)),
        hovertemplate=f"{hover_label}: %{{x}}<br>Count: %{{y}}{extra}",
        name=name or "",
        showlegend=showlegend,
    )
    line = None
    if n >= 10:
        line = go.Scatter(
            x=grid,
            y=_kde(values, grid, min_bw=min_bw) * n * binwidth,
            mode="lines",
            line=dict(color=color, width=2.5, shape="spline"),
            hoverinfo="skip",
            showlegend=False,
        )
    return bar, line


# ---------------------------------------------------------------------------
# Per-year chart functions
# ---------------------------------------------------------------------------


def chart_map(papers_yr: list, year: int) -> None:
    cluster_ids = sorted(set(p["cluster_id"] for p in papers_yr))
    cluster_label_map = {
        p["cluster_id"]: p["cluster_label"] for p in papers_yr
    }

    traces = []
    for cid in cluster_ids:
        cps = [p for p in papers_yr if p["cluster_id"] == cid]
        color = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
        traces.append(
            go.Scattergl(
                x=[p["umap_x"] for p in cps],
                y=[p["umap_y"] for p in cps],
                mode="markers",
                name=cluster_label_map[cid],
                marker=dict(
                    color=color, size=5, opacity=0.75, line=dict(width=0)
                ),
                customdata=[[p["url"], p["title"]] for p in cps],
                hovertemplate="%{customdata[1]}<extra></extra>",
            )
        )

    fig = go.Figure(
        data=traces,
        layout=base_layout(
            showlegend=True,
            legend=dict(
                font=dict(size=13.5),
                itemsizing="constant",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=RULE,
                borderwidth=1,
                x=1.01,
                xanchor="left",
                y=1,
                yanchor="top",
            ),
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                title="",
                showline=False,
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                title="",
                showline=False,
            ),
            margin=dict(l=20, r=220, t=20, b=20),
            dragmode="pan",
        ),
    )

    save_chart(fig, f"map_{year}.html", interactive=True, extra_js=CLICK_JS)


def chart_subjects(papers_yr: list, year: int) -> None:
    counts: Counter = Counter()
    for p in papers_yr:
        for s in p.get("subject_areas", []):
            counts[_short_subject(s)] += 1

    top = counts.most_common(20)
    labels = [x[0] for x in reversed(top)]
    values = [x[1] for x in reversed(top)]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=year_main(year), line=dict(color="#fff", width=0.5)
            ),
            hovertemplate="%{y}: %{x} papers<extra></extra>",
        )
    )
    fig.update_layout(
        base_layout(
            bargap=0.32,
            margin=dict(l=200, r=36, t=20, b=50),
            yaxis=dict(tickfont=dict(size=14.5), **_wrap_ticks(labels)),
            xaxis=dict(
                title="Number of papers", gridcolor=RULE, nticks=5, tickangle=0
            ),
        )
    )
    CHART_HEIGHTS[f"subjects_{year}"] = _bar_height(
        [_wrap(x, 26) for x in labels], 14.5
    )
    save_chart(fig, f"subjects_{year}.html")


def chart_scores(papers_yr: list, year: int, scale_max: int) -> None:
    def _reviewer_scores(papers):
        return [
            r["score_raw"]
            for p in papers
            for r in p.get("reviews", [])
            if r.get("score_raw") is not None
        ]

    def _avg_scores(papers):
        return [
            p["avg_score_raw"]
            for p in papers
            if p.get("avg_score_raw") is not None
        ]

    early_yr = [p for p in papers_yr if p.get("early_accepted")]
    scores_all, scores_early = (
        _reviewer_scores(papers_yr),
        _reviewer_scores(early_yr),
    )
    avg_all, avg_early = _avg_scores(papers_yr), _avg_scores(early_yr)

    main = year_main(year)
    dark = _darken(main, 0.45)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Individual Reviewer Scores", "Paper Average Scores"),
    )

    # Col 1; discrete reviewer scores: all papers behind, early accepted in front
    bar, line = _hist_kde_traces(
        scores_all,
        main,
        discrete=True,
        name="All papers",
        showlegend=True,
        hover_label="Score",
    )
    fig.add_trace(bar, row=1, col=1)
    if scores_early:
        bar_e, line_e = _hist_kde_traces(
            scores_early,
            dark,
            discrete=True,
            width_frac=0.45,
            opacity=0.9,
            name="Early accepted",
            showlegend=True,
            hover_label="Score",
        )
        fig.add_trace(bar_e, row=1, col=1)
        if line_e:
            fig.add_trace(line_e, row=1, col=1)
    if line:
        fig.add_trace(line, row=1, col=1)

    # Col 2; paper averages: shared bins so the overlay aligns
    edges = np.histogram_bin_edges(
        np.asarray(avg_all, dtype=float), bins=2 * (scale_max - 1)
    )
    bar, line = _hist_kde_traces(
        avg_all,
        main,
        bin_edges=edges,
        name="All papers",
        hover_label="Avg score",
    )
    bar.hovertemplate = (
        "Avg score: %{x:.2f}<br>Count: %{y}<extra>All papers</extra>"
    )
    fig.add_trace(bar, row=1, col=2)
    if avg_early:
        bar_e, line_e = _hist_kde_traces(
            avg_early,
            dark,
            bin_edges=edges,
            width_frac=0.45,
            opacity=0.9,
            name="Early accepted",
            hover_label="Avg score",
        )
        bar_e.hovertemplate = (
            "Avg score: %{x:.2f}<br>Count: %{y}<extra>Early accepted</extra>"
        )
        fig.add_trace(bar_e, row=1, col=2)
        if line_e:
            fig.add_trace(line_e, row=1, col=2)
    if line:
        fig.add_trace(line, row=1, col=2)

    fig.update_layout(
        **base_layout(margin=dict(l=60, r=30, t=70, b=50)),
        barmode="overlay",
        bargap=0,
        legend=dict(
            orientation="h",
            x=0,
            y=1.35,
            xanchor="left",
            yanchor="top",
            font=dict(size=14.5),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        gridcolor=RULE,
        title_text=f"Score (1-{scale_max} scale)",
        range=[0.4, scale_max + 0.6],
        dtick=1,
        col=1,
    )
    # Clip the averages panel to the data range; the full 1-max span leaves
    # dead space and skinny bars
    span = edges[-1] - edges[0]
    fig.update_xaxes(
        gridcolor=RULE,
        title_text=f"Score (1-{scale_max} scale)",
        range=[edges[0] - 0.08 * span, edges[-1] + 0.08 * span],
        col=2,
    )
    fig.update_yaxes(gridcolor=RULE, title_text="Count", col=1)
    fig.update_yaxes(gridcolor=RULE, col=2)
    save_chart(fig, f"scores_{year}.html")


def chart_controversy(papers_yr: list, year: int, scale_max: int) -> None:
    ranges = [
        p["score_range"]
        for p in papers_yr
        if p.get("score_range") is not None and not p.get("early_accepted")
    ]

    # Secondary year color; the two Review Score Distribution plots share the
    # main color, this one takes the other (1-1-2 pattern across the row)
    bar, line = _hist_kde_traces(
        ranges, year_secondary(year), discrete=True, hover_label="Score range"
    )
    fig = go.Figure(data=[t for t in (bar, line) if t is not None])
    fig.update_layout(
        base_layout(
            margin=dict(l=60, r=30, t=40, b=50),
            xaxis=dict(
                title="Score range (max − min)", gridcolor=RULE, dtick=1
            ),
            yaxis=dict(title="Number of papers", gridcolor=RULE),
            bargap=0,
            showlegend=False,
        )
    )
    save_chart(fig, f"controversy_{year}.html")


VERDICT_RANK = {v: i for i, v in enumerate(VERDICT_ORDER)}

VERDICT_COLORS = {
    "Strong Reject": "#991b1b",
    "Reject": "#dc2626",
    "Weak Reject": "#f59e0b",
    "Weak Accept": "#4ade80",
    "Accept": "#16a34a",
    "Strong Accept": "#166534",
}

# Link tint by direction of change (improved / unchanged / downgraded)
FLOW_UP = "rgba(22,163,74,0.30)"
FLOW_SAME = "rgba(100,116,139,0.18)"
FLOW_DOWN = "rgba(220,38,38,0.30)"


def _rebuttal_flows(papers_yr: list) -> Counter:
    """Reviewer verdict transitions pre → post rebuttal, empty where unrecorded.

    Split out of chart_rebuttal_sankey because main() has to know whether the
    Sankey will be drawn before it draws that year's first chart: the Sankey is
    the only chart here needing the full plotly bundle, and that decides which
    bundle every other chart on the same page gets. See _PAGE_BUNDLE_FLOOR.
    """
    flows: Counter = Counter()
    for p in papers_yr:
        for r in p.get("reviews", []):
            pre = _verdict(r.get("recommendation_label", ""))
            post = _verdict(r.get("post_rebuttal_label", ""))
            if pre not in VERDICT_RANK or post not in VERDICT_RANK:
                continue
            flows[(pre, post)] += 1
    return flows


def chart_rebuttal_sankey(papers_yr: list, year: int) -> bool:
    """Sankey of reviewer verdicts pre → post rebuttal.

    Only possible where post-rebuttal verdict labels exist (2024/2025).
    Reviews without a post-rebuttal response (N/A) are excluded.
    Returns True if the chart was written.
    """
    flows = _rebuttal_flows(papers_yr)
    if not flows:
        logger.info(
            f"  Skipping rebuttal_{year}.html; no post-rebuttal verdict labels"
        )
        return False

    display_order = list(reversed(VERDICT_ORDER))  # best verdict at the top
    pre_nodes = [v for v in display_order if any(k[0] == v for k in flows)]
    post_nodes = [v for v in display_order if any(k[1] == v for k in flows)]

    pre_tot = {
        v: sum(c for (a, _), c in flows.items() if a == v) for v in pre_nodes
    }
    post_tot = {
        v: sum(c for (_, b), c in flows.items() if b == v) for v in post_nodes
    }

    def _y_positions(order: list, totals: dict) -> list:
        """Center nodes vertically, sized proportionally, top → bottom."""
        total = sum(totals.values())
        pad = 0.03
        avail = 1 - pad * (len(order) - 1)
        ys, cum = [], 0.0
        for k in order:
            frac = totals[k] / total * avail
            ys.append(min(max(cum + frac / 2, 0.001), 0.999))
            cum += frac + pad
        return ys

    labels = [f"{v} · {pre_tot[v]:,}" for v in pre_nodes] + [
        f"{v} · {post_tot[v]:,}" for v in post_nodes
    ]
    # Plain verdict names for hover text; the visible labels already carry counts
    node_names = pre_nodes + post_nodes
    node_colors = [VERDICT_COLORS[v] for v in pre_nodes] + [
        VERDICT_COLORS[v] for v in post_nodes
    ]
    node_x = [0.001] * len(pre_nodes) + [0.999] * len(post_nodes)
    node_y = _y_positions(pre_nodes, pre_tot) + _y_positions(
        post_nodes, post_tot
    )

    pre_idx = {v: i for i, v in enumerate(pre_nodes)}
    post_idx = {v: len(pre_nodes) + i for i, v in enumerate(post_nodes)}

    src, tgt, val, link_colors = [], [], [], []
    for (pre, post), c in sorted(flows.items(), key=lambda kv: -kv[1]):
        src.append(pre_idx[pre])
        tgt.append(post_idx[post])
        val.append(c)
        d = VERDICT_RANK[post] - VERDICT_RANK[pre]
        link_colors.append(
            FLOW_UP if d > 0 else FLOW_DOWN if d < 0 else FLOW_SAME
        )

    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            valueformat=",.0f",
            node=dict(
                label=labels,
                color=node_colors,
                x=node_x,
                y=node_y,
                customdata=node_names,
                pad=12,
                thickness=14,
                line=dict(width=0),
                hovertemplate="%{customdata}: %{value:,.0f} reviews<extra></extra>",
            ),
            link=dict(
                source=src,
                target=tgt,
                value=val,
                color=link_colors,
                hovertemplate="%{source.customdata} → %{target.customdata}: "
                "%{value:,.0f} reviews<extra></extra>",
            ),
        )
    )
    # Headline takeaway: of the reviewers who came in on the reject side, what
    # share ended up on the accept side? Without this the Sankey reads as a
    # tangle of ribbons with no stated conclusion.
    rej_pre = sum(
        c
        for (a, _), c in flows.items()
        if VERDICT_RANK[a] < VERDICT_RANK["Weak Accept"]
    )
    rej_to_acc = sum(
        c
        for (a, b), c in flows.items()
        if VERDICT_RANK[a] < VERDICT_RANK["Weak Accept"]
        and VERDICT_RANK[b] >= VERDICT_RANK["Weak Accept"]
    )
    headline = (
        f"{rej_to_acc / rej_pre * 100:.0f}% of reject-leaning reviewers "
        f"moved to Accept after rebuttal"
        if rej_pre
        else ""
    )

    fig.update_layout(
        base_layout(
            margin=dict(l=20, r=20, t=64, b=30),
            font=dict(family=FONT, size=15.5, color=TEXT),
            annotations=[
                dict(
                    x=0.0,
                    y=1.06,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    showarrow=False,
                    text="<b>Pre-rebuttal</b>",
                    font=dict(size=15.5, color=MUTED),
                ),
                dict(
                    x=1.0,
                    y=1.06,
                    xref="paper",
                    yref="paper",
                    xanchor="right",
                    showarrow=False,
                    text="<b>Post-rebuttal</b>",
                    font=dict(size=15.5, color=MUTED),
                ),
                dict(
                    x=0.5,
                    y=1.15,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    showarrow=False,
                    text=headline,
                    font=dict(size=17, color=TEXT),
                ),
            ],
        )
    )
    save_chart(fig, f"rebuttal_{year}.html")
    return True


def chart_code(papers_yr: list, year: int) -> None:
    area_total: Counter = Counter()
    area_code: Counter = Counter()

    for p in papers_yr:
        has_code = bool(p.get("has_code"))
        for s in p.get("subject_areas", []):
            label = _short_subject(s)
            area_total[label] += 1
            if has_code:
                area_code[label] += 1

    # Only areas with >= 8 papers; sort by code rate
    areas = [
        (s, area_code[s] / area_total[s] * 100, area_total[s])
        for s in area_total
        if area_total[s] >= 8
    ]
    areas.sort(key=lambda x: x[1])
    areas = areas[-20:]

    labels = [x[0] for x in areas]
    pcts = [round(x[1], 1) for x in areas]
    totals = [x[2] for x in areas]

    year_avg = (
        sum(1 for p in papers_yr if p.get("has_code")) / len(papers_yr) * 100
    )

    fig = go.Figure(
        go.Bar(
            x=pcts,
            y=labels,
            orientation="h",
            marker=dict(
                color=year_main(year), line=dict(color="#fff", width=0.5)
            ),
            hovertemplate="%{y}: %{x:.1f}%  (%{customdata} papers)<extra></extra>",
            customdata=totals,
        )
    )
    fig.add_vline(
        x=year_avg,
        line_dash="dash",
        line_color=MUTED,
        line_width=1.5,
        annotation_text=f"Year avg {year_avg:.1f}%",
        annotation_position="top",
    )
    fig.update_layout(
        base_layout(
            bargap=0.32,
            margin=dict(l=200, r=36, t=30, b=50),
            xaxis=dict(
                title="% papers with code", range=[0, 105], gridcolor=RULE
            ),
            yaxis=dict(tickfont=dict(size=14.5), **_wrap_ticks(labels)),
        )
    )
    CHART_HEIGHTS[f"code_{year}"] = _bar_height(
        [_wrap(x, 26) for x in labels], 14.5
    )
    save_chart(fig, f"code_{year}.html")


def chart_authors(papers_yr: list, year: int) -> None:
    author_counts: Counter = Counter()
    for p in papers_yr:
        for a in p.get("authors", []):
            author_counts[a] += 1

    top20 = author_counts.most_common(20)
    labels = [x[0] for x in reversed(top20)]
    values = [x[1] for x in reversed(top20)]

    # Ticks at multiples of 5 plus the max value (if not already one); dtick=1
    # produced a wall of rotated labels in the half-width iframe.
    max_v = max(values)
    tickvals = list(range(0, max_v + 1, 5))
    if max_v % 5:
        tickvals.append(max_v)

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=year_main(year),
            hovertemplate="%{y}: %{x} papers<extra></extra>",
        )
    )
    fig.update_layout(
        base_layout(
            margin=dict(l=200, r=30, t=30, b=50),
            xaxis=dict(
                title="Number of papers",
                tickmode="array",
                tickvals=tickvals,
                tickangle=0,
                gridcolor=RULE,
            ),
            yaxis=dict(tickfont=dict(size=14.5)),
        )
    )
    CHART_HEIGHTS[f"authors_{year}"] = _bar_height(labels, 14.5)
    save_chart(fig, f"authors_{year}.html")


D3_NETWORK_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
  html,body{margin:0;height:100%;font-family:__FONT__;background:#fff}
  /* display:block matters: an inline <svg> sits on a text baseline, so the
     line-box descender added ~18px below it and the page overflowed its
     iframe by that much at every height. */
  #c{display:block;width:100%;height:100vh;background:__PLOTBG__;cursor:grab}
  #c.grabbing{cursor:grabbing}
  .edge{stroke:#e2e8f0;stroke-width:1}
  .node{stroke:#fff;stroke-width:1;cursor:pointer}
  .node:hover{stroke:__TEXT__;stroke-width:1.5}
  .lbl{font-size:12px;fill:__TEXT__;pointer-events:none;text-anchor:middle;font-family:__FONT__}
  /* top/left matter: with no offsets an absolutely positioned box keeps its
     static position, which is directly below the 100vh svg, so the tooltip
     pushed the page 14px past the viewport before JS ever moved it. */
  #tip{position:absolute;top:0;left:0;pointer-events:none;background:#fff;border:1px solid #e2e8f0;
       border-radius:6px;padding:6px 9px;font-size:15.5px;color:__TEXT__;
       box-shadow:0 2px 8px rgba(0,0,0,.08);opacity:0;transition:opacity .1s;
       white-space:nowrap;z-index:10;font-family:__FONT__}
  #hint{position:absolute;left:12px;bottom:10px;font-size:14.5px;color:#94a3b8;
        pointer-events:none;font-family:__FONT__;white-space:nowrap}
</style></head>
<body>
<svg id="c"></svg><div id="tip"></div>
<div id="hint">Drag a node to reposition it &middot; drag the background to pan &middot; scroll to zoom</div>
<script>
const DATA=__DATA__;
const svg=d3.select("#c"), tip=d3.select("#tip");
const pad=48;
const xs=DATA.nodes.map(d=>d.x), ys=DATA.nodes.map(d=>d.y);
const ex0=d3.extent(xs), ey0=d3.extent(ys);
const sx=d3.scaleLinear().domain(ex0[0]===ex0[1]?[ex0[0]-1,ex0[0]+1]:ex0),
      sy=d3.scaleLinear().domain(ey0[0]===ey0[1]?[ey0[0]-1,ey0[0]+1]:ey0);
function fit(){
  const W=window.innerWidth,H=window.innerHeight;
  sx.range([pad,W-pad]); sy.range([H-pad,pad]);
  DATA.nodes.forEach(n=>{n.px=sx(n.x);n.py=sy(n.y);});
}
fit();
const g=svg.append("g");
const edgeSel=g.append("g").selectAll("line").data(DATA.edges).join("line").attr("class","edge");
const nodeSel=g.append("g").selectAll("circle").data(DATA.nodes).join("circle")
  .attr("class","node").attr("r",d=>d.size/2).attr("fill",d=>d.color).attr("opacity",0.85);
const lblSel=g.append("g").selectAll("text").data(DATA.nodes).join("text")
  .attr("class","lbl").text(d=>d.label);
function refresh(){
  edgeSel.attr("x1",e=>DATA.nodes[e[0]].px).attr("y1",e=>DATA.nodes[e[0]].py)
         .attr("x2",e=>DATA.nodes[e[1]].px).attr("y2",e=>DATA.nodes[e[1]].py);
  nodeSel.attr("cx",d=>d.px).attr("cy",d=>d.py);
  lblSel.attr("x",d=>d.px).attr("y",d=>d.py-d.size/2-3);
}
refresh();
/* fit() maps the packed layout onto the frame, and until 2026-09-03 it ran
   exactly once. This page loads no Plotly, so the shared responsive script
   cannot help it: rotating a phone, or the parent resizing the frame after the
   chart reports its height, left the network laid out for the old size. */
let _rt=null;
window.addEventListener("resize",()=>{clearTimeout(_rt);_rt=setTimeout(()=>{fit();refresh();},120);});
svg.call(d3.zoom().scaleExtent([0.2,8]).on("zoom",ev=>g.attr("transform",ev.transform)))
   .on("mousedown.cur",()=>svg.classed("grabbing",true))
   .on("mouseup.cur",()=>svg.classed("grabbing",false));
nodeSel.call(d3.drag()
    .on("start",function(){d3.select(this).raise();})
    .on("drag",function(ev,d){
      const k=d3.zoomTransform(svg.node()).k;
      d.px+=ev.dx/k; d.py+=ev.dy/k; refresh();
    }))
  .on("mouseover",(ev,d)=>tip.html(d.hover).style("opacity",1)
      .style("left",(ev.clientX+12)+"px").style("top",(ev.clientY-10)+"px"))
  .on("mousemove",ev=>tip.style("left",(ev.clientX+12)+"px").style("top",(ev.clientY-10)+"px"))
  .on("mouseout",()=>tip.style("opacity",0));
window.addEventListener("resize",()=>{fit();refresh();});
</script></body></html>
"""


# How many of the most prolific authors get a visible name label on the network.
LABEL_TOP_N = 25


def _coauthor_graph(papers_yr: list, min_papers: int) -> tuple:
    """Co-authorship graph over authors with >= min_papers papers this year."""
    author_counts: Counter = Counter()
    for p in papers_yr:
        for a in p.get("authors", []):
            author_counts[a] += 1

    # Sorted, not a set. Node insertion order fixes G.nodes() order, which
    # decides which initial position spring_layout hands each author and how
    # ties are broken everywhere below. Python randomises string hashing per
    # process, so a set here made every build lay the network out differently
    # from identical data; the five coauthor_*.html files were the only ones
    # that changed between two consecutive builds. Sorting costs nothing and
    # is the whole fix. Verified by PYTHONHASHSEED=0 reproducing the old files.
    active_set = {a for a, c in author_counts.items() if c >= min_papers}
    active = sorted(active_set)

    G = nx.Graph()
    G.add_nodes_from(active)
    for p in papers_yr:
        authors_in = [a for a in p.get("authors", []) if a in active_set]
        for i in range(len(authors_in)):
            for j in range(i + 1, len(authors_in)):
                u, v = authors_in[i], authors_in[j]
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)
    return G, author_counts


def _detect_communities(G: "nx.Graph") -> dict:
    """{node: community index}, largest community first. Falls back to all-zero."""
    try:
        comms = nx.community.greedy_modularity_communities(G)
    except (
        Exception
    ) as exc:  # networkx version differences / degenerate graphs
        logger.warning(
            f"  Community detection unavailable ({exc}); single color"
        )
        return {n: 0 for n in G.nodes()}
    # Size descending, then by first member: size alone leaves equal-sized
    # communities in set order, which is not stable across processes.
    comms = sorted(comms, key=lambda c: (-len(c), min(c)))
    return {n: i for i, c in enumerate(comms) for n in c}


def _coauthor_layout(G: "nx.Graph") -> dict:
    """Spring-layout each connected component separately, then shelf-pack the
    components so small fragments don't overlap or crowd the main cluster."""
    # Same total ordering as the communities above, and for the same reason:
    # component order decides the shelf-packing, so ties must break the same
    # way on every run.
    comps = sorted(nx.connected_components(G), key=lambda c: (-len(c), min(c)))

    boxes = []
    for comp in comps:
        n = len(comp)
        if n == 1:
            sub_pos = {next(iter(comp)): np.array([0.0, 0.0])}
        else:
            # Build the component graph explicitly instead of using
            # G.subgraph(comp). That returns a filtered VIEW, and networkx
            # iterates whichever container is smaller: when a component is less
            # than half the graph it walks its own internal set of allowed
            # nodes, so the view's node order is set order, which Python varies
            # per process. spring_layout seeds initial positions by node index,
            # so an identical graph laid out identically still came out
            # different on every build. seed=42 does not help; it fixes the
            # random numbers, not which node receives which one.
            sub = nx.Graph()
            sub.add_nodes_from(sorted(comp))
            sub.add_edges_from(
                (u, v, d)
                for u, v, d in G.edges(data=True)
                if u in comp and v in comp
            )
            sub_pos = nx.spring_layout(
                sub, k=2.5 / max(n**0.5, 1), seed=42, iterations=50
            )
        xs = [pt[0] for pt in sub_pos.values()]
        ys = [pt[1] for pt in sub_pos.values()]
        span_x = max(max(xs) - min(xs), 1e-6)
        span_y = max(max(ys) - min(ys), 1e-6)
        # normalize to [0,1] then scale by sqrt(n): bigger components get more room
        scale = max(n**0.5, 1.0)
        norm = {
            node: (
                (pt[0] - min(xs)) / span_x * scale,
                (pt[1] - min(ys)) / span_y * scale,
            )
            for node, pt in sub_pos.items()
        }
        boxes.append({"pos": norm, "w": scale, "h": scale})

    # shelf-pack left-to-right, wrapping into rows
    row_limit = max(sum(b["w"] for b in boxes) ** 0.5 * 1.6, boxes[0]["w"])
    gap = 0.35
    pos: dict = {}
    cx = cy = row_h = 0.0
    for b in boxes:
        if cx > 0 and cx + b["w"] > row_limit:
            cx = 0.0
            cy -= row_h + gap
            row_h = 0.0
        for node, (x, y) in b["pos"].items():
            pos[node] = (cx + x, cy + y)
        cx += b["w"] + gap
        row_h = max(row_h, b["h"])
    return pos


def _coauthor_data(papers_yr: list, year: int, min_papers: int):
    """Shared model behind both renderers: (nodes, edges) or None if empty."""
    G, author_counts = _coauthor_graph(papers_yr, min_papers)
    if len(G.nodes) == 0:
        logger.info(
            f"  Skipping coauthor_{year}.html; no authors with >={min_papers} papers"
        )
        return None

    pos = _coauthor_layout(G)
    community = _detect_communities(G)
    order = list(G.nodes())
    index = {n: i for i, n in enumerate(order)}

    # Only the most prolific authors get a printed name; labelling all of them is
    # unreadable at this density. Everyone keeps their hover tooltip.
    ranked = sorted((author_counts[n] for n in order), reverse=True)
    label_threshold = ranked[min(LABEL_TOP_N - 1, len(ranked) - 1)]

    nodes = [
        {
            "x": float(pos[n][0]),
            "y": float(pos[n][1]),
            "size": 6 + author_counts[n] * 4,
            "color": CLUSTER_PALETTE[community[n] % len(CLUSTER_PALETTE)],
            "label": (
                n.split(",")[0] if author_counts[n] >= label_threshold else ""
            ),
            "hover": (
                f"{n}<br>{author_counts[n]} papers "
                f"&middot; {G.degree(n)} collaborators"
            ),
        }
        for n in order
    ]
    edges = [[index[u], index[v]] for u, v in G.edges()]
    return nodes, edges


def _render_coauthor_plotly(nodes: list, edges: list, year: int) -> None:
    """Legacy renderer; static figure, pan/zoom only (no node dragging)."""
    edge_x: list = []
    edge_y: list = []
    for a, b in edges:
        edge_x += [nodes[a]["x"], nodes[b]["x"], None]
        edge_y += [nodes[a]["y"], nodes[b]["y"], None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color=RULE, width=1),
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[n["x"] for n in nodes],
            y=[n["y"] for n in nodes],
            mode="markers+text",
            marker=dict(
                size=[n["size"] for n in nodes],
                color=[n["color"] for n in nodes],
                opacity=0.85,
                line=dict(color="#fff", width=1),
            ),
            text=[n["label"] for n in nodes],
            textposition="top center",
            textfont=dict(size=13, color=TEXT),
            hovertext=[n["hover"] for n in nodes],
            hoverinfo="text",
            showlegend=False,
        )
    )
    blank = dict(
        showticklabels=False, showgrid=False, zeroline=False, showline=False
    )
    fig.update_layout(
        base_layout(
            showlegend=False,
            xaxis=blank,
            yaxis=dict(**blank, scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, t=20, b=20),
            dragmode="pan",
        )
    )
    save_chart(fig, f"coauthor_{year}.html", interactive=True)


def _render_coauthor_d3(nodes: list, edges: list, year: int) -> None:
    """Interactive renderer; draggable nodes, pan, zoom. Same layout as Plotly."""
    save_template(
        D3_NETWORK_TEMPLATE,
        f"coauthor_{year}.html",
        {
            **theme_tokens(),
            "__TITLE__": f"MICCAI {year} co-authorship network",
            "__DATA__": json.dumps({"nodes": nodes, "edges": edges}),
        },
        note="  (d3)",
    )


def chart_coauthor(
    papers_yr: list,
    year: int,
    min_papers: int,
    renderer: str = "d3",
) -> None:
    """Co-authorship network. Node size = paper count, color = detected community.

    renderer is set globally by chart_settings.network_renderer in config.yaml:
      "d3"     - draggable nodes (default)
      "plotly" - legacy static figure
    Both share identical layout, sizing and colors; only the drawing differs.
    """
    data = _coauthor_data(papers_yr, year, min_papers)
    if data is None:
        return
    nodes, edges = data
    if renderer == "plotly":
        _render_coauthor_plotly(nodes, edges, year)
    else:
        _render_coauthor_d3(nodes, edges, year)


def chart_buzzwords(papers_yr: list, year: int) -> None:
    matches: dict[str, int] = {name: 0 for name in BUZZWORDS}
    n = len(papers_yr)

    for p in papers_yr:
        text = (
            p.get("title", "") + " " + (p.get("abstract", "") or "")
        ).lower()
        for name, patterns in BUZZWORDS.items():
            if any(re.search(pat, text, re.IGNORECASE) for pat in patterns):
                matches[name] += 1

    items = sorted(matches.items(), key=lambda x: x[1])
    labels = [x[0] for x in items]
    counts = [x[1] for x in items]
    pcts = [round(c / n * 100, 1) for c in counts]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker_color=year_main(year),
            customdata=pcts,
            hovertemplate="%{y}: %{x} papers (%{customdata:.1f}%)<extra></extra>",
        )
    )
    fig.update_layout(
        base_layout(
            margin=dict(l=200, r=80, t=30, b=50),
            xaxis=dict(title="Papers matching (may overlap)", gridcolor=RULE),
            yaxis=dict(tickfont=dict(size=14.5)),
        )
    )
    CHART_HEIGHTS[f"buzzwords_{year}"] = _bar_height(labels, 14.5)
    save_chart(fig, f"buzzwords_{year}.html")


FIRST_WORD_STOPWORDS = {
    # Articles / determiners
    "a",
    "an",
    "the",
    # Prepositions / function words
    "on",
    "in",
    "for",
    "of",
    "towards",
    "toward",
    "from",
    "via",
    "with",
    "by",
    "to",
    "at",
    # Generic claim words that add no domain signal
    "novel",
    "new",
}


def chart_naming(papers_yr: list, year: int) -> None:
    titles = [p.get("title", "") for p in papers_yr]

    # 1. Word counts
    word_counts = [len(t.split()) for t in titles]

    # 2. First words; filter articles, prepositions, and generic claim words
    first_words: Counter = Counter()
    for t in titles:
        words = t.split()
        if words:
            w = words[0].lower().rstrip(":")
            if w not in FIRST_WORD_STOPWORDS:
                first_words[w] += 1

    # Take enough candidates so we have 15 after any residual filtering
    top_first = [
        (w, c)
        for w, c in first_words.most_common(50)
        if w not in FIRST_WORD_STOPWORDS
    ][:15]
    fw_labels = [x[0].capitalize() for x in reversed(top_first)]
    fw_values = [x[1] for x in reversed(top_first)]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Title Length (word count)",
            "Most Common First Words",
        ),
        column_widths=[0.38, 0.62],
    )

    # Word count histogram + KDE
    bar, line = _hist_kde_traces(
        word_counts, year_main(year), discrete=True, hover_label="Words"
    )
    bar.hovertemplate = "Words: %{x}<br>Papers: %{y}<extra></extra>"
    fig.add_trace(bar, row=1, col=1)
    if line:
        fig.add_trace(line, row=1, col=1)

    # First words bar
    fig.add_trace(
        go.Bar(
            x=fw_values,
            y=fw_labels,
            orientation="h",
            marker_color=year_main(year),
            opacity=0.8,
            hovertemplate="%{y}: %{x} titles<extra></extra>",
            showlegend=False,
            name="First word",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        **base_layout(margin=dict(l=60, r=30, t=40, b=50)),
        showlegend=False,
        bargap=0,
        height=380,
    )
    fig.update_xaxes(gridcolor=RULE)
    fig.update_yaxes(gridcolor=RULE)
    save_chart(fig, f"naming_{year}.html")


# ---------------------------------------------------------------------------
# Cross-year chart functions
# ---------------------------------------------------------------------------


REVIEW_TEXT_FIELDS = (
    "text_contribution",
    "text_strengths",
    "text_weaknesses",
    "text_detailed_comments",
)


def _review_lengths(papers_yr: list) -> list:
    """Per-review character count (whitespace excluded) over the four free-text
    review fields. One entry per review, not per paper."""
    out = []
    for p in papers_yr:
        for r in p.get("reviews", []):
            out.append(
                sum(
                    len(re.sub(r"\s", "", (r.get(f) or "")))
                    for f in REVIEW_TEXT_FIELDS
                )
            )
    return out


def _verdict_side(rank: int) -> str:
    """Accept side (Weak Accept / Accept / Strong Accept) vs Reject side."""
    return "A" if rank >= VERDICT_RANK["Weak Accept"] else "R"


# The trends overview is a hand-written 3x3 CSS grid of nine independent Plotly
# divs rather than one make_subplots figure. Reason: each panel needs its own
# y-range headroom, its own margin, and (for two of them) annotations pinned to
# the panel top: all of which are far simpler per-figure than juggling nine
# axis objects, and the CSS grid reflows to 2 columns on narrow screens, which
# a subplot grid cannot do.
_TRENDS_GRID_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>MICCAI trends overview</title>
__PLOTLYSCRIPT__
<style>
html,body{margin:0;font-family:__FONT__;background:__BG__}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px 16px;padding:10px 12px}
.panel{display:flex;flex-direction:column}
.ptitle{font-size:16px;font-weight:600;color:__TEXT__;padding:2px 2px 0}
.ptitle .sub{font-weight:400;color:__MUTED__;font-size:14px}
.pchart{height:172px}
@media(max-width:820px){.grid{grid-template-columns:repeat(2,1fr)}}\n/* One column on a phone. At two columns a panel is about 150px wide, which\n   wraps every heading to three lines and runs the per-year value labels\n   together into an unreadable string. The frame reports its own height, so\n   nine stacked panels are not cropped. */\n@media(max-width:560px){.grid{grid-template-columns:1fr;gap:2px 0}}
</style></head><body>
<div class="grid">
__PANELS__
</div>
</body></html>
"""


def chart_trends_overview(
    papers: list, submissions: dict, year_cfg: dict
) -> None:
    """3x3 grid of small multiples of key metrics across years (index page).

    Panels 1-6 are plain per-year bars. Panels 7-8 (Avg Review Score, Avg Review
    Length) carry +/-1 SD whiskers, so their value labels are pinned to the top
    of the panel instead of riding on the bar (label and whisker collided).
    Panel 9 is the rebuttal accept/reject-crossing breakdown.
    """
    years = sorted(set(p["year"] for p in papers))
    yr_labels = [str(y) for y in years]

    # Each year keeps its designated logo color across all cross-year charts
    bar_colors = [YEAR_COLORS.get(yr, ACCENT) for yr in years]

    def _stats(yr):
        ps = [p for p in papers if p["year"] == yr]
        n = len(ps)
        authors = set(a for p in ps for a in p.get("authors", []))
        n_code = sum(1 for p in ps if p.get("has_code"))
        n_early = sum(1 for p in ps if p.get("early_accepted"))
        n_sub = submissions.get(yr)
        norm = [
            p["avg_score_normalized"]
            for p in ps
            if p.get("avg_score_normalized") is not None
        ]
        raw = [
            p["avg_score_raw"]
            for p in ps
            if p.get("avg_score_raw") is not None
        ]
        scale_max = year_cfg.get(yr, {}).get("review_scale_max", 6)
        lens = _review_lengths(ps)
        return {
            "n_submitted": n_sub,
            "n_papers": n,
            "accept_rate": round(n / n_sub * 100, 1) if n_sub else None,
            # Early accepts as a fraction of ALL submissions (~8-13%), not of
            # accepted papers; the denominator comes from config
            "pct_early": round(n_early / n_sub * 100, 1) if n_sub else None,
            "n_authors": len(authors),
            "pct_code": round(n_code / n * 100, 1) if n else 0,
            # Normalized x100 = "% of that year's scale". Comparable across the
            # 1-9 / 1-8 / 1-6 eras; the raw native mean is printed on the panel.
            "score_pct": round(float(np.mean(norm)) * 100, 1)
            if norm
            else None,
            "score_pct_sd": round(float(np.std(norm)) * 100, 1)
            if norm
            else None,
            "score_raw_lbl": f"{np.mean(raw):.2f}/{scale_max}" if raw else "n/a",
            "len_mean": int(round(float(np.mean(lens)))) if lens else None,
            "len_sd": int(round(float(np.std(lens)))) if lens else None,
        }

    st = {yr: _stats(yr) for yr in years}

    def _fmt_int(v):
        return f"{v:,.0f}"

    def _fmt_pct(v):
        return f"{v:.1f}%"

    # (key, title, subtitle, value formatter, SD key or None)
    panels = [
        ("n_submitted", "Submissions", "", _fmt_int, None),
        ("n_papers", "Papers Accepted", "", _fmt_int, None),
        ("accept_rate", "Acceptance Rate", "", _fmt_pct, None),
        (
            "pct_early",
            "Early Accepted",
            "(% of submissions)",
            _fmt_pct,
            None,
        ),
        ("n_authors", "Unique Authors", "", _fmt_int, None),
        ("pct_code", "Papers with Code", "", _fmt_pct, None),
        (
            "score_pct",
            "Avg Review Score",
            "(% of scale · ±SD · raw)",
            None,
            "score_pct_sd",
        ),
        (
            "len_mean",
            "Avg Review Length",
            "(num. chars. · ±SD)",
            _fmt_int,
            "len_sd",
        ),
    ]

    base_font = dict(family=FONT, size=14.5, color=TEXT)
    hoverlabel = dict(
        font=dict(family=FONT, size=14.5, color=TEXT),
        bgcolor="#fff",
        bordercolor=RULE,
    )
    connector = "rgba(100,116,139,0.6)"

    blocks = []
    for i, (key, title, sub, fmt, errkey) in enumerate(panels, start=1):
        vals = [st[yr][key] for yr in years]
        present = [v for v in vals if v is not None]
        if not present:
            continue
        errs = [st[yr][errkey] or 0 for yr in years] if errkey else None
        suffix = "%" if fmt is _fmt_pct else ""

        bar = {
            "type": "bar",
            "x": yr_labels,
            "y": vals,
            "marker": {"color": bar_colors},
            "cliponaxis": False,
            "hovertemplate": "%{x}: %{y}" + suffix + "<extra></extra>",
            "showlegend": False,
        }
        if errs:
            # Labels ride at the top of the panel, clear of the whiskers
            bar["error_y"] = {
                "type": "data",
                "array": errs,
                "visible": True,
                "color": "rgba(30,41,59,0.55)",
                "thickness": 1.3,
                "width": 4,
            }
            head = max(v + e for v, e in zip(vals, errs) if v is not None)
            headroom, top_margin = 1.34, 18
            labels = (
                [st[yr]["score_raw_lbl"] for yr in years]
                if key == "score_pct"
                else [
                    fmt(st[yr][key]) if st[yr][key] is not None else "n/a"
                    for yr in years
                ]
            )
            annotations = [
                {
                    "x": x,
                    "xref": "x",
                    "yref": "paper",
                    "yanchor": "top",
                    "y": 1,
                    "text": lbl,
                    "showarrow": False,
                    "font": {"size": 14, "color": TEXT},
                }
                for x, lbl in zip(yr_labels, labels)
            ]
        else:
            bar["text"] = [fmt(v) if v is not None else "n/a" for v in vals]
            bar["textposition"] = "outside"
            bar["textfont"] = {"size": 10.5, "color": TEXT}
            head = max(present)
            headroom, top_margin, annotations = 1.22, 10, []

        line = {
            "type": "scatter",
            "x": yr_labels,
            "y": vals,
            "mode": "lines+markers",
            "line": {"color": connector, "width": 1.4},
            "marker": {"size": 3.5, "color": connector},
            "hoverinfo": "skip",
            "showlegend": False,
        }
        layout = {
            "paper_bgcolor": BG,
            "plot_bgcolor": PLOT_BG,
            "font": base_font,
            "margin": {"l": 6, "r": 6, "t": top_margin, "b": 22},
            "xaxis": {
                "tickfont": {"size": 13.5},
                "showgrid": False,
                "linecolor": RULE,
            },
            "yaxis": {
                "showgrid": False,
                "showticklabels": False,
                "showline": False,
                "zeroline": False,
                "range": [0, round(head * headroom, 3)],
            },
            "hoverlabel": hoverlabel,
        }
        if annotations:
            layout["annotations"] = annotations
        blocks.append((i, title, sub, [bar, line], layout))

    # --- panel 9: does the rebuttal help? -----------------------------------
    # Share of reviewers who crossed the accept/reject line after rebuttal.
    # Only years with post-rebuttal VERDICT labels qualify (2024/2025); earlier
    # years recorded post-rebuttal values as raw scores on incompatible scales,
    # so "the score went up" is not comparable across eras; crossing the
    # accept/reject line is the one definition that holds everywhere it exists.
    flip_years, flip = [], {"pos": [], "none": [], "neg": []}
    for yr in years:
        pos = none = neg = tot = 0
        for p in (q for q in papers if q["year"] == yr):
            for r in p.get("reviews", []):
                pre = _verdict(r.get("recommendation_label"))
                post = _verdict(r.get("post_rebuttal_label"))
                if pre not in VERDICT_RANK or post not in VERDICT_RANK:
                    continue
                tot += 1
                s0 = _verdict_side(VERDICT_RANK[pre])
                s1 = _verdict_side(VERDICT_RANK[post])
                if s0 == "R" and s1 == "A":
                    pos += 1
                elif s0 == "A" and s1 == "R":
                    neg += 1
                else:
                    none += 1
        if tot < 30:  # too few comparable verdicts to report a percentage
            continue
        flip_years.append(str(yr))
        flip["pos"].append(round(pos / tot * 100, 1))
        flip["none"].append(round(none / tot * 100, 1))
        flip["neg"].append(round(neg / tot * 100, 1))

    if flip_years:
        span = f"{flip_years[0]}-{flip_years[-1][-2:]}"

        def _flip_bar(name, key, color, tcolor, hover):
            return {
                "type": "bar",
                "name": name,
                "x": flip_years,
                "y": flip[key],
                "marker": {"color": color},
                "text": [
                    f"{v:.0f}%" if v >= 10 else f"{v:.1f}%" for v in flip[key]
                ],
                "textposition": "outside",
                "textfont": {"size": 13.5, "color": tcolor},
                "cliponaxis": False,
                "hovertemplate": "%{x} · " + hover + ": %{y}%<extra></extra>",
            }

        blocks.append(
            (
                9,
                "Does Rebuttal Help?",
                f"(% reviewers, {span})",
                [
                    _flip_bar(
                        "→ Accept",
                        "pos",
                        "#16a34a",
                        "#166534",
                        "moved toward Accept",
                    ),
                    _flip_bar(
                        "No change", "none", "#94a3b8", "#475569", "no change"
                    ),
                    _flip_bar(
                        "→ Reject",
                        "neg",
                        "#dc2626",
                        "#991b1b",
                        "moved toward Reject",
                    ),
                ],
                {
                    "paper_bgcolor": BG,
                    "plot_bgcolor": PLOT_BG,
                    "barmode": "group",
                    "bargap": 0.3,
                    "bargroupgap": 0.1,
                    "font": base_font,
                    "margin": {"l": 8, "r": 8, "t": 24, "b": 22},
                    "xaxis": {
                        "tickfont": {"size": 13.5},
                        "showgrid": False,
                        "linecolor": RULE,
                    },
                    "yaxis": {
                        "showgrid": False,
                        "showticklabels": False,
                        "showline": False,
                        "zeroline": False,
                        "range": [0, 95],
                    },
                    "legend": {
                        "orientation": "h",
                        "x": 0.5,
                        "y": 1.16,
                        "xanchor": "center",
                        "yanchor": "top",
                        "font": {"size": 13},
                    },
                    "hoverlabel": hoverlabel,
                },
            )
        )

    html_panels = []
    for idx, title, sub, traces, layout in blocks:
        heading = title + (f" <span class='sub'>{sub}</span>" if sub else "")
        html_panels.append(
            f'<div class="panel"><div class="ptitle">{heading}</div>'
            f'<div class="pchart" id="p{idx}"></div>\n'
            f'<script>Plotly.newPlot("p{idx}",{json.dumps(traces)},'
            f"{json.dumps(layout)},"
            "{displayModeBar:false,responsive:true});</script></div>"
        )

    save_template(
        _TRENDS_GRID_TEMPLATE,
        "trends_overview.html",
        {**theme_tokens(), "__PANELS__": "\n".join(html_panels)},
    )


# ---------------------------------------------------------------------------


def chart_map_all(papers: list) -> None:
    years = sorted(set(p["year"] for p in papers))
    cluster_ids = sorted(set(p["cluster_id"] for p in papers))
    cluster_label_map = {p["cluster_id"]: p["cluster_label"] for p in papers}

    # Build two sets of traces: "by year" and "by cluster"
    # Use Plotly updatemenus to toggle visibility

    traces_year = []
    for yr in years:
        yps = [p for p in papers if p["year"] == yr]
        traces_year.append(
            go.Scattergl(
                x=[p["umap_x"] for p in yps],
                y=[p["umap_y"] for p in yps],
                mode="markers",
                name=str(yr),
                marker=dict(
                    color=YEAR_COLORS.get(yr, ACCENT),
                    size=4,
                    opacity=0.65,
                    line=dict(width=0),
                ),
                customdata=[[p["url"], p["title"]] for p in yps],
                hovertemplate="%{customdata[1]}<extra></extra>",
                visible=True,
            )
        )

    traces_cluster = []
    for cid in cluster_ids:
        cps = [p for p in papers if p["cluster_id"] == cid]
        color = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
        traces_cluster.append(
            go.Scattergl(
                x=[p["umap_x"] for p in cps],
                y=[p["umap_y"] for p in cps],
                mode="markers",
                name=cluster_label_map[cid],
                marker=dict(
                    color=color, size=4, opacity=0.65, line=dict(width=0)
                ),
                customdata=[[p["url"], p["title"]] for p in cps],
                hovertemplate="%{customdata[1]}<extra></extra>",
                visible=False,
            )
        )

    all_traces = traces_year + traces_cluster
    n_yr = len(traces_year)
    n_cl = len(traces_cluster)

    fig = go.Figure(
        data=all_traces,
        layout=base_layout(
            showlegend=True,
            legend=dict(
                font=dict(size=13),
                itemsizing="constant",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=RULE,
                borderwidth=1,
                x=1.01,
                xanchor="left",
                y=1,
                yanchor="top",
            ),
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                title="",
                showline=False,
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                title="",
                showline=False,
            ),
            # autoexpand=False pins the right margin so the plotting area does
            # NOT resize when toggling Year (5 short entries) vs Cluster (20 long
            # names) - otherwise Plotly re-fits the margin and the map jumps width.
            margin=dict(l=20, r=300, t=55, b=20, autoexpand=False),
            dragmode="pan",
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.02,
                    y=1.09,
                    xanchor="left",
                    buttons=[
                        dict(
                            label="Color by Year",
                            method="update",
                            args=[
                                {"visible": [True] * n_yr + [False] * n_cl},
                                {"showlegend": True},
                            ],
                        ),
                        dict(
                            label="Color by Cluster",
                            method="update",
                            args=[
                                {"visible": [False] * n_yr + [True] * n_cl},
                                {"showlegend": True},
                            ],
                        ),
                    ],
                    showactive=True,
                    bgcolor="#fff",
                    bordercolor=RULE,
                    font=dict(size=15.5),
                )
            ],
        ),
    )

    save_chart(fig, "map_all.html", interactive=True, extra_js=CLICK_JS)


def chart_subject_heatmap(papers: list) -> None:
    years, totals, share, _presence = _subject_shares(papers)
    top_areas = [a for a, _ in totals.most_common(30)]

    # z matrix: rows = areas, cols = years; values = % of that year's papers
    z = [
        [round(share[area][i], 1) for i in range(len(years))]
        for area in top_areas
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(yr) for yr in years],
            y=top_areas,
            colorscale=[[0, "#f0f9ff"], [0.5, "#0891b2"], [1, "#075985"]],
            hovertemplate="%{y}<br>%{x}: %{z:.1f}% of papers<extra></extra>",
            colorbar=dict(
                title=dict(text="% of year", side="right"),
                tickfont=dict(size=13.5),
            ),
        )
    )
    fig.update_layout(
        base_layout(
            margin=dict(l=250, r=80, t=30, b=50),
            yaxis=dict(
                tickfont=dict(size=13.5),
                autorange="reversed",
                **_trunc_ticks(top_areas),
            ),
            xaxis=dict(tickfont=dict(size=15.5)),
        )
    )
    save_chart(fig, "subject_heatmap.html")


# The subject-area taxonomy CHANGED between years (e.g. 2025 renamed "CT" to
# "CT / X-Ray" and added a "Deep Learning" catch-all; 2024 renamed CAD; the
# supervised-representation-learning family has a different spelling almost
# every year). Cross-year charts merge the obvious renames and drop catch-alls
# Otherwise areas appear to crash to 0% in the years using the other name.
# Per-year charts intentionally keep each year's own taxonomy.
SUBJECT_DROP = {
    "Other",
    "other",
    "Deep Learning",  # 2025-only catch-all (736 papers)
}
SUBJECT_ALIASES = {
    "CT": "CT / X-Ray",
    "X-Ray": "CT / X-Ray",
    "Computer Aided Diagnosis, Treatment Response, and Outcome Prediction": (
        "Computer Aided Diagnosis"
    ),
    "Image Formation and Reconstruction": "Image Reconstruction",
    "Transfer learning": "Transfer Learning",
    "Data efficient Learning": "Data Efficient Learning",
    "Model Generalizability": "Model Generalizability / Federated Learning",
    "Heart": "Cardiac",
    "Semi-/Weakly-/Un-/Self-supervised representation learning": (
        "Semi-/Weakly-/Self-supervised Learning"
    ),
    "Semi-/Weakly-/Un-/Self-supervised Representation Learning": (
        "Semi-/Weakly-/Self-supervised Learning"
    ),
    "Semi- / Weakly- / Self-supervised Learning": (
        "Semi-/Weakly-/Self-supervised Learning"
    ),
    "Self-supervised learning": "Semi-/Weakly-/Self-supervised Learning",
    "Semi-supervised learning": "Semi-/Weakly-/Self-supervised Learning",
    "Weakly supervised learning": "Semi-/Weakly-/Self-supervised Learning",
}


def _canon_subject(label: str) -> str | None:
    """Canonical cross-year subject name, or None for dropped catch-alls."""
    short = _short_subject(label)
    if short in SUBJECT_DROP:
        return None
    return SUBJECT_ALIASES.get(short, short)


def _subject_shares(papers: list) -> tuple:
    """Per-year canonical subject-area shares for cross-year charts.

    Counts UNIQUE papers per (year, area); alias merging must not double
    count papers that carried several tags of the same family.
    Returns (years, totals, share, presence):
      share[area]    = [% of that year's papers, one entry per year]
      presence[area] = number of years with at least one paper in the area
    """
    years = sorted(set(p["year"] for p in papers))
    year_area: dict[int, dict] = {yr: defaultdict(set) for yr in years}
    year_total = Counter(p["year"] for p in papers)
    for p in papers:
        canon = {
            c
            for c in (_canon_subject(s) for s in p.get("subject_areas", []))
            if c
        }
        for c in canon:
            year_area[p["year"]][c].add(p["paper_id"])

    totals: Counter = Counter()
    for yr in years:
        for area, ids in year_area[yr].items():
            totals[area] += len(ids)

    share = {
        area: [
            len(year_area[yr].get(area, ())) / year_total[yr] * 100
            for yr in years
        ]
        for area in totals
    }
    presence = {
        area: sum(1 for yr in years if year_area[yr].get(area))
        for area in totals
    }
    return years, totals, share, presence


def chart_subject_lines(papers: list) -> None:
    """Small-multiple line charts: top 12 subject areas, share of papers per year."""
    years, totals, share, presence = _subject_shares(papers)
    # Only areas that exist in EVERY year's taxonomy. A line that falls to 0%
    # because MICCAI dropped or renamed the tag is noise, not signal, and
    # allowing a single missing year was enough to let one through: "Image
    # Reconstruction" ran at 10-13% from 2021 to 2024 and then plunged to zero,
    # because MICCAI 2025 deleted that subject area. Reconstruction work did
    # not stop; 14.8% of 2025 papers still mention it, they are simply filed
    # under the "Deep Learning" catch-all and "Image Synthesis / Augmentation /
    # Super-Resolution" instead. Requiring presence in all years is the whole
    # fix; SUBJECT_ALIASES above handles the renames that ARE recoverable.
    eligible = [a for a in totals if presence[a] == len(years)]
    top = sorted(eligible, key=lambda a: -totals[a])[:12]
    yr_labels = [str(y) for y in years]

    fig = make_subplots(
        rows=4,
        cols=3,
        subplot_titles=[_trunc(a, 34) for a in top],
        horizontal_spacing=0.07,
        vertical_spacing=0.12,
    )
    for i, area in enumerate(top):
        row, col = i // 3 + 1, i % 3 + 1
        fig.add_trace(
            go.Scatter(
                x=yr_labels,
                y=[round(v, 1) for v in share[area]],
                mode="lines+markers",
                line=dict(color=ACCENT, width=2),
                marker=dict(size=5, color=ACCENT),
                hovertemplate=f"{area}<br>%{{x}}: %{{y:.1f}}% of papers<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT, size=14.5, color=TEXT),
        margin=dict(l=40, r=20, t=30, b=30),
        height=660,
        hoverlabel=dict(
            font=dict(family=FONT, size=15.5, color=TEXT),
            bgcolor="#fff",
            bordercolor=RULE,
        ),
    )
    fig.update_annotations(font=dict(size=14.5, color=TEXT))
    fig.update_xaxes(tickfont=dict(size=13.5), gridcolor=RULE, linecolor=RULE)
    fig.update_yaxes(
        tickfont=dict(size=13.5),
        gridcolor=RULE,
        ticksuffix="%",
        rangemode="tozero",
    )
    save_chart(fig, "subject_lines.html")


def chart_subject_bump(papers: list) -> None:
    """Bump chart: rank of the top 15 subject areas per year (1 = most papers)."""
    years, totals, share, presence = _subject_shares(papers)
    yr_labels = [str(y) for y in years]

    eligible = [a for a in totals if presence[a] == len(years)]
    top = sorted(eligible, key=lambda a: -totals[a])[:15]

    # Rank within each year among the eligible areas
    rank_per_year: list[dict] = []
    for i, _yr in enumerate(years):
        order = sorted(eligible, key=lambda a: -share[a][i])
        rank_per_year.append({a: r + 1 for r, a in enumerate(order)})

    # End-of-line labels via a text trace (not fig.add_annotation): a data-
    # anchored annotation combined with xshift makes Plotly blow out the
    # categorical axis autorange to fit the shifted text, crushing all the
    # real category ticks into a sliver on the left. text on the marker
    # itself has no such effect.
    fig = go.Figure()
    for i, area in enumerate(top):
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        ranks = [rank_per_year[j][area] for j in range(len(years))]
        text = [""] * (len(ranks) - 1) + [_trunc(area, 28)]
        fig.add_trace(
            go.Scatter(
                x=yr_labels,
                y=ranks,
                mode="lines+markers+text",
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color),
                text=text,
                textposition="middle right",
                textfont=dict(size=13.5, color=color),
                hovertemplate=f"{area}<br>%{{x}}: rank %{{y}}<extra></extra>",
                showlegend=False,
            )
        )

    # Clip the axis at rank 20 (reversed via range order; do NOT also set
    # autorange, it overrides the explicit range); lines that fall below
    # simply exit the bottom of the chart
    fig.update_layout(
        base_layout(
            margin=dict(l=50, r=240, t=20, b=40),
            xaxis=dict(gridcolor=RULE, tickfont=dict(size=15.5)),
            yaxis=dict(
                title="Rank by paper count",
                gridcolor=RULE,
                dtick=1,
                range=[20.5, 0.5],
            ),
            showlegend=False,
        )
    )
    save_chart(fig, "subject_bump.html")


# Hand-written page rather than pio.to_html: the From/To year window is two
# <select>s driving Plotly.react. Every from<to window is precomputed in Python
# and embedded as JSON, so switching windows is a redraw, never a recompute.
_MOVERS_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>Biggest movers by subject area</title>
__PLOTLYSCRIPT__
<style>
html,body{margin:0;font-family:__FONT__;background:__BG__;color:__TEXT__}
.bar-ctrls{display:flex;align-items:center;gap:10px;padding:8px 14px 2px;
           font-size:16px;color:#475569}
.bar-ctrls select{font:inherit;padding:3px 8px;border:1px solid #cbd5e1;
                  border-radius:6px;background:#fff;color:__TEXT__}
.bar-ctrls .sep{color:__MUTED2__}
/* The plot fills whatever the control row leaves, using flex rather than
   calc(100vh - 42px). That 42px was the control row's height at desktop
   widths; when the controls wrap to two lines on a phone the row is 58px and
   the content became 100vh + 16px, i.e. taller than the frame. Since the
   frame now sizes itself from the content it reports, that was a loop: every
   height message made the page 16px taller again. Flex cannot overflow. */
html,body{height:100%}
body{display:flex;flex-direction:column}
.bar-ctrls{flex:0 0 auto;flex-wrap:wrap}
#mv{width:100%;flex:1 1 auto;min-height:0}\n/* On a phone these charts have more category rows than a viewport-height plot
   can give 26px each, so the y labels overprint each other. Let the page grow
   past the frame and report its real height to the parent instead. A fixed
   pixel min-height, never a vh value: the content height must not depend on
   the frame height, or the height message and the frame resize each other in a
   loop, which is exactly what calc(100vh - 42px) used to do here. */
@media(max-width:760px){
  html,body{height:auto}
  body{display:block}
  #mv{height:auto;min-height:780px}
}

</style></head><body>
<div class="bar-ctrls">
  <span>Window:</span>
  <label>From <select id="fromY"></select></label>
  <span class="sep">&rarr;</span>
  <label>To <select id="toY"></select></label>
  <span id="warn" style="color:#b45309"></span>
</div>
<div id="mv"></div>
<script>
const YEARS=__YEARS__;
const WIN=__WINDOWS__;
const ACCENT="__ACCENT__", NEG="#e11d48", MUTED="__MUTED__", RULE="__RULE__";
function wrap(s,n){ if(s.length<=n)return s; const w=s.split(" ");let line="",out=[];
 for(const x of w){ if((line+" "+x).trim().length>n&&line){out.push(line);line=x;}
  else line=(line+" "+x).trim();
  if(out.length===1&&line.length>n){line=line.slice(0,n-1).trim()+"\\u2026";break;} }
 out.push(line); return out.slice(0,2).join("<br>"); }
function draw(){
 const f=+fromY.value, t=+toY.value;
 const key=f+"_"+t; const warn=document.getElementById("warn");
 if(!WIN[key]){ warn.textContent="From year must be before To year"; return; }
 warn.textContent="";
 const m=WIN[key];
 const disp=m.map(d=>wrap(d.a,34)), vals=m.map(d=>d.d);
 const colors=vals.map(v=>v<0?NEG:ACCENT);
 const trace={type:"bar",orientation:"h",x:vals,y:disp,
   marker:{color:colors},
   text:vals.map(v=>(v>0?"+":"")+v.toFixed(1)),textposition:"outside",
   textfont:{size:13.5},cliponaxis:false,
   customdata:m.map(d=>[d.a,d.s0,d.s1]),
   hovertemplate:"%{customdata[0]}<br>"+f+": %{customdata[1]:.1f}% \\u2192 "
     +t+": %{customdata[2]:.1f}%<br>Change: %{x:+.1f} pp<extra></extra>"};
 const layout={paper_bgcolor:"__BG__",plot_bgcolor:"__PLOTBG__",
   font:{family:"__FONT__",size:15.5,color:"__TEXT__"},
   margin:{l:240,r:44,t:10,b:64},
   xaxis:{title:{text:"Change in share of papers, "+f+" \\u2192 "+t
     +" (percentage points)",font:{size:15.5,color:MUTED}},
     gridcolor:RULE,zeroline:false},
   yaxis:{tickfont:{size:14},automargin:false},
   shapes:[{type:"line",x0:0,x1:0,yref:"paper",y0:0,y1:1,
            line:{color:MUTED,width:1}}],
   hoverlabel:{font:{family:"__FONT__",size:15.5,color:"__TEXT__"},
               bgcolor:"#fff",bordercolor:RULE}};
 Plotly.react("mv",[trace],layout,{displayModeBar:false,responsive:true});
}
YEARS.forEach(y=>{ fromY.add(new Option(y,y)); toY.add(new Option(y,y)); });
// Opens on first year to SECOND-last, not the full span. MICCAI replaced its
// subject-area taxonomy in the last year, so a window ending there ranks the
// relabelling rather than the research; see the note card on index.html.
// Math.max keeps this valid when there are only two years.
fromY.value=YEARS[0]; toY.value=YEARS[Math.max(1,YEARS.length-2)];
fromY.onchange=draw; toY.onchange=draw;
draw();
</script>
</body></html>
"""


def chart_subject_movers(papers: list) -> None:
    """Diverging bar: change in share of papers over a selectable year window.

    Every from<to window is precomputed here; the page switches between them
    client-side, so the reader can ask "what moved between 2022 and 2024?"
    rather than only seeing the fixed first-to-last-year comparison.
    """
    years, totals, share, _presence = _subject_shares(papers)

    def _movers(fi: int, ti: int) -> list:
        # Enough volume for the change to be meaningful, and present in BOTH
        # endpoint years; an area absent from one endpoint's taxonomy would
        # otherwise show a fake ±full-share swing.
        rows = [
            {
                "a": a,
                "d": round(share[a][ti] - share[a][fi], 1),
                "s0": round(share[a][fi], 1),
                "s1": round(share[a][ti], 1),
            }
            for a in share
            if totals[a] >= 30 and share[a][fi] > 0 and share[a][ti] > 0
        ]
        rows.sort(key=lambda r: r["d"])
        return rows[:8] + rows[-8:]  # 8 biggest fallers + 8 biggest risers

    windows = {
        f"{years[fi]}_{years[ti]}": _movers(fi, ti)
        for fi in range(len(years))
        for ti in range(fi + 1, len(years))
    }

    save_template(
        _MOVERS_TEMPLATE,
        "subject_movers.html",
        {
            **theme_tokens(),
            "__MUTED2__": "#94a3b8",
            "__YEARS__": json.dumps(years),
            "__WINDOWS__": json.dumps(windows),
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Plotly chart HTML files"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Years to build per-year charts for (default: all in data)",
    )
    args = parser.parse_args()

    with open(ALL_JSON, encoding="utf-8") as f:
        papers = json.load(f)
    logger.info(f"Loaded {len(papers)} papers")

    # Round the UMAP coordinates, in memory only, exactly like the cluster
    # labels and presentation types below; miccai_all.json is never rewritten.
    #
    # embed.py writes these as float64 repr of a float32, so every one arrives
    # as 17 significant digits ("5.7215704917907715") and there are two per
    # paper per trace. map_all.html draws all 3,717 papers twice, once coloured
    # by year and once by cluster, which is 14,924 of those numbers and made it
    # the largest file on the site: 1.6 MB, 435 KB gzipped, and the first chart
    # on the home page, above the fold where lazy loading cannot help.
    #
    # The coordinates span about 10.5 units, so 0.001 is 0.086 px across a
    # 900 px frame and stays sub-pixel past 10x zoom. Nothing else reads
    # umap_x/umap_y, so this touches the seven maps and nothing else.
    for p in papers:
        for k in ("umap_x", "umap_y"):
            if p.get(k) is not None:
                p[k] = round(p[k], 3)

    cluster_labels: dict[str, str] = {}
    if CL_JSON.exists():
        with open(CL_JSON, encoding="utf-8") as f:
            cluster_labels = json.load(f)
        # Patch cluster_label on every paper so all chart functions see the current names
        for p in papers:
            cid = p.get("cluster_id")
            if cid is not None:
                p["cluster_label"] = cluster_labels.get(
                    str(cid), f"Cluster {cid}"
                )

    # Oral / spotlight types, patched in memory exactly like the cluster labels
    # above - miccai_all.json is never rewritten, so re-running normalize.py
    # cannot drop them. Absent orals.json, every type is None and the orals
    # charts are skipped; the rest of the build is unaffected.
    # Imported here rather than at module scope: build_oral_charts imports the
    # helpers defined in this file, so a top-level import would be circular.
    import oral_stats

    orals_data = oral_stats.load_orals()
    n_orals = oral_stats.attach_types(papers, orals_data)
    if n_orals:
        logger.info(f"Attached presentation types to {n_orals} papers")

    years_in_data = sorted(set(p["year"] for p in papers))
    target_years = args.years if args.years else years_in_data
    logger.info(f"Building per-year charts for: {target_years}")

    year_cfg = load_year_config()
    chart_settings = load_chart_settings()
    network_renderer = chart_settings["network_renderer"]
    logger.info(f"Co-authorship network renderer: {network_renderer}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-year charts
    for year in target_years:
        papers_yr = [p for p in papers if p["year"] == year]
        if not papers_yr:
            logger.warning(f"No papers found for {year}, skipping")
            continue
        logger.info(f"--- {year}: {len(papers_yr)} papers ---")

        cfg = year_cfg.get(year, {})
        scale_max = cfg.get("review_scale_max", 6)
        coauthor_min = cfg.get("coauthorship_network_min_papers", 2)

        # A year whose reviews carry post-rebuttal verdicts gets a Sankey, and
        # sankey lives only in the full plotly bundle, so every chart on that
        # year's page loads the full bundle too. Splitting the page instead
        # means downloading three bundles rather than one; see
        # _PAGE_BUNDLE_FLOOR for the measurement.
        floor = "full" if _rebuttal_flows(papers_yr) else None
        with page_bundle_floor(floor):
            chart_map(papers_yr, year)
            chart_subjects(papers_yr, year)
            chart_scores(papers_yr, year, scale_max)
            chart_controversy(papers_yr, year, scale_max)
            wrote_sankey = chart_rebuttal_sankey(papers_yr, year)
            if not wrote_sankey:
                # Clean up any stale chart from a previous build
                (OUT_DIR / f"rebuttal_{year}.html").unlink(missing_ok=True)
            chart_code(papers_yr, year)
            chart_authors(papers_yr, year)
            chart_coauthor(papers_yr, year, coauthor_min, network_renderer)
            chart_buzzwords(papers_yr, year)
            chart_naming(papers_yr, year)

    # Cross-year charts (use all papers in data, not just target_years)
    logger.info("--- Cross-year charts ---")
    chart_map_all(papers)
    chart_subject_lines(papers)
    chart_subject_movers(papers)
    chart_trends_overview(papers, load_submissions(), year_cfg)

    # The bump chart and the heatmap restate exactly the same rise/fall data as
    # Trending Topics + Biggest Movers, so their index cards were dropped in the
    # 2026-08 redesign. chart_subject_bump and
    # chart_subject_heatmap are kept intact; re-enable both the call here and
    # the card in templates/index.html to bring either view back.
    for stale in ("subject_bump.html", "subject_heatmap.html"):
        (OUT_DIR / stale).unlink(missing_ok=True)

    # Orals & spotlights page (no-op without data/processed/orals.json)
    import build_oral_charts

    build_oral_charts.build_all(papers, year_cfg)

    # Hand the computed frame heights to build_site.py. Written even when
    # empty so a partial --years run cannot leave a stale file behind.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(HEIGHTS_JSON, "w", encoding="utf-8") as f:
        json.dump(CHART_HEIGHTS, f, indent=1, sort_keys=True)
    logger.info(f"  Saved {HEIGHTS_JSON.name} ({len(CHART_HEIGHTS)} charts)")

    logger.info(f"All charts saved to {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
