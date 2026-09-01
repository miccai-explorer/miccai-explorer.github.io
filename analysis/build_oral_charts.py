#!/usr/bin/env python3
"""
Charts for the Orals & Spotlights page → website/charts/orals_*.html

Kept out of build_charts.py, which is already long: the oral analysis has its
own colour scheme, its own statistics and its own page, and nothing else in the
site depends on it. build_charts.py calls build_all() at the end of its run, so
`python analysis/build_charts.py` still builds the whole site in one command.

Everything here degrades to a no-op when data/processed/orals.json is absent, so
a checkout without the oral data still builds cleanly.

Design notes
------------
Type colours come from oral_stats.TYPE_COLORS rather than the per-year logo
palette: a reader comparing types across years needs "oral" to look the same in
every panel. Poster-only is deliberately the recessive grey; it is the baseline
the other two are being compared against, not a third highlight.

Following DESIGN.md, no figure sets layout.title; each chart is embedded in a
card whose head already carries the title and subtitle.
"""

import json
import logging
from collections import defaultdict

import numpy as np
import oral_stats as S
import plotly.graph_objects as go
from build_charts import (
    CLICK_JS,
    MUTED,
    OUT_DIR,
    RULE,
    TEXT,
    YEAR_COLORS,
    base_layout,
    save_chart,
    save_template,
    theme_tokens,
)
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

TYPES = S.TYPE_ORDER
LBL = S.TYPE_LABEL
COL = S.TYPE_COLORS

# Chart type scale. Named rather than inline because these three are tuned
# together against the iframe heights; see DESIGN.md "Chart typography".
PIN_SIZE = 13.5     # value labels pinned to the top of a panel
LEGEND_SIZE = 14.5


def _years_with_orals(papers: list) -> list:
    return sorted(
        {
            p["year"]
            for p in papers
            if p.get("presentation_type") in ("oral", "spotlight")
        }
    )


# ---------------------------------------------------------------------------
# 1. Overview: how many, and what share of the program
# ---------------------------------------------------------------------------


def chart_overview(papers: list, years: list) -> None:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Presentations by type",
            "Share of accepted papers selected",
        ),
        horizontal_spacing=0.12,
    )

    for ptype in ("oral", "spotlight"):
        counts = []
        for y in years:
            counts.append(
                sum(
                    1
                    for p in papers
                    if p["year"] == y and p.get("presentation_type") == ptype
                )
            )
        fig.add_trace(
            go.Bar(
                x=[str(y) for y in years],
                y=counts,
                name=LBL[ptype],
                marker_color=COL[ptype],
                text=[str(c) if c else "" for c in counts],
                textposition="inside",
                textfont=dict(color="#fff", size=14.5),
                hovertemplate="%{x} " + LBL[ptype] + ": %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    rates = []
    for y in years:
        sub = [p for p in papers if p["year"] == y]
        sel = sum(
            1
            for p in sub
            if p.get("presentation_type") in ("oral", "spotlight")
        )
        rates.append(100 * sel / len(sub) if sub else 0)
    fig.add_trace(
        go.Bar(
            x=[str(y) for y in years],
            y=rates,
            marker_color=[YEAR_COLORS.get(y, "#64748b") for y in years],
            text=[f"{r:.1f}%" for r in rates],
            textposition="outside",
            textfont=dict(size=14.5),
            showlegend=False,
            hovertemplate="%{x}: %{y:.1f}% of accepted papers<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(title_text="Papers", row=1, col=1, gridcolor=RULE)
    fig.update_yaxes(
        title_text="% of accepted",
        row=1,
        col=2,
        gridcolor=RULE,
        range=[0, max(rates) * 1.28 if rates else 1],
    )
    fig.update_xaxes(gridcolor=RULE)
    fig.update_layout(
        **base_layout(
            barmode="stack",
            margin=dict(l=55, r=25, t=76, b=45),
            legend=dict(
                orientation="h", y=1.24, x=0, font=dict(size=14.5)
            ),
        )
    )
    for a in fig.layout.annotations:
        a.font.size = 15.5
        a.font.color = TEXT
    save_chart(fig, "orals_overview.html")


# ---------------------------------------------------------------------------
# 2. Review scores by type
# ---------------------------------------------------------------------------


_GROUP_W = 0.78  # total width one year's group of bars may occupy
# Poster's fill colour is a deliberately recessive grey; as *text* it is too
# faint to read, so the pinned labels use the darker muted tone instead.
_LABEL_COLOR = {
    "oral": COL["oral"],
    "spotlight": COL["spotlight"],
    "poster": MUTED,
}


def _group_positions(year_idx: int, n_present: int) -> tuple:
    """Bar centres and bar width for n_present bars on one year's slot.

    Positions are set explicitly (with barmode="overlay") rather than left to
    Plotly's "group" mode for two reasons: the value labels are pinned to the
    top of the panel and have to sit exactly above their own bar, and "group"
    reserves a slot for every trace at every x, so 2021-2023 -- which have no
    spotlight type -- would show a gap where the missing bar would have been.

    The group always spans the same width, divided by however many bars are
    present. Holding the *bar* width constant instead packed two-bar years
    closer together than three-bar years, which is backwards: it left the
    pinned labels of 2021-2023 touching each other while the three-bar years
    had room to spare.
    """
    width = _GROUP_W / n_present
    start = year_idx - _GROUP_W / 2
    return (
        [start + (j + 0.5) * width for j in range(n_present)],
        width * 0.88,
    )


def chart_scores(papers: list, years: list, year_cfg: dict) -> None:
    """Mean review score per type per year.

    Follows the Avg Review Score panel on the overview page: bars are the mean
    as a percentage of that year's own scale (the scale changed 1-9 -> 1-8 ->
    1-6, so raw means are not comparable across years), the mean on the raw
    native scale is pinned to the top of the panel as "6.54/9" regardless of
    bar height, and the hover carries mean +/- 1 SD.

    The whiskers stay 95% bootstrap CIs rather than +/-1 SD: this chart exists
    to compare types, and an interval that answers "where is the mean" is the
    one to draw next to a comparison. The SD is in the hover for spread.
    """
    smax_of = {
        y: year_cfg.get(y, {}).get("review_scale_max", 6) for y in years
    }

    stats: dict = {}
    for y in years:
        sub = [p for p in papers if p["year"] == y]
        for ptype in TYPES:
            vals = [
                100 * v
                for v in S.type_values(sub, ptype, "avg_score_normalized")
            ]
            raws = S.type_values(sub, ptype, "avg_score_raw")
            if not vals:
                continue
            pt, lo, hi = S.bootstrap_ci(vals)
            stats[(y, ptype)] = {
                "mean": pt,
                "sd": float(np.std(vals)),
                "lo": lo,
                "hi": hi,
                "n": len(vals),
                "raw": float(np.mean(raws)) if raws else float("nan"),
            }

    pos: dict = {}
    bar_w: dict = {}
    for i, y in enumerate(years):
        present = [t for t in TYPES if (y, t) in stats]
        centres, width = _group_positions(i, len(present))
        for ptype, x in zip(present, centres):
            pos[(y, ptype)] = x
            bar_w[(y, ptype)] = width

    fig = go.Figure()
    annotations = []
    for ptype in TYPES:
        keys = [(y, ptype) for y in years if (y, ptype) in stats]
        if not keys:
            continue
        st = [stats[k] for k in keys]
        fig.add_trace(
            go.Bar(
                x=[pos[k] for k in keys],
                y=[d["mean"] for d in st],
                width=[bar_w[k] for k in keys],
                name=LBL[ptype],
                marker_color=COL[ptype],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[
                        0 if np.isnan(d["hi"]) else d["hi"] - d["mean"]
                        for d in st
                    ],
                    arrayminus=[
                        0 if np.isnan(d["lo"]) else d["mean"] - d["lo"]
                        for d in st
                    ],
                    color=MUTED,
                    thickness=1.3,
                    width=3,
                ),
                customdata=[
                    f"{LBL[ptype]} {y}"
                    f"<br>{d['mean']:.1f} \u00b1 {d['sd']:.1f}% of scale"
                    f"<br>95% CI [{d['lo']:.1f}, {d['hi']:.1f}]"
                    f" \u00b7 n = {d['n']}"
                    for (y, _), d in zip(keys, st)
                ],
                hovertemplate="%{customdata}<extra></extra>",
            )
        )
        annotations += [
            dict(
                x=pos[k],
                xref="x",
                yref="paper",
                y=1.0,
                yanchor="top",
                text=f"{d['raw']:.2f}/{smax_of[k[0]]}",
                showarrow=False,
                font=dict(size=PIN_SIZE, color=_LABEL_COLOR[ptype]),
            )
            for k, d in zip(keys, st)
        ]

    # No in-chart caption: the card above the chart already explains the pinned
    # labels and the whiskers, and repeating it inside the frame cost a line and
    # crowded the top tick.
    fig.update_layout(
        **base_layout(
            barmode="overlay",
            margin=dict(l=60, r=25, t=62, b=45),
            legend=dict(orientation="h", y=1.20, x=0, font=dict(size=LEGEND_SIZE)),
            yaxis=dict(
                title="Mean review score (% of that year's scale)",
                gridcolor=RULE,
                range=[0, 100],
            ),
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(years))),
                ticktext=[str(y) for y in years],
                range=[-0.5, len(years) - 0.5],
                gridcolor=RULE,
            ),
            annotations=annotations,
        )
    )
    save_chart(fig, "orals_scores.html")


# ---------------------------------------------------------------------------
# 3. Effect sizes: the centrepiece
# ---------------------------------------------------------------------------

_COMPARISONS = [
    ("oral", "poster", "Oral vs poster-only"),
    ("spotlight", "poster", "Spotlight vs poster-only"),
    ("oral", "spotlight", "Oral vs spotlight"),
]
_CMP_COLOR = {
    "Oral vs poster-only": COL["oral"],
    "Spotlight vs poster-only": COL["spotlight"],
    "Oral vs spotlight": "#d97706",
}


def chart_effect(papers: list, years: list) -> None:
    """Forest plot of Cliff's delta with bootstrap CIs.

    Cliff's delta answers "how much more often does a paper from group A
    outscore one from group B", which is meaningful on an ordinal scale and
    comparable between years whose review scales differ.
    """
    fig = go.Figure()

    # Magnitude bands (Romano et al.) drawn behind the points.
    for lo, hi, shade in (
        (-0.147, 0.147, "rgba(148,163,184,0.13)"),
        (0.147, 0.33, "rgba(148,163,184,0.08)"),
        (-0.33, -0.147, "rgba(148,163,184,0.08)"),
    ):
        fig.add_vrect(
            x0=lo, x1=hi, fillcolor=shade, line_width=0, layer="below"
        )
    fig.add_vline(x=0, line=dict(color=MUTED, width=1.2, dash="dot"))

    rows: list = []
    for a_t, b_t, label in _COMPARISONS:
        for y in years:
            sub = [p for p in papers if p["year"] == y]
            a = S.type_values(sub, a_t, "avg_score_raw")
            b = S.type_values(sub, b_t, "avg_score_raw")
            if len(a) < 3 or len(b) < 3:
                continue
            d, lo, hi, mag = S.cliffs_delta_ci(a, b)
            rows.append((label, y, d, lo, hi, mag, len(a), len(b)))

    # One y slot per (comparison, year), grouped by comparison.
    ticks, tickvals = [], []
    pos = 0
    for _, _, label in _COMPARISONS:
        entries = [r for r in rows if r[0] == label]
        if not entries:
            continue
        xs, ys, eh, el, hover = [], [], [], [], []
        for label_, y, d, lo, hi, mag, na, nb in entries:
            pos += 1
            xs.append(d)
            ys.append(pos)
            eh.append(0 if np.isnan(hi) else hi - d)
            el.append(0 if np.isnan(lo) else d - lo)
            ticks.append(str(y))
            tickvals.append(pos)
            hover.append(
                f"{label_} - {y}<br>Cliff's δ = {d:+.3f}"
                f"<br>95% CI [{lo:+.3f}, {hi:+.3f}]<br>{mag} effect"
                f"<br>n = {na} vs {nb}"
            )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=label,
                marker=dict(size=9, color=_CMP_COLOR[label]),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=eh,
                    arrayminus=el,
                    color=_CMP_COLOR[label],
                    thickness=1.6,
                    width=4,
                ),
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
            )
        )
        pos += 1  # blank slot between comparison groups

    fig.update_layout(
        **base_layout(
            margin=dict(l=60, r=30, t=52, b=48),
            legend=dict(orientation="h", y=1.13, x=0, font=dict(size=14.5)),
            xaxis=dict(
                title="Cliff's δ  (0 = no difference, + = first group scores higher)",
                gridcolor=RULE,
                zeroline=False,
                range=[-0.25, 1.0],
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticks,
                gridcolor=RULE,
                autorange="reversed",
            ),
        )
    )
    save_chart(fig, "orals_effect.html")


# ---------------------------------------------------------------------------
# 4. Early accepts: over-representation, and the confound it creates
# ---------------------------------------------------------------------------


def _type_rate_chart(
    papers: list,
    years: list,
    predicate,
    *,
    y_title: str,
    hover_verb: str,
    filename: str,
) -> None:
    """A "share of each type, per year" bar chart with Wilson intervals.

    Shared by the early-acceptance and code-release charts, which are the same
    figure with a different predicate. Both follow the score chart's layout:
    the bar carries the rate (comparable across types and years) and the counts
    it was computed from are pinned to the top of the panel, clear of the
    whiskers. Pinning the counts also keeps the denominator visible, which is
    the whole confusion these charts invite ("43 of 60" reads very differently
    depending on whether the 60 is orals or early accepts).
    """
    stats: dict = {}
    for y in years:
        for ptype, (rate, lo, hi, k, n) in S.rate_by_type(
            papers, y, predicate
        ).items():
            stats[(y, ptype)] = {
                "rate": 100 * rate,
                "lo": 100 * lo,
                "hi": 100 * hi,
                "k": k,
                "n": n,
            }

    pos: dict = {}
    bar_w: dict = {}
    for i, y in enumerate(years):
        present = [t for t in TYPES if (y, t) in stats]
        centres, width = _group_positions(i, len(present))
        for ptype, x in zip(present, centres):
            pos[(y, ptype)] = x
            bar_w[(y, ptype)] = width

    fig = go.Figure()
    annotations = []
    for ptype in TYPES:
        keys = [(y, ptype) for y in years if (y, ptype) in stats]
        if not keys:
            continue
        st = [stats[k] for k in keys]
        fig.add_trace(
            go.Bar(
                x=[pos[k] for k in keys],
                y=[d["rate"] for d in st],
                width=[bar_w[k] for k in keys],
                name=LBL[ptype],
                marker_color=COL[ptype],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[d["hi"] - d["rate"] for d in st],
                    arrayminus=[d["rate"] - d["lo"] for d in st],
                    color=MUTED,
                    thickness=1.3,
                    width=3,
                ),
                customdata=[
                    f"{LBL[ptype]} {y}"
                    f"<br>{d['rate']:.1f}% {hover_verb}"
                    f"<br>{d['k']} of {d['n']} {LBL[ptype].lower()} papers"
                    f" \u00b7 95% CI [{d['lo']:.1f}, {d['hi']:.1f}]"
                    for (y, _), d in zip(keys, st)
                ],
                hovertemplate="%{customdata}<extra></extra>",
            )
        )
        annotations += [
            dict(
                x=pos[k],
                xref="x",
                yref="paper",
                y=1.0,
                yanchor="top",
                text=f"{d['k']}/{d['n']}",
                showarrow=False,
                font=dict(size=PIN_SIZE, color=_LABEL_COLOR[ptype]),
            )
            for k, d in zip(keys, st)
        ]

    fig.update_layout(
        **base_layout(
            barmode="overlay",
            margin=dict(l=60, r=25, t=62, b=45),
            legend=dict(
                orientation="h", y=1.20, x=0, font=dict(size=LEGEND_SIZE)
            ),
            yaxis=dict(title=y_title, gridcolor=RULE, range=[0, 100]),
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(years))),
                ticktext=[str(y) for y in years],
                range=[-0.5, len(years) - 0.5],
                gridcolor=RULE,
            ),
            annotations=annotations,
        )
    )
    save_chart(fig, filename)


def chart_early(papers: list, years: list) -> None:
    _type_rate_chart(
        papers,
        years,
        lambda p: p.get("early_accepted"),
        y_title="Early-accepted share of that type (%)",
        hover_verb="were early-accepted",
        filename="orals_early.html",
    )


def chart_strata(papers: list, years: list) -> None:
    """Oral-vs-poster effect, pooled and split by early-accept status.

    Early-accepted papers are decided before rebuttal and orals are chosen from
    accepted papers, so an oral cohort loaded with early accepts could show a
    score gap with no separate 'orals score higher' effect at all. Splitting the
    comparison inside each stratum removes that explanation.
    """
    series = [
        ("All papers", None, COL["oral"]),
        ("Early-accepted only", True, "#059669"),
        ("Not early-accepted", False, "#d97706"),
    ]
    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color=MUTED, width=1.2, dash="dot"))
    for lo_, hi_, shade in ((-0.147, 0.147, "rgba(148,163,184,0.13)"),):
        fig.add_vrect(
            x0=lo_, x1=hi_, fillcolor=shade, line_width=0, layer="below"
        )

    ticks, tickvals = [], []
    pos = 0
    for name, flag, color in series:
        xs, ys, eh, el, hover = [], [], [], [], []
        for y in years:
            sub = [
                p
                for p in papers
                if p["year"] == y
                and (flag is None or bool(p.get("early_accepted")) is flag)
            ]
            a = S.type_values(sub, "oral", "avg_score_raw")
            b = S.type_values(sub, "poster", "avg_score_raw")
            if len(a) < 3 or len(b) < 3:
                continue
            d, lo, hi, mag = S.cliffs_delta_ci(a, b)
            pos += 1
            xs.append(d)
            ys.append(pos)
            eh.append(0 if np.isnan(hi) else hi - d)
            el.append(0 if np.isnan(lo) else d - lo)
            ticks.append(str(y))
            tickvals.append(pos)
            hover.append(
                f"{name} - {y}<br>Cliff's δ = {d:+.3f}"
                f"<br>95% CI [{lo:+.3f}, {hi:+.3f}]<br>{mag} effect"
                f"<br>n = {len(a)} oral vs {len(b)} poster"
            )
        if not xs:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=name,
                marker=dict(size=9, color=color),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=eh,
                    arrayminus=el,
                    color=color,
                    thickness=1.6,
                    width=4,
                ),
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
            )
        )
        pos += 1

    fig.update_layout(
        **base_layout(
            margin=dict(l=60, r=30, t=52, b=48),
            legend=dict(orientation="h", y=1.13, x=0, font=dict(size=14.5)),
            xaxis=dict(
                title="Cliff's δ; oral vs poster-only review score",
                gridcolor=RULE,
                zeroline=False,
                range=[-0.25, 1.0],
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticks,
                gridcolor=RULE,
                autorange="reversed",
            ),
        )
    )
    save_chart(fig, "orals_strata.html")


# ---------------------------------------------------------------------------
# 5. Reviewer confidence
# ---------------------------------------------------------------------------


def chart_confidence(papers: list, years: list) -> None:
    for p in papers:
        p["_mean_conf"] = S.mean_confidence(p)

    fig = go.Figure()
    for ptype in TYPES:
        xs, ys, eh, el, hover = [], [], [], [], []
        for y in years:
            sub = [p for p in papers if p["year"] == y]
            vals = S.type_values(sub, ptype, "_mean_conf")
            if len(vals) < 3:
                continue
            pt, lo, hi = S.bootstrap_ci(vals)
            xs.append(str(y))
            ys.append(pt)
            eh.append(0 if np.isnan(hi) else hi - pt)
            el.append(0 if np.isnan(lo) else pt - lo)
            hover.append(
                f"{LBL[ptype]} {y}<br>mean confidence {pt:.2f} / 4"
                f"<br>95% CI [{lo:.2f}, {hi:.2f}]<br>n = {len(vals)}"
            )
        if not xs:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+lines",
                name=LBL[ptype],
                marker=dict(size=9, color=COL[ptype]),
                line=dict(color=COL[ptype], width=1.5),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=eh,
                    arrayminus=el,
                    color=COL[ptype],
                    thickness=1.4,
                    width=4,
                ),
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
            )
        )
    fig.update_layout(
        **base_layout(
            margin=dict(l=70, r=25, t=48, b=45),
            legend=dict(orientation="h", y=1.15, x=0, font=dict(size=14.5)),
            yaxis=dict(
                title="Mean reviewer confidence (1-4)",
                gridcolor=RULE,
                range=[2.9, 3.9],
            ),
            xaxis=dict(gridcolor=RULE),
        )
    )
    fig.add_annotation(
        x=0,
        y=1.03,
        xref="paper",
        yref="paper",
        xanchor="left",
        showarrow=False,
        text=(
            "1 Not confident · 2 Somewhat · 3 Confident but not certain · "
            "4 Very confident"
        ),
        font=dict(size=13.5, color=MUTED),
    )
    save_chart(fig, "orals_confidence.html")


# ---------------------------------------------------------------------------
# 6. Subject areas
# ---------------------------------------------------------------------------

_AREAS_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>Selection rate by subject area</title>
<script src="https://cdn.plot.ly/plotly-__PLOTLYJS__.min.js"></script>
<style>
html,body{margin:0;font-family:__FONT__;background:__BG__;color:__TEXT__}
.bar-ctrls{display:flex;align-items:center;gap:10px;padding:8px 14px 2px;
           font-size:16px;color:#475569}
.bar-ctrls select{font:inherit;padding:3px 8px;border:1px solid #cbd5e1;
                  border-radius:6px;background:#fff;color:__TEXT__}
.bar-ctrls .note{color:__MUTED__}
#ar{width:100%;height:calc(100vh - 42px)}
</style></head><body>
<div class="bar-ctrls">
  <label>Year <select id="yr"></select></label>
  <span class="note" id="note"></span>
</div>
<div id="ar"></div>
<script>
const KEYS=__KEYS__;
const DATA=__DATA__;
const POS="__POS__", NEG="__NEG__", MUTED="__MUTED__", RULE="__RULE__";
// Area names run long ("Surgical Visualization and Mixed/Augmented/Virtual
// Reality"); wrap to at most two lines so the left margin can stay fixed.
// The untruncated name is always in the hover.
function wrap(s,n){ if(s.length<=n)return s; const w=s.split(" ");let line="",out=[];
 for(const x of w){ if((line+" "+x).trim().length>n&&line){out.push(line);line=x;}
  else line=(line+" "+x).trim();
  if(out.length===1&&line.length>n){line=line.slice(0,n-1).trim()+"\u2026";break;} }
 out.push(line); return out.slice(0,2).join("<br>"); }
function draw(){
 const d=DATA[yr.value];
 document.getElementById("note").textContent=
   d.n_sel+" of "+d.n_papers+" accepted papers got a talk ("
   +d.baseline.toFixed(1)+"% overall)";
 const vals=d.rows.map(r=>r.d);
 const trace={type:"bar",orientation:"h",
   x:vals, y:d.rows.map(r=>wrap(r.a,30)),
   marker:{color:vals.map(v=>v>=0?POS:NEG)},
   text:vals.map(v=>(v>0?"+":"")+v.toFixed(1)),textposition:"outside",
   textfont:{size:13.5},cliponaxis:false,
   customdata:d.rows.map(r=>[r.f,r.k,r.n,r.r,r.lo,r.hi,
     r.y?"<br><i>label used in "+r.y+" only</i>":""]),
   hovertemplate:"%{customdata[0]}<br>%{customdata[1]} of %{customdata[2]}"
     +" accepted papers in this area got a talk"
     +"<br>%{customdata[3]:.1f}% [95% CI %{customdata[4]:.1f},"
     +" %{customdata[5]:.1f}]<br>Baseline for this window: "
     +d.baseline.toFixed(1)+"%%{customdata[6]}<extra></extra>"};
 const layout={paper_bgcolor:"__BG__",plot_bgcolor:"__PLOTBG__",
   font:{family:"__FONT__",size:15.5,color:"__TEXT__"},
   margin:{l:232,r:56,t:10,b:46},
   xaxis:{title:{text:"Percentage points above / below the "
     +d.baseline.toFixed(1)+"% overall selection rate",
     font:{size:15.5,color:MUTED}},gridcolor:RULE,zeroline:false},
   yaxis:{tickfont:{size:13},automargin:false},
   shapes:[{type:"line",x0:0,x1:0,yref:"paper",y0:0,y1:1,
            line:{color:MUTED,width:1}}],
   hoverlabel:{font:{family:"__FONT__",size:15.5,color:"__TEXT__"},
               bgcolor:"#fff",bordercolor:RULE}};
 Plotly.react("ar",[trace],layout,{displayModeBar:false,responsive:true});
}
KEYS.forEach(k=>yr.add(new Option(k.label,k.key)));
yr.value="all";
yr.onchange=draw;
draw();
</script>
</body></html>
"""


def _area_rows(papers: list, min_papers: int, top_n: int) -> dict:
    """One dropdown option's worth of area data (rows already trimmed).

    Each row also records the years the area label actually appears in. MICCAI
    renames its subject-area taxonomy between years -- no label is used in all
    five, and 102 of the 161 distinct labels appear in exactly one year (2025
    moved to a "Surgery -> ..." prefixed scheme) -- so an "all years" row can
    cover a subset of years while being drawn against the five-year baseline.
    Naming the span in the hover keeps that visible instead of implied.
    """
    sel_by_area: dict = defaultdict(int)
    tot_by_area: dict = defaultdict(int)
    yrs_by_area: dict = defaultdict(set)
    for p in papers:
        chosen = p["presentation_type"] in ("oral", "spotlight")
        for area in p.get("subject_areas", []) or []:
            tot_by_area[area] += 1
            yrs_by_area[area].add(p["year"])
            if chosen:
                sel_by_area[area] += 1

    n_sel = sum(
        1 for p in papers if p["presentation_type"] in ("oral", "spotlight")
    )
    baseline = 100 * n_sel / len(papers) if papers else 0.0
    all_years = sorted({p["year"] for p in papers})

    rows = []
    for area, tot in tot_by_area.items():
        if tot < min_papers:
            continue
        k = sel_by_area.get(area, 0)
        p_, lo, hi = S.wilson_ci(k, tot)
        ys = sorted(yrs_by_area[area])
        span = (
            ""
            if len(ys) == len(all_years)
            else (
                str(ys[0])
                if len(ys) == 1
                else (
                    f"{ys[0]}-{ys[-1]}"
                    if ys == list(range(ys[0], ys[-1] + 1))
                    else ", ".join(str(v) for v in ys)
                )
            )
        )
        rows.append(
            {
                "f": area,
                "y": span,
                "a": area.split("->")[-1].strip() if "->" in area else area,
                "d": round(100 * p_ - baseline, 1),
                "r": round(100 * p_, 1),
                "lo": round(100 * lo, 1),
                "hi": round(100 * hi, 1),
                "k": k,
                "n": tot,
            }
        )
    rows.sort(key=lambda r: r["d"])
    if len(rows) > top_n:
        half = top_n // 2
        rows = rows[:half] + rows[-half:]
    return {
        "baseline": round(baseline, 1),
        "n_sel": n_sel,
        "n_papers": len(papers),
        "rows": rows,
    }


def chart_areas(papers: list, min_papers: int = 25, top_n: int = 18) -> None:
    """Which subject areas are over- or under-represented in the program.

    Hand-written HTML rather than save_chart(), for the same reason as
    subject_movers.html on the overview page: a Year dropdown is a few lines of
    plain JavaScript, every option is precomputed here in Python, and switching
    redraws with Plotly.react rather than recomputing anything.

    A paper can carry several subject areas, so the areas are not a partition
    and the rates do not sum to the overall rate. Each bar is that area's own
    selection rate against the all-papers baseline *for the selected window* --
    the baseline is recomputed per year, because the share of accepted papers
    given a talk is not constant across years.
    """
    considered = [p for p in papers if p.get("presentation_type")]
    years = _years_with_orals(papers)

    data = {"all": _area_rows(considered, min_papers, top_n)}
    keys = [{"key": "all", "label": "All years"}]
    for y in sorted(years, reverse=True):
        sub = [p for p in considered if p["year"] == y]
        data[str(y)] = _area_rows(sub, min_papers, top_n)
        keys.append({"key": str(y), "label": str(y)})

    save_template(
        _AREAS_TEMPLATE,
        "orals_areas.html",
        {
            **theme_tokens(),
            "__POS__": COL["oral"],
            "__NEG__": COL["poster"],
            "__KEYS__": json.dumps(keys),
            "__DATA__": json.dumps(data),
        },
    )


# ---------------------------------------------------------------------------
# 7. Semantic map
# ---------------------------------------------------------------------------


def chart_map(papers: list) -> None:
    """The UMAP map with the selected papers lit up over a faint poster field."""
    fig = go.Figure()
    posters = [
        p
        for p in papers
        if p.get("presentation_type") == "poster"
        and p.get("umap_x") is not None
    ]
    fig.add_trace(
        go.Scattergl(
            x=[p["umap_x"] for p in posters],
            y=[p["umap_y"] for p in posters],
            mode="markers",
            name=f"Poster only ({len(posters)})",
            marker=dict(size=3.5, color="rgba(148,163,184,0.30)"),
            customdata=[[p.get("url"), p["title"]] for p in posters],
            hovertemplate="%{customdata[1]}<extra></extra>",
        )
    )
    for ptype in ("spotlight", "oral"):
        sel = [
            p
            for p in papers
            if p.get("presentation_type") == ptype
            and p.get("umap_x") is not None
        ]
        if not sel:
            continue
        fig.add_trace(
            go.Scattergl(
                x=[p["umap_x"] for p in sel],
                y=[p["umap_y"] for p in sel],
                mode="markers",
                name=f"{LBL[ptype]} ({len(sel)})",
                marker=dict(
                    size=7,
                    color=COL[ptype],
                    line=dict(width=0.5, color="#fff"),
                ),
                customdata=[[p.get("url"), p["title"]] for p in sel],
                hovertemplate="%{customdata[1]}<extra></extra>",
            )
        )
    fig.update_layout(
        **base_layout(
            margin=dict(l=20, r=20, t=44, b=20),
            dragmode="pan",
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=14.5)),
            xaxis=dict(
                visible=False, showgrid=False, zeroline=False
            ),
            yaxis=dict(
                visible=False,
                showgrid=False,
                zeroline=False,
                scaleanchor="x",
                scaleratio=1,
            ),
        )
    )
    # Click-to-open, same as the other semantic maps: the points already
    # carry [url, title] in customdata, which is what CLICK_JS reads.
    save_chart(fig, "orals_map.html", interactive=True, extra_js=CLICK_JS)


# ---------------------------------------------------------------------------
# 8. Profile: code, team size, title length
# ---------------------------------------------------------------------------


def chart_code(papers: list, years: list) -> None:
    """Share of papers releasing code, by type, per year.

    This replaced a pooled three-panel "profile" chart that also showed authors
    per paper and title length. Those two were flat: pooled over all years the
    three types sat within a fraction of an author (5.8 / 6.0 / 6.2) and within
    a fifth of a word (10.5 / 10.6 / 10.6) of each other, with fully overlapping
    intervals in both. Plotting a number that says nothing invites the reader to
    hunt for a pattern in noise, so those two are stated in one line of prose on
    the page instead.

    Code release is the one of the three worth a chart, and only per year:
    pooled it looks flat, but the direction is not stable. Posters led in 2022,
    the types were level in 2023, and orals pulled ahead in 2024 and 2025. The
    pooled figure averages a changing sign into a fake null.
    """
    _type_rate_chart(
        papers,
        years,
        lambda p: p.get("has_code"),
        y_title="Papers releasing code (%)",
        hover_verb="released code",
        filename="orals_code.html",
    )


# ---------------------------------------------------------------------------


def build_all(papers: list, year_cfg: dict) -> bool:
    """Build every orals_*.html. Returns False when there is no oral data."""
    years = _years_with_orals(papers)
    if not years:
        logger.info("  No oral/spotlight data; skipping orals charts")
        for stale in (
            "orals_overview.html",
            "orals_scores.html",
            "orals_effect.html",
            "orals_early.html",
            "orals_strata.html",
            "orals_confidence.html",
            "orals_areas.html",
            "orals_map.html",
            "orals_code.html",
        ):
            (OUT_DIR / stale).unlink(missing_ok=True)
        return False

    logger.info(f"--- Orals & spotlights charts ({years}) ---")
    chart_overview(papers, years)
    chart_scores(papers, years, year_cfg)
    chart_effect(papers, years)
    chart_early(papers, years)
    chart_strata(papers, years)
    chart_confidence(papers, years)
    chart_areas(papers)
    chart_map(papers)
    chart_code(papers, years)
    return True
