"""Responsive layout checks: does the built site fit the screen it is on?

Run standalone (prints a report, exits nonzero on failure):

    python tests/check_responsive.py
    python tests/check_responsive.py --widths 360 --pages index.html

Or under pytest, where it skips itself if Chrome is missing:

    python -m pytest tests/check_responsive.py -q

The `check_` prefix is deliberate and load-bearing: pytest collects `test_*.py`
by default, so this file is NOT picked up by a bare `python -m pytest tests/`,
and IS picked up when named explicitly as above. That is the intended split.
The rest of the suite is 96 data tests that run in four seconds and need
nothing but the repository; this one drives a browser, needs `website/` to have
been built, and takes minutes. Renaming it to `test_responsive.py` would add
that to every run. If you want it in CI, name it in the CI command rather than
renaming the file.

WHY THIS EXISTS
---------------
The site rendered at roughly 45% width on a phone for months, and nothing
could see it. Every existing test starts from the data: golden counts, parser
fixtures, statistics. A layout bug writes no wrong number, so the whole suite
passed while `map_all.html` was drawing a plot area two pixels wide.

Every check below is a failure that was actually found, not a rule invented in
advance. Checks 1 to 5 come from 2026-09-03; 6 to 8 were added on
2026-09-04, after all five of the others passed on three charts that were
visibly broken on a phone; 9 on 2026-09-08, after all eight passed on a map
that redrew itself at a different scale when the reader touched its own
controls.

A check that never fires is indistinguishable from a check that passes, and
this file has now shipped two of those: VOID found nothing at all until its
selector was corrected, and the whole suite reported clean on the regressions
below. **Verify a new check by re-breaking the bug it is for and watching it
fail.** Every one of 6, 7 and 8 was confirmed that way.

1. OVERFLOW    no element wider than the viewport. One overflowing element
               (`.nav-links`, 668px against a 360px viewport) made the whole
               document 805px wide, and Chrome for Android scales an
               overflowing page down to fit. That is the half-width page.

2. CROPPED     no chart iframe whose content is taller than its frame.
               `trends_overview.html` reflows to two columns under 820px and
               needs 1187px; its frame was fixed at 660px, so three of the nine
               panels were simply cut off.

3. SQUEEZED    no plot whose plotting region is a small fraction of the space
               it was given. Plotly margins are in pixels, so `margin=dict(l=20,
               r=300)` on a 280px-wide chart leaves nothing. This is the check
               that catches the 2px axis.

4. CLIPPED     no tick label or axis title that runs outside its own chart
               box. `subject_movers` passed checks 1-3 at 360px while drawing
               "...elf-supervised Learning" with its left half cut off: the
               figure was the right size and only the text was wrong. This is
               the `automargin: false` failure, the mirror image of check 3's
               `automargin: true` one.

5. COLLIDING   no two pieces of text overlapping each other, anywhere in the
               figure: ticks, axis titles and annotations alike. The Trending
               Topics panels rendered their year ticks on top of one another;
               the orals charts overprinted thirteen per-bar value labels into
               a smear; stacking subplots put one panel's axis title through
               the next panel's heading.

6. HOLLOW      no chart frame much taller than anything drawn inside it. The
               exact converse of check 2, and leaving it out let the worst
               regression of this work through: a chart whose height claim was
               applied and then reverted drew twelve stacked panels squashed
               into the top 660px of a 2136px frame, with a screen and a half of
               blank below. Everything was legible, nothing overflowed and
               nothing was cut off, so checks 1 to 5 all passed.

7. VOID        no empty band BETWEEN two things a chart drew. A figure can be
               exactly as tall as its contents with the hole in the middle:
               `scores_2025` put 102px between its key and its first panel,
               because a legend pinned in paper units grows with the figure.
               Panel rectangles are computed from the layout, never found in the
               DOM: Plotly's paper background is a rect spanning the whole
               figure, so including it covers every gap there could be, and
               excluding it makes a flat data line the only ink in its panel.

8. UNTITLED    a figure that labels some of its panels but not all of them. A
               hidden annotation neither overflows, collides nor crops, so
               nothing else here could see `subject_lines` lose nine of its
               twelve panel names to a rule meant for overprinted value labels.

9. TOGGLED     a chart's own controls must not resize its plotting area. This
               is the only check that touches the page: everything above
               measures one static load, so a figure that is right when it
               arrives and wrong after a tap passes all eight. `map_all` drew
               its 3,717 points 923px tall coloured by year and 620px tall
               coloured by cluster, in the same frame, because the legend
               underneath took what it needed out of the plot.

WHY NOT PLAYWRIGHT
------------------
It would be less code. But this project keeps a deliberately small dependency
footprint (pyproject.toml calls Pillow out as the only optional one), and
playwright additionally downloads its own browser. Chrome is already installed
on any machine someone would develop this on, and `--dump-dom` with
`--virtual-time-budget` is enough: the measuring happens in JavaScript inside
the page, and Python only starts a server, runs the browser, and reads JSON
back out. Nothing is added to pyproject.toml.

The measurement runs each page inside a fixed-width iframe rather than resizing
the browser window, because headless Chrome does not reliably honour
--window-size (it reported 500px for a requested 360px). An iframe's
innerWidth is exact.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import re
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEBSITE = ROOT / "website"

# The four widths worth checking, and why each one is here:
#   360   Galaxy S23+ and most Android phones in portrait. The reported bug.
#   390   iPhone 14/15 in portrait, the narrowest common iOS width.
#   768   tablet portrait, and a phone in landscape. The compact nav tier.
#   1000  the width at which the full desktop nav stops fitting (it needs
#         1002px), so this is the boundary the tiers were built around.
DEFAULT_WIDTHS = (360, 390, 768, 1000)

# Every page shares base.html, so the nav is covered by any of them; the rest
# are here for their charts. All five year pages are checked, not a sample:
# they carry the same eleven charts against different data, and the data is
# what decides how long a label is, how many categories an axis has and whether
# a panel gets a heading at all. The sample used to be 2025 and 2021, on the
# grounds that they are the two ends of the range and share a template; the
# reader who found the regressions this check was extended for reported them as
# happening on every year page, which is not something a two-year sample can
# either confirm or rule out.
DEFAULT_PAGES = (
    "index.html",
    "orals.html",
    "papers.html",
    "about.html",
    "year/2026.html",
    "year/2025.html",
    "year/2024.html",
    "year/2023.html",
    "year/2022.html",
    "year/2021.html",
)

# A plot whose axis is under 45% of the width it was given has had its space
# eaten by fixed pixel margins. Measured for reference: a healthy desktop chart
# with the widest label margin in the codebase (l=250 on a 900px div) sits at
# 0.74; the broken map at 360px sat at 0.007.
MIN_AXIS_RATIO = 0.45
# Panels smaller than this are grid cells (the 3x3 trends grid drops to ~150px
# per panel on a phone, which is correct, not broken), so the ratio test does
# not apply to them.
MIN_DIV_WIDTH_TO_CHECK = 200
# An iframe may exceed its declared height by a couple of pixels through
# rounding without anything actually being cut off.
CROP_TOLERANCE_PX = 4
# See the note beside the collision test for why this is 6 and not 0.
COLLIDE_TOLERANCE_PX = 6
# How much taller than its own contents a chart frame may be before the gap
# reads as a band of white space. A chart card can legitimately carry a little
# padding under the plot; a screen and a half of nothing is the bug this
# catches.
HOLLOW_TOLERANCE_PX = 60
# An empty horizontal band inside a chart, between two things it drew. Wider
# than this and it reads as a hole rather than as breathing room.
#
# Set from measurement, on scores_2025 at 360px: the legend-to-first-panel gap
# it was written for is 102px, and the same gap with that bug fixed is about
# 25px. The legitimate case it has to clear is a stacked subplot's row gap,
# which is around 90px but is mostly filled by the upper panel's axis title and
# the lower panel's heading, leaving well under 70px actually empty.
VOID_TOLERANCE_PX = 70
# How far a plotting area may move when the reader works a chart's own
# controls. Not zero: a legend one row taller in one mode is a pixel or two of
# rounding through the margins. The failure this exists for was 303px.
TOGGLE_TOLERANCE_PX = 8


HARNESS = """<!doctype html>
<html><head><meta charset="utf-8"><title>responsive check</title>
<style>html,body{margin:0}#f{border:0;display:block;height:12000px}</style>
</head><body>
<iframe id="f"></iframe>
<pre id="out">PENDING</pre>
<script>
const PAGES = __PAGES__;
const WIDTHS = __WIDTHS__;
const MIN_AXIS_RATIO = __MIN_AXIS_RATIO__;
const MIN_DIV_WIDTH = __MIN_DIV_WIDTH__;
const HOLLOW_TOL = __HOLLOW_TOL__;
const VOID_TOL = __VOID_TOL__;
const TOGGLE_TOL = __TOGGLE_TOL__;
const CROP_TOL = __CROP_TOL__;
const COLLIDE_TOL = __COLLIDE_TOL__;

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function label(el) {
  const cls = (el.getAttribute && el.getAttribute('class')) || '';
  return el.tagName.toLowerCase() + (cls ? '.' + cls.trim().split(/\\s+/).join('.') : '');
}

async function measure(page, width) {
  const f = document.getElementById('f');
  f.style.width = width + 'px';
  await new Promise(res => { f.onload = res; f.src = page; });
  // The iframe is 12000px tall so that loading="lazy" chart frames are all
  // inside the viewport and actually load. Plotly then needs a moment.
  await sleep(4000);   // charts settle in stages; see RESPONSIVE_JS passes

  const W = f.contentWindow, D = f.contentDocument;
  const vw = W.innerWidth;
  const res = {page: page, width: width, vw: vw,
               doc: D.documentElement.scrollWidth,
               overflow: [], cropped: [], squeezed: [], clipped: [],
               colliding: [], hollow: [], untitled: [], voids: [],
               toggled: [], notes: []};

  // 1. OVERFLOW
  // An element wider than the viewport is only a bug if the reader cannot get
  // at it. Inside a container that scrolls horizontally on purpose (the Top
  // Papers tables, the All Papers browse table, the section nav) a wide child
  // is the intended design, so walk up and exempt those.
  function inScroller(el) {
    for (let p = el.parentElement; p && p !== D.body; p = p.parentElement) {
      const ox = W.getComputedStyle(p).overflowX;
      if (ox === 'auto' || ox === 'scroll') return true;
    }
    return false;
  }
  const seen = new Set();
  for (const el of D.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.width > vw + 1 && !inScroller(el)) {
      const k = label(el);
      if (!seen.has(k)) { seen.add(k); res.overflow.push({el: k, w: Math.round(r.width)}); }
    }
  }

  // Text findings are collected TWICE, ~500ms apart, and only those present
  // both times are reported. These charts re-render: the hand-written ones call
  // Plotly.react when their dropdown initialises, and the responsive script
  // relayouts on top of that. Mid-render, Plotly can briefly have two sets of
  // tick labels in the DOM, which a single measurement reads as two category
  // names printed on top of each other. That produced a confident 138px
  // "collision" on a chart whose ticks are 29px apart when settled.
  function textScan() {
    const found = {clipped: [], colliding: []};
    for (const fr of D.querySelectorAll('iframe.chart-frame')) {
      const nm = (fr.getAttribute('src') || '?').split('/').pop();
      let dd = null;
      try { dd = fr.contentDocument; } catch (e) { dd = null; }
      if (!dd) continue;
      for (const p of dd.querySelectorAll('.js-plotly-plot')) {
        const box = p.getBoundingClientRect();
        if (box.width < 40) continue;
        const texts = p.querySelectorAll(
          '.xtick text, .ytick text, .g-xtitle text, .g-ytitle text, .gtitle,' +
          ' .annotation text, [class*="bartext"], .legendtext');
        const rr = [];
        for (const t of texts) {
          const r = t.getBoundingClientRect();
          if (!r.width) continue;
          const str = (t.textContent || '').trim();
          const cls = (t.parentNode && t.parentNode.getAttribute &&
                       t.parentNode.getAttribute('class')) || t.getAttribute('class') || '?';
          rr.push({r: r, s: str, c: cls});
          // All four edges. Testing only left and right was a real gap: the
          // confidence chart's scale key, once moved below the plot, ran off
          // the bottom of the frame and the check called it clean.
          const over = {left: box.left - r.left, right: r.right - box.right,
                        top: box.top - r.top, bottom: r.bottom - box.bottom};
          let worst = null;
          for (const k in over) {
            // 4px for the same reason the collision test uses 6: a text box
            // carries side bearings and line leading beyond the glyphs, so a
            // two or three pixel overhang is not visible.
            if (over[k] > 4 && (!worst || over[k] > over[worst])) worst = k;
          }
          if (worst) {
            found.clipped.push({chart: nm, text: str.slice(0, 34), side: worst,
                                by: Math.round(over[worst])});
          }
        }
        let reported = false;
        for (let i = 0; i < rr.length && !reported; i++) {
          for (let j = i + 1; j < rr.length && !reported; j++) {
            const a = rr[i].r, b = rr[j].r;
            const dx = Math.min(a.right, b.right) - Math.max(a.left, b.left);
            const dy = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
            // Vertical overlap has to be a real share of the text's height,
            // not a couple of pixels. A text box includes its line leading, so
            // two-line wrapped tick labels 29px apart with 32px boxes touch by
            // 3px without the glyphs ever meeting. Measured: that false case is
            // 9% of the box height, while the confidence key genuinely printed
            // over a legend entry at 42%.
            const minH = Math.min(a.bottom - a.top, b.bottom - b.top);
            if (dx > COLLIDE_TOL && dy > Math.max(2, minH * 0.25)) {
              found.colliding.push({chart: nm,
                                    a: rr[i].s.slice(0, 18) + ' <' + rr[i].c + '>',
                                    b: rr[j].s.slice(0, 18) + ' <' + rr[j].c + '>',
                                    overlap: Math.round(dx),
                                    ra: [Math.round(a.left),Math.round(a.top),Math.round(a.right),Math.round(a.bottom)],
                                    rb: [Math.round(b.left),Math.round(b.top),Math.round(b.right),Math.round(b.bottom)],
                                    ntext: rr.length});
              reported = true;
            }
          }
        }
      }
    }
    return found;
  }
  const scanA = textScan();
  await sleep(500);
  const scanB = textScan();
  const keyC = (x) => x.chart + '|' + x.text + '|' + x.side;
  const keyO = (x) => x.chart + '|' + x.a + '|' + x.b;
  const seenB_C = new Set(scanB.clipped.map(keyC));
  const seenB_O = new Set(scanB.colliding.map(keyO));
  res.clipped = scanA.clipped.filter(x => seenB_C.has(keyC(x)));
  res.colliding = scanA.colliding.filter(x => seenB_O.has(keyO(x)));

  for (const fr of D.querySelectorAll('iframe.chart-frame')) {
    const name = (fr.getAttribute('src') || '?').split('/').pop();
    let d2 = null;
    try { d2 = fr.contentDocument; } catch (e) { d2 = null; }
    if (!d2 || !d2.documentElement) { res.notes.push(name + ': unreadable'); continue; }

    // 2. CROPPED
    const declared = Math.round(fr.getBoundingClientRect().height);
    const content = d2.documentElement.scrollHeight;
    if (content > declared + CROP_TOL) {
      res.cropped.push({chart: name, frame: declared, content: content, cut: content - declared});
    }

    // 3. SQUEEZED
    const plots = d2.querySelectorAll('.js-plotly-plot');
    if (!plots.length && !d2.querySelector('svg')) {
      res.notes.push(name + ': nothing rendered (plotly blocked?)');
      continue;
    }
    for (const p of plots) {
      const pw = p.getBoundingClientRect().width;
      if (pw < MIN_DIV_WIDTH) continue;
      const fl = p._fullLayout;
      // _size.w is the whole plotting region, margins excluded. Use it rather
      // than xaxis._length: on a subplot figure (subject_lines has 12 panels,
      // naming has 2) xaxis._length is ONE panel's width, so comparing it to
      // the full div width fails every multi-panel chart for no reason.
      if (!fl || !fl._size || typeof fl._size.w !== 'number') continue;
      const ratio = fl._size.w / pw;
      if (ratio < MIN_AXIS_RATIO) {
        res.squeezed.push({chart: name, divW: Math.round(pw),
                           axis: Math.round(fl._size.w),
                           ratio: Number(ratio.toFixed(3)),
                           marginL: fl.margin && fl.margin.l,
                           marginR: fl.margin && fl.margin.r});
      }

      // 6. UNTITLED
      // Every panel of a figure that labels ANY of its panels must have a
      // visible label. The panels are read from the layout as it stands, and
      // an annotation counts as a heading if it sits just above a panel's top
      // edge inside that panel's columns, so this says nothing about how the
      // headings got there.
      //
      // Nothing here could see subject_lines lose nine of its twelve panel
      // names: they were hidden, not moved, and a hidden annotation neither
      // overflows, collides, nor crops. The chart passed every check while
      // three quarters of it had gone blank.
      const panels = [];
      for (const k in fl) {
        if (!/^xaxis/.test(k) || !fl[k] || !fl[k].domain) continue;
        const yk = 'yaxis' + k.slice(5);
        if (fl[yk] && fl[yk].domain) panels.push({xd: fl[k].domain, yd: fl[yk].domain});
      }
      if (panels.length > 1) {
        const heads = panels.map(() => false);
        for (const a of (fl.annotations || [])) {
          if (a.visible === false || !String(a.text || '').trim()) continue;
          if (a.xref !== 'paper' || a.yref !== 'paper') continue;
          for (let pi = 0; pi < panels.length; pi++) {
            const q = panels[pi];
            if (a.x >= q.xd[0] - 0.02 && a.x <= q.xd[1] + 0.02 &&
                a.y >= q.yd[1] - 0.02 && a.y <= q.yd[1] + 0.10) heads[pi] = true;
          }
        }
        const have = heads.filter(Boolean).length;
        if (have > 0 && have < panels.length) {
          res.untitled.push({chart: name, have: have, panels: panels.length});
        }
      }
    }

    // 5. HOLLOW
    // The frame is materially taller than anything drawn in it, which is a
    // band of white space in the middle of the page.
    //
    // This is the exact converse of the CROPPED check above, and leaving it out
    // is what let the worst regression of this work through: a chart whose
    // height claim was applied and then reverted drew twelve stacked panels
    // squashed into the top 660px of a 2136px frame, with a screen and a half
    // of blank below. Everything in it was legible, nothing overflowed, and
    // nothing was cut off, so every check passed.
    let ink = 0;
    for (const el of d2.querySelectorAll('.js-plotly-plot, svg, canvas')) {
      const rr = el.getBoundingClientRect();
      if (rr.height > 0) ink = Math.max(ink, rr.bottom);
    }
    if (ink > 0 && declared - ink > HOLLOW_TOL) {
      res.hollow.push({chart: name, frame: declared, ink: Math.round(ink),
                       blank: Math.round(declared - ink)});
    }

    // 7. VOID
    // A band of empty space BETWEEN two things the chart drew.
    //
    // HOLLOW above only sees space below everything, and a chart can be the
    // full height of its frame with the emptiness in the middle of it: scores_*
    // put 102px between its key and its first panel, because the legend is
    // parked in paper units that grow with the figure. Nothing overlapped,
    // nothing was cut off, nothing was hidden, and the frame was exactly as
    // tall as its contents, so every other check passed.
    //
    // The plot background rectangle counts as ink, so an empty-looking scatter
    // panel is covered and only genuinely unused bands are reported.
    // A panel counts as occupied over its whole plotting rectangle, and those
    // rectangles are computed from the layout rather than found in the DOM.
    //
    // Reading them off the drawn elements does not work, in both directions.
    // Plotly's paper background is a rect.bg spanning the entire figure, so
    // including it covers every gap there could ever be and the check finds
    // nothing; excluding it leaves a flat data line as the only ink in its
    // panel, and subject_lines' 43px-tall trace inside a 123px panel then reads
    // as a 136px hole. The layout says exactly where each panel is.
    // Every figure on the page, not just the first: trends_overview is nine
    // separate Plotly divs in a CSS grid, and reading panels from one of them
    // while reading text from all nine reported a 182px hole at every width,
    // including on the desktop layout that has shipped for months.
    const spans = [];
    let panelCount = 0, panelTop = Infinity, panelBot = -Infinity;
    for (const gd0 of d2.querySelectorAll('.js-plotly-plot')) {
      const fl0 = gd0._fullLayout, sz = fl0 && fl0._size;
      if (!sz) continue;
      const base = gd0.getBoundingClientRect().top;
      for (const k in fl0) {
        if (!/^xaxis/.test(k) || !fl0[k] || !fl0[k].domain) continue;
        const yk = 'yaxis' + k.slice(5);
        if (!fl0[yk] || !fl0[yk].domain) continue;
        const yd = fl0[yk].domain;
        const a = base + sz.t + (1 - yd[1]) * sz.h;
        const b = base + sz.t + (1 - yd[0]) * sz.h;
        spans.push([a, b]);
        panelTop = Math.min(panelTop, a);
        panelBot = Math.max(panelBot, b);
        panelCount++;
      }
    }
    if (panelCount) {
      for (const el of d2.querySelectorAll(
          '.legend, .annotation, .xtick text, .ytick text, .g-xtitle,' +
          ' .g-ytitle, .gtitle, .updatemenu-header-group')) {
        const rr = el.getBoundingClientRect();
        if (rr.height > 0 && rr.width > 0 && rr.height < declared * 0.95) {
          spans.push([rr.top, rr.bottom]);
        }
      }
      spans.sort((a, b) => a[0] - b[0]);
      let end = spans[0][1], worst = 0, at = 0;
      for (const sp of spans) {
        // A gap lying entirely between the topmost and bottommost panel is the
        // figure's own row spacing, which the build chose and which the desktop
        // page has always had. Only space outside the band of panels is this
        // check's business: a key stranded above the first panel, or a legend
        // stranded below the last one.
        const inside = end >= panelTop && sp[0] <= panelBot;
        if (!inside && sp[0] - end > worst) { worst = sp[0] - end; at = end; }
        if (sp[1] > end) end = sp[1];
      }
      if (worst > VOID_TOL) {
        res.voids.push({chart: name, gap: Math.round(worst), at: Math.round(at)});
      }
    }
  }

  // 9. TOGGLED
  // A chart's own controls must not change the size of its plotting area.
  //
  // Everything above measures one static load, so a figure that is correct
  // when it arrives and wrong after a tap reads as perfect. The semantic map
  // holds both colour modes as 25 traces and shows 5 or 20 of them; on a phone
  // the legend sits under the plot, and Plotly gives the bottom margin exactly
  // what the legend it drew needs. Five short year names took 93px and twenty
  // wrapped cluster names took 423px out of the same frame, so the difference
  // went into the plot and the map was drawn 923px tall in one mode and 620px
  // in the other. Nothing overlapped, overflowed, or was cut off in either.
  //
  // Runs last because it leaves the page on whichever mode it clicked last.
  for (const fr of D.querySelectorAll('iframe.chart-frame')) {
    const d2 = fr.contentDocument;
    if (!d2) continue;
    const gd = d2.querySelector('.js-plotly-plot');
    if (!gd || !gd._fullLayout) continue;
    const btns = d2.querySelectorAll('.updatemenu-button');
    if (btns.length < 2) continue;
    const name = (fr.getAttribute('src') || '').split('/').pop();
    const seen = [];
    for (const b of btns) {
      b.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true,
                                               view: fr.contentWindow}));
      await sleep(2500);       // the responsive script settles in passes
      const sz = gd._fullLayout._size;
      if (!sz) continue;
      seen.push({mode: (b.textContent || '').trim(),
                 w: Math.round(sz.w), h: Math.round(sz.h)});
    }
    for (let i = 1; i < seen.length; i++) {
      const dw = Math.abs(seen[0].w - seen[i].w);
      const dh = Math.abs(seen[0].h - seen[i].h);
      if (dw > TOGGLE_TOL || dh > TOGGLE_TOL) {
        res.toggled.push({chart: name, a: seen[0], b: seen[i],
                          dw: dw, dh: dh});
      }
    }
  }
  return res;
}

(async () => {
  const all = [];
  for (const page of PAGES) {
    for (const width of WIDTHS) {
      try { all.push(await measure(page, width)); }
      catch (e) { all.push({page: page, width: width, error: String(e)}); }
    }
  }
  document.getElementById('out').textContent = 'BEGINJSON' + JSON.stringify(all) + 'ENDJSON';
})();
</script></body></html>
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves website/, plus the harness page from memory.

    The harness is not written into website/ because that directory is the
    build output; dropping a test file in it would show up in a deploy.
    """

    harness = ""

    def do_GET(self):  # noqa: N802  (stdlib naming)
        if self.path.split("?")[0] == "/__check__.html":
            body = self.harness.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *a):  # keep the report readable
        pass


def _chrome() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser", "google-chrome-stable"):
        path = shutil.which(name)
        if path:
            return path
    return None


def run(pages=DEFAULT_PAGES, widths=DEFAULT_WIDTHS, timeout=600) -> list[dict]:
    """Measure every page at every width. Returns the raw per-case results."""
    if not (WEBSITE / "index.html").exists():
        raise SystemExit(
            f"No build found at {WEBSITE}. Run analysis/build_charts.py and "
            f"analysis/build_site.py first."
        )
    browser = _chrome()
    if browser is None:
        raise SystemExit("No Chrome or Chromium on PATH; cannot run the layout checks.")

    harness = (
        HARNESS.replace("__PAGES__", json.dumps(["/" + p for p in pages]))
        .replace("__WIDTHS__", json.dumps(list(widths)))
        .replace("__MIN_AXIS_RATIO__", repr(MIN_AXIS_RATIO))
        .replace("__MIN_DIV_WIDTH__", repr(MIN_DIV_WIDTH_TO_CHECK))
        .replace("__CROP_TOL__", repr(CROP_TOLERANCE_PX))
        .replace("__COLLIDE_TOL__", repr(COLLIDE_TOLERANCE_PX))
        .replace("__HOLLOW_TOL__", repr(HOLLOW_TOLERANCE_PX))
        .replace("__VOID_TOL__", repr(VOID_TOLERANCE_PX))
        .replace("__TOGGLE_TOL__", repr(TOGGLE_TOLERANCE_PX))
    )

    port = _free_port()
    handler = functools.partial(_Handler, directory=str(WEBSITE))
    handler.harness = harness
    _Handler.harness = harness

    with socketserver.ThreadingTCPServer(("127.0.0.1", port), handler) as httpd:
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        # Virtual time lets the in-page sleeps resolve without costing real
        # seconds, while still pausing for genuine network fetches (plotly.js
        # comes from a CDN).
        # 4s to settle each page, plus the TOGGLED check's clicks, which
        # spend virtual time like any other wait.
        budget = 20_000 + 10_000 * len(pages) * len(widths)
        proc = subprocess.run(
            [
                browser,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--hide-scrollbars",
                f"--virtual-time-budget={budget}",
                "--dump-dom",
                f"http://127.0.0.1:{port}/__check__.html",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        httpd.shutdown()

    m = re.search(r"BEGINJSON(.*?)ENDJSON", proc.stdout, re.S)
    if not m:
        raise SystemExit(
            "The harness did not report. Chrome stderr:\n" + proc.stderr[-2000:]
        )
    return json.loads(m.group(1))


def report(results: list[dict]) -> int:
    """Print a human-readable report. Returns the number of failures."""
    fails = 0
    for r in results:
        if r.get("error"):
            print(f"  ERROR  {r['page']} @ {r['width']}: {r['error']}")
            fails += 1
            continue
        head = f"{r['page']} @ {r['width']}px (doc {r['doc']}px)"
        problems = []
        for o in r["overflow"]:
            problems.append(f"OVERFLOW  {o['el']} is {o['w']}px wide")
        for c in r["cropped"]:
            problems.append(
                f"CROPPED   {c['chart']} needs {c['content']}px, frame is "
                f"{c['frame']}px ({c['cut']}px cut off)"
            )
        for s in r["squeezed"]:
            problems.append(
                f"SQUEEZED  {s['chart']} axis {s['axis']}px in a {s['divW']}px div "
                f"(ratio {s['ratio']}, margins l={s['marginL']} r={s['marginR']})"
            )
        for c in r.get("clipped", []):
            problems.append(
                f"CLIPPED   {c['chart']} text {c['text']!r} runs {c['by']}px "
                f"past the {c['side']} edge"
            )
        for c in r.get("colliding", []):
            problems.append(
                f"COLLIDING {c['chart']} tick {c['a']!r} overlaps {c['b']!r} "
                f"by {c['overlap']}px  A={c.get('ra')} B={c.get('rb')} n={c.get('ntext')}"
            )
        for h in r.get("hollow", []):
            problems.append(
                f"HOLLOW    {h['chart']} draws {h['ink']}px in a {h['frame']}px "
                f"frame ({h['blank']}px of white space below it)"
            )
        for v in r.get("voids", []):
            problems.append(
                f"VOID      {v['chart']} has a {v['gap']}px empty band at "
                f"y={v['at']}px inside the chart"
            )
        for u in r.get("untitled", []):
            problems.append(
                f"UNTITLED  {u['chart']} labels {u['have']} of its "
                f"{u['panels']} panels"
            )
        for t in r.get("toggled", []):
            problems.append(
                f"TOGGLED   {t['chart']} draws a "
                f"{t['a']['w']}x{t['a']['h']}px plot under {t['a']['mode']!r} "
                f"and {t['b']['w']}x{t['b']['h']}px under {t['b']['mode']!r}"
            )
        if problems:
            fails += len(problems)
            print(f"\nFAIL  {head}")
            for p in problems:
                print(f"        {p}")
        else:
            print(f"ok    {head}")
        for n in r.get("notes", []):
            print(f"        note: {n}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--widths", nargs="+", type=int, default=list(DEFAULT_WIDTHS))
    ap.add_argument("--pages", nargs="+", default=list(DEFAULT_PAGES))
    args = ap.parse_args()

    results = run(pages=args.pages, widths=args.widths)
    fails = report(results)
    print()
    if fails:
        print(f"{fails} problem(s) found.")
        return 1
    print("All responsive checks passed.")
    return 0


# ── pytest entry point ────────────────────────────────────────────────────
# Skips rather than fails where the inputs are absent, so CI without a browser
# or without a build does not report a red suite for a missing tool.


def test_responsive_layout():
    import pytest

    if _chrome() is None:
        pytest.skip("no Chrome/Chromium on PATH")
    if not (WEBSITE / "index.html").exists():
        pytest.skip("no build in website/; run build_charts.py and build_site.py")

    results = run()
    fails = report(results)
    assert fails == 0, f"{fails} responsive layout problem(s); see the report above"


if __name__ == "__main__":
    sys.exit(main())
