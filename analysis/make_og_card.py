#!/usr/bin/env python3
"""Draw static/og-card.png, the 1200x630 image link previews show.

Run by hand, not by the build:

    python analysis/make_og_card.py

The output is a TRACKED SOURCE asset in static/, which build_site.py then
copies to website/assets/ with everything else. That is the opposite of how the
year logos work (converted at build time by logo_assets.py), and the reason is
fonts: this card renders text, so building it in CI would need a specific
typeface installed on the runner. A missing font there would not fail the
build, it would silently draw the card in a fallback face, and nobody would
see it because nobody looks at a social card on a deploy. Rendering it once,
here, and reviewing the PNG in the diff removes that whole class of problem.

Faces are resolved by family NAME through fontconfig, never by path: the
preferred families are installed in a home directory on this machine, and
make_release.sh refuses to publish a tree containing that kind of path.

Re-run it when the headline numbers change enough to be worth restating, which
in practice means when a new year of MICCAI is added. Nothing breaks if it is
never re-run; the card simply quotes an older count.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("static/og-card.png")
ALL_JSON = Path("data/processed/miccai_all.json")
# The year the chart draws. The most recent one with a complete
# review record, which is also the one the headline counts end at.
CHART_YEAR = 2025

# 1200x630 is the size every platform states and the one they crop least.
W, H = 1200, 630

# Pale sage ground, deep ink-green type, and a score ramp.
#
# What this is NOT, and why. Three light-card defaults turn up regardless of
# subject: warm cream near #F4F1EA with a serif and a terracotta accent; the
# navy-and-cyan that is the house style of every AI product shipped since 2023;
# and near-black with one acid accent. This ground is a COOL sage (green above
# red and blue) rather than the warm cream, which is the difference between
# looking like a botanical plate and looking like every other generated card.
#
# The site itself is light (#f1f5f9 page, white cards), so a light card also
# stops the odd situation of a dark card advertising a light site. The earlier
# dark draft had borrowed its navy from one element, the stat strip, and then
# represented the whole thing.
GROUND = (233, 237, 232)  # #e9ede8  pale sage; cool, not cream
INK = (23, 41, 31)  # #17291f  deep ink green; 12.9:1 on the ground
MUTED = (90, 109, 98)  # #5a6d62  sage grey; 4.7:1, passes AA as body text

# The bars run this ramp from the lowest score to the highest. The colour is
# the score axis, not decoration: it is the one place a reader can see that the
# scale has a direction, and which end of it is the good one.
#
# Green sits at the HIGH end. A higher reviewer score is a better review, and
# green reading as "good" is about as settled as colour convention gets; the
# bronze marks the rejections at the bottom of the scale. The first draft had
# this backwards, ramping green to bronze as the score rose, which quietly told
# the opposite story to the one the data tells.
#
# Colour is redundant here, not load-bearing: every bar is labelled with its
# own score underneath. That matters because green against amber is one of the
# pairs red-green colour blindness flattens, and the numerals are what carry
# the axis for a reader who cannot separate the two.
#
# On the dark draft this ramp ran dark-to-bright, which is the natural reading
# there. It cannot here: a pale bar on a pale ground measures about 1.5:1 and
# simply disappears. So on light the ramp holds every bar inside a legible
# 3.4:1 to 5.0:1 band and shifts WARMTH instead of weight. Same information,
# drawn the only way this ground allows.
BAR_LOW = (168, 118, 26)  # #a8761a  score 1, bronze, 3.4:1
BAR_HIGH = (77, 107, 96)  # #4d6b60  score 6, green, 5.0:1

# Font families in preference order, resolved BY NAME through fontconfig
# rather than by path. Two reasons, and the second one is not optional:
#
#   1. A path is wrong on every machine but one.
#   2. Fira Sans here is installed under the maintainer's home directory, and
#      make_release.sh greps the whole tree for exactly that kind of path
#      before publishing. Hardcoding it would fail the release sweep, which is
#      the correct outcome: a username does not belong in a public repository.
#
# Inter is the site's own face and sits first, so a machine that has it draws
# a card that matches the site exactly. Fira Sans is second: it is a humanist
# screen face with well-drawn numerals, which this card is mostly made of.
FONT_FAMILIES = ["Inter", "Fira Sans", "Noto Sans", "DejaVu Sans"]

# Weights are fontconfig's numeric scale, NOT CSS's. Light 50, Regular 80,
# Medium 100, SemiBold 180, Bold 200, ExtraBold 205. Roles below are ordered
# fallbacks: the first weight that is genuinely installed wins.
#
# The card is light text on a dark ground, where type optically thickens, so
# body asks for Regular and takes Light before Medium. Stepping DOWN a weight
# on dark is the usual correction; stepping up is the usual mistake.
#
# That is not theoretical here. Fira Sans is installed on this machine in 22
# faces from Thin to Black with NO Regular among them, and fc-match does not
# say so: asked for weight 80 it returns Medium at weight 100, one step the
# wrong way, and reports its style as "Medium,Regular" so a style-name check
# accepts it. Only comparing the numeric weight catches this.
LIGHT, REGULAR, MEDIUM, SEMIBOLD, BOLD, EXTRABOLD = 50, 80, 100, 180, 200, 205

ROLES = {
    # A Light headline at 54px rather than the reflexive heavy bold. At this
    # size Light still has plenty of presence and it lets the chart, not the
    # type, be the loudest thing on the card.
    "wordmark": ([SEMIBOLD, MEDIUM, BOLD], 38),
    "headline": ([LIGHT, REGULAR, MEDIUM], 48),
    "meta": ([REGULAR, LIGHT, MEDIUM], 24),
    "label": ([MEDIUM, SEMIBOLD, REGULAR], 19),
}
_WEIGHT_NAMES = {
    LIGHT: "Light",
    REGULAR: "Regular",
    MEDIUM: "Medium",
    SEMIBOLD: "SemiBold",
    BOLD: "Bold",
    EXTRABOLD: "ExtraBold",
}


def _query(family: str, weight: int) -> str | None:
    """Ask fontconfig for one family at one exact weight, upright.

    Both returned values have to be checked, and each catches a different
    silent substitution that the other misses:

      family  fc-match never fails. Given a family that is not installed it
              returns its best guess, some unrelated default. Without this
              check the script would report "using Inter" on a machine with
              none and draw the card in DejaVu.
      weight  fc-match also answers with the NEAREST weight it has, and its
              style string is no help: Fira Sans Medium reports its style as
              "Medium,Regular", so a style-name check accepts it as Regular.
              Comparing the number is what makes a missing weight visible and
              lets the ladder in ROLES fall through to the next one.
    """
    fmt = "%{file}\t%{family}\t%{weight}"
    try:
        out = subprocess.run(
            ["fc-match", "-f", fmt, f"{family}:weight={weight}:slant=0"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    parts = out.stdout.split("\t")
    if out.returncode != 0 or len(parts) != 3:
        return None
    path, families, got_weight = parts
    # %{family} can be a comma-separated list: "Fira Sans,Fira Sans Medium".
    names = {f.strip().lower() for f in families.split(",")}
    if not path or family.lower() not in names:
        return None
    try:
        if int(got_weight) != weight:
            return None
    except ValueError:
        return None
    return path


def _fonts():
    """Return {role: ImageFont}, all four faces from one family.

    All four come from one family on purpose: mixing families across a single
    card reads as a mistake rather than as a choice. A family that cannot
    supply every role is skipped whole for the same reason.
    """
    from PIL import ImageFont

    for family in FONT_FAMILIES:
        picked = {}
        for role, (weights, size) in ROLES.items():
            for w in weights:
                path = _query(family, w)
                if path:
                    picked[role] = (path, w, size)
                    break
            if role not in picked:
                break
        if len(picked) != len(ROLES):
            continue

        logger.info(f"  Family: {family}")
        for role, (path, w, size) in picked.items():
            wanted = ROLES[role][0][0]
            note = (
                ""
                if w == wanted
                else f"   <- no {_WEIGHT_NAMES[wanted]} installed"
            )
            logger.info(
                f"    {role:<6} {_WEIGHT_NAMES[w]:<10} {size:>3}px  "
                f"{Path(path).name}{note}"
            )
        return {
            r: ImageFont.truetype(picked[r][0], picked[r][2]) for r in ROLES
        }

    raise SystemExit(
        "No usable font found. Install one of: "
        + ", ".join(FONT_FAMILIES)
        + " (and fontconfig, for fc-match)."
    )


def total_reviews() -> int:
    """Every review across every year, for the headline.

    Deliberately not the chart year's count: the headline describes what the
    site covers, the chart describes one year of it. Counting both from the
    same file means neither can go stale while the other is updated.
    """
    with open(ALL_JSON, encoding="utf-8") as fh:
        return sum(len(p.get("reviews") or []) for p in json.load(fh))


def score_distribution(year: int = 2025) -> tuple[list[int], int]:
    """Counts of each reviewer score for one year, read from the real data.

    The chart on this card is the actual distribution, not a drawn shape. That
    is the whole point of it: this project exists because MICCAI publishes its
    reviewer scores, so the card shows them rather than showing an abstract
    texture that could belong to any project.

    Returns (counts indexed 1..scale_max, scale_max). Scores outside 1..max are
    dropped: both 2024 and 2025 carry exactly one review recorded as 0 on a
    1-6 scale, which is one in three thousand and not worth drawing a bar for.
    """
    if not ALL_JSON.exists():
        raise SystemExit(
            f"{ALL_JSON} not found. This card draws the real score "
            f"distribution, so it needs the processed data; run the pipeline "
            f"first, or pass a year that exists."
        )
    with open(ALL_JSON, encoding="utf-8") as f:
        papers = json.load(f)

    counts: dict[int, int] = {}
    scale = 0
    for paper in papers:
        if paper.get("year") != year:
            continue
        for review in paper.get("reviews") or []:
            raw, mx = review.get("score_raw"), review.get("score_max")
            if raw is None or mx is None:
                continue
            scale = max(scale, mx)
            if 1 <= raw <= mx:
                counts[raw] = counts.get(raw, 0) + 1
    if not counts:
        raise SystemExit(f"No reviewer scores found for {year}.")
    return [counts.get(i, 0) for i in range(1, scale + 1)], scale


def _fit(d, text: str, font, limit: int, what: str) -> str:
    """Return text, or raise if it would run past the card's safe width.

    A clipped social card is not a visible failure anywhere: the script exits
    0, the PNG looks fine at a glance, and the missing half-word only shows up
    once the link is posted somewhere. The headline overflowed by 14px on the
    first draft of this layout and the render still looked plausible. So the
    width is asserted rather than trusted.
    """
    w = d.textlength(text, font=font)
    if w > limit:
        raise SystemExit(
            f"{what} is {w:.0f}px wide, over the {limit}px safe width:\n"
            f"  {text!r}\n"
            f"Shorten it, or drop the size in ROLES."
        )
    return text


def main() -> int:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        logger.error("Pillow is required: pip install Pillow")
        return 1

    f = _fonts()
    counts, scale = score_distribution(CHART_YEAR)
    total = sum(counts)
    logger.info(
        f"  {CHART_YEAR}: {total:,} reviewer scores on a 1-{scale} scale"
    )

    img = Image.new("RGB", (W, H), GROUND)
    d = ImageDraw.Draw(img)

    PAD = 76

    # ── type block ──────────────────────────────────────────
    d.text((PAD, 66), "MICCAI Explorer", font=f["wordmark"], fill=INK)

    # Right-aligned on the wordmark's own baseline, so the attribution is read
    # with the name of the thing rather than buried at a corner.
    byline = "Kumar Abhishek"
    d.text(
        (W - PAD - d.textlength(byline, font=f["meta"]), 78),
        byline,
        font=f["meta"],
        fill=MUTED,
    )

    y = 158
    safe = W - 2 * PAD
    for line in (
        # "Every accepted MICCAI paper from 2021 to 2025,",
        # f"and the {total_reviews():,} peer reviews behind them.",
        "MICCAI papers and their peer reviews",
        "from 2021 to 2025.",
    ):
        _fit(d, line, f["headline"], safe, "headline")
        d.text((PAD, y), line, font=f["headline"], fill=INK)
        y += 58

    d.text((PAD, 300), "miccai-explorer.github.io", font=f["meta"], fill=INK)

    # ── the chart ───────────────────────────────────────────
    # Six real bars beat three hundred drawn dots: a social card is read at
    # about 500px wide in a timeline, where fine texture turns to mush and a
    # bar chart still reads.
    base_y = 548  # baseline the bars stand on
    max_h = 186
    n = len(counts)
    slot = (W - 2 * PAD) / n
    bar_w = slot * 0.56
    peak = max(counts)

    for i, count in enumerate(counts):
        cx = PAD + slot * (i + 0.5)
        # A floor of 6px, not 1: score 6 is 2.3% of reviews and would
        # otherwise be a hairline indistinguishable from the baseline.
        h = max(6, round(count / peak * max_h))
        t = i / (n - 1) if n > 1 else 0
        col = tuple(
            round(BAR_LOW[j] + (BAR_HIGH[j] - BAR_LOW[j]) * t)
            for j in range(3)
        )
        d.rectangle(
            [cx - bar_w / 2, base_y - h, cx + bar_w / 2, base_y], fill=col
        )
        # The score under its own bar. Centred per bar rather than set as an
        # evenly spaced row, so a short bar keeps its label.
        lab = str(i + 1)
        d.text(
            (cx - d.textlength(lab, font=f["label"]) / 2, base_y + 14),
            lab,
            font=f["label"],
            fill=MUTED,
        )

    # Above the bars, not on the numeral row: right-aligned there it sat on
    # top of the 5 and the 6, which are the two shortest bars and so the two
    # whose labels have the least room to give.
    cap = f"Reviewer scores, MICCAI {CHART_YEAR}"
    d.text(
        (W - PAD - d.textlength(cap, font=f["label"]), base_y - max_h - 44),
        cap,
        font=f["label"],
        fill=MUTED,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # optimize is worth it here: the card is fetched by every link preview.
    img.save(OUT, "PNG", optimize=True)
    kb = OUT.stat().st_size / 1024
    logger.info(f"Saved {OUT} ({W}x{H}, {kb:.0f} KB)")
    if kb > 300:
        logger.warning(
            "  Card is over 300 KB; some scrapers skip large images."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
