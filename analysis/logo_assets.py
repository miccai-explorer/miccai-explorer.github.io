#!/usr/bin/env python3
"""
Shrink the conference year logos on their way into website/assets/.

Reads  : static/logos/*.png   (tracked source; never modified)
Writes : website/assets/logos/*.webp

Called by build_site.py right where static/ is copied to website/assets/, so
adding a year needs no step of its own: drop miccai_2026.png into static/logos/
and rebuild. The PNGs stay the source of truth because they are what a person
edits and what the conference publishes; the WebP files are build output like
everything else under website/.

Usage (standalone, to inspect what the build would produce):
    python analysis/logo_assets.py [--out DIR]

Why not a git hook: a hook that rewrites tracked files makes a commit differ
from what was reviewed, and it would put generated artefacts in static/, which
this repository reserves for source. Converting at build time keeps the rule
that static/ is source and website/ is generated.

Measured on the five 2021-2025 logos, total over all five (they all sit in the
nav, so every page load fetches all of them):

    as committed, PNG at 576-768 px wide     223 KB
    WebP at the same pixel size               92 KB
    resized to 160 px tall, still PNG        119 KB
    resized to 160 px tall, WebP              61 KB   <- what this writes

Most of the saving is the resize, not the format. The logos are displayed at
80 px tall on the year pages (.year-logo-img) and 28 px in the nav, so 160 px
covers a 2x display and the sources were two to four times larger than they can
ever be shown.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SRC_DIR = Path("static/logos")

# Twice the tallest size any logo is drawn at (80 px, .year-logo-img in
# static/style.css). Raise this if that CSS grows; a logo shown larger than
# half this height will look soft on a high-density display.
MAX_HEIGHT = 160

# alpha_quality is the setting that matters here, not quality. About three
# quarters of every logo is fully transparent, and libwebp encodes the alpha
# channel separately: at the default alpha_quality=100 the five files come to
# 94 KB, at 70 they come to 61 KB. Compared at 4x zoom against the nav
# background there is no visible fringing on the thin script lettering, which
# is the worst case in this set. Going lower buys almost nothing (58 KB at 60),
# so there is no reason to push it.
QUALITY = 80
ALPHA_QUALITY = 70
METHOD = 6  # slowest and smallest; five small images, so the time is free


def optimize(src_dir: Path, dst_dir: Path) -> dict[str, str]:
    """Write a WebP copy of every PNG in src_dir into dst_dir.

    Returns a {"miccai_2025.png": "miccai_2025.webp"} map of the files it
    converted, for the caller to rewrite its logo filenames with.

    Returns an empty dict, and leaves any already-copied PNGs in place, when
    Pillow is not installed. The site is then exactly what it was before this
    module existed, which is the point: a missing optional dependency must not
    break the build.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning(
            "  Pillow not installed: logos ship as PNG "
            "(about 160 KB per page load larger). pip install pillow"
        )
        return {}

    renamed: dict[str, str] = {}
    dst_dir.mkdir(parents=True, exist_ok=True)
    before = after = 0
    for png in sorted(src_dir.glob("*.png")):
        img = Image.open(png).convert("RGBA")
        if img.height > MAX_HEIGHT:
            width = round(img.width * MAX_HEIGHT / img.height)
            img = img.resize((width, MAX_HEIGHT), Image.LANCZOS)
        out = dst_dir / f"{png.stem}.webp"
        img.save(
            out,
            "WEBP",
            quality=QUALITY,
            alpha_quality=ALPHA_QUALITY,
            method=METHOD,
        )
        # The PNG was copied in by the caller's copytree; drop it so the site
        # does not carry both. Nothing references it: every use goes through
        # year_meta[yr].logo, which the caller rewrites from the returned map.
        (dst_dir / png.name).unlink(missing_ok=True)
        renamed[png.name] = out.name
        before += png.stat().st_size
        after += out.stat().st_size

    if renamed:
        logger.info(
            f"  Logos: {len(renamed)} PNG -> WebP, "
            f"{before / 1024:.0f} KB -> {after / 1024:.0f} KB "
            f"({100 * (1 - after / before):.0f}% smaller)"
        )
    return renamed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert static/logos/*.png to WebP, as build_site.py does."
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("website/assets/logos"),
        help="output directory (default: website/assets/logos)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if not SRC_DIR.exists():
        logger.error(f"{SRC_DIR} not found; run from the repository root")
        return 1
    # Standalone, there is no copytree to have put the PNGs there, so put them
    # there first: optimize() deletes the PNG beside each WebP it writes, and
    # this way the standalone run leaves exactly what a build would leave.
    args.out.mkdir(parents=True, exist_ok=True)
    for png in SRC_DIR.glob("*.png"):
        shutil.copy2(png, args.out / png.name)
    renamed = optimize(SRC_DIR, args.out)
    if not renamed:
        return 1
    for old, new in sorted(renamed.items()):
        logger.info(f"    {old} -> {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
