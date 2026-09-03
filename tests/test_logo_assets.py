"""Guards on the year-logo conversion in analysis/logo_assets.py.

The failure this file is really here for is a silent one. `optimize()` returns
an empty map and leaves the PNGs alone when Pillow is missing, which is what
keeps a fresh clone building, but it also means a deploy environment without
Pillow would ship the unconverted PNGs and look completely normal while doing
it. Asserting that the conversion actually happened turns that into a failed
test instead of a site that is quietly 160 KB heavier per page load.

The rest guards the rule that `static/` is source: the build reads those PNGs
and must never write to them.
"""

import hashlib
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import logo_assets  # noqa: E402

SRC = ROOT / "static" / "logos"


def _hashes() -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(SRC.glob("*.png"))
    }


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """Run the conversion exactly as build_site.py does: copy, then optimize."""
    out = tmp_path_factory.mktemp("logos")
    before = _hashes()
    for png in SRC.glob("*.png"):
        shutil.copy2(png, out / png.name)
    return out, logo_assets.optimize(SRC, out), before


def test_every_png_became_a_webp(converted):
    out, renamed, _ = converted
    pngs = sorted(p.name for p in SRC.glob("*.png"))
    assert pngs, "no logos found; is this running from the repository root?"
    assert sorted(renamed) == pngs
    for name in pngs:
        assert (out / f"{Path(name).stem}.webp").is_file()


def test_no_png_is_left_beside_the_webp(converted):
    out, _, _ = converted
    assert list(out.glob("*.png")) == []


def test_the_conversion_is_worth_doing(converted):
    """Under half the bytes. Measured at 61 KB against 223 KB, so this has a
    lot of slack; it fires if the conversion silently stops happening, not on
    a logo that happens to compress badly."""
    out, _, _ = converted
    before = sum(p.stat().st_size for p in SRC.glob("*.png"))
    after = sum(p.stat().st_size for p in out.glob("*.webp"))
    assert after < before / 2, f"{after} bytes from {before}: conversion is off"


def test_height_is_capped_and_the_shape_is_kept(converted):
    Image = pytest.importorskip("PIL.Image", reason="Pillow drives the check")
    out, _, _ = converted
    for png in sorted(SRC.glob("*.png")):
        src = Image.open(png)
        webp = Image.open(out / f"{png.stem}.webp")
        assert webp.height <= logo_assets.MAX_HEIGHT
        # Aspect ratio within a pixel of the source, since the width is rounded.
        expected = round(src.width * webp.height / src.height)
        assert abs(webp.width - expected) <= 1


def test_the_source_pngs_are_not_touched(converted):
    """static/ is source. A build that rewrote these would be a bug, and it is
    the reason the conversion is a build step and not a git hook."""
    _, _, before = converted
    assert _hashes() == before
