#!/usr/bin/env python3
"""
Extract oral / spotlight presentation schedules from the MICCAI program-book PDFs
→ data/raw/orals_YYYY.json

The PDFs live in data/manually_downloaded/OralSchedules/{year}.pdf and were
collected by hand (they are not published as structured data anywhere). All five
carry a real text layer, so no OCR is involved.

Why this is not a regex over `pdftotext` output
-----------------------------------------------
2021, 2024 and 2025 print two parallel session columns side by side. Naive text
extraction reads across both columns and splices unrelated titles together, which
silently corrupts the very field we join on. So words are placed into columns by
their x-midpoint before any text is reassembled.

Parsing model (identical for every year, parameters in config.yaml)
-------------------------------------------------------------------
1. pdfplumber gives words with font name, size and (x, y).
2. Assign each word to a column by x-midpoint; emit columns left to right.
3. Group words into lines by y-proximity.
4. Classify each line by font size into session-heading / session-meta / content.
5. Segment content lines into presentation blocks; by an explicit time prefix
   ("09:30-09:45") where the year prints one, otherwise wherever the vertical gap
   between lines exceeds `block_gap`. Intra-block leading is ~11-12pt in every
   year and roughly doubles between blocks, so this is a wide margin, not a
   knife edge.
6. Inside a block, the title is the run of lines before the first author or
   speaker marker. Where the year sets titles bold (2021-2023), the bold run is
   used instead; more reliable, because 2023 has speaker lines with no
   "Speaker:" prefix and 2021 sets some author lines in the regular face.

Type assignment is exact rather than heuristic: sessions headed "Oral & Spotlight
Session N" carry explicit "Oral Presentations:" / "Spotlight Presentations:"
subheaders, and sessions headed "Oral Session N" / "Oral N" are entirely oral.
The two-type split exists only from 2024 onward.

Usage:
    python analysis/extract_orals.py                 # all years in config
    python analysis/extract_orals.py --year 2025
    python analysis/extract_orals.py --dump 2024   # print blocks, write nothing
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

CONFIG = Path("config.yaml")
OUT_DIR = Path("data/raw")

# Lines that are page furniture rather than program content. Applied after the
# font-size filter, which already removes most of it; these catch the cases where
# banner text happens to be set at the same size as real content (2021's date
# banner, 2022's venue strapline).
#
# Patterns are deliberately specific. "Computer Assisted Intervention" appears in
# real session themes (2023 Oral 10), so only the all-caps banner form is matched.
BOILERPLATE = [
    re.compile(r"^\d{1,3}$"),                                  # page numbers
    re.compile(r"^MICCAI\s*\d{4}"),
    re.compile(r"presentation Schedule"),
    re.compile(r"Resorts World Convention Centre"),
    re.compile(r"International Conference on Medical Image Computing", re.I),
    re.compile(r"^AND COMPUTER ASSISTED INTERVENTION$"),
    re.compile(r"^ORAL AND SPOTLIGHT PRESENTATIONS$"),
    re.compile(r"^ORAL PRESENTATION PROGRAM$"),
    re.compile(r"^\d+\s*(TH|ST|ND|RD)\b", re.I),
    # 2021 splits its date/time banner across both columns into fragments
    re.compile(r"^(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day,?$"),
    re.compile(r"^\d{1,2}:\d{2}\s*[\u2013-]$"),
    re.compile(r"^\d{1,2}:\d{2}\s*\(UTC\)$"),
    re.compile(
        r"^(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s*\d{4}$"
    ),
]

TIME_RE = re.compile(
    r"^\d{1,2}:\d{2}\s*[-\u2013\u2014]\s*\d{1,2}:\d{2}\s*(?P<rest>.*)$"
)
# A date/time strapline. 2022 sets this at the same font size as the session
# heading, so without this it would be appended to the session theme.
DATE_TIME_RE = re.compile(
    r"\d{1,2}:\d{2}\s*(?:[-\u2013\u2014]|to)\s*\d{1,2}:\d{2}", re.I
)
HASH_RE = re.compile(r"^#\d+\s*")
HASH_ONLY_RE = re.compile(r"^#\d+$")
TYPE_RE = re.compile(r"^(Oral|Spotlight)\s+Presentations\s*:\s*$", re.I)
INVITED_RE = re.compile(r"^Invited\s+Session\s+Speakers?", re.I)
# Program entries that are not papers: 2023 schedules an invited talk and a
# panel inside otherwise ordinary oral sessions.
PANEL_RE = re.compile(r"^Panel\s+Discussion\b|^Panell?ists\s*:", re.I)
CHAIRS_RE = re.compile(r"^Session\s+Chairs?\s*:?|^Chairs\s*:", re.I)


@dataclass
class Line:
    """One reassembled line of text with the geometry needed to classify it."""

    text: str
    size: float
    x0: float
    top: float
    bold: bool
    italic: bool
    page: int
    col: int
    new_flow: bool  # first line of a column: vertical gap to previous is meaningless


@dataclass
class Block:
    """A run of content lines that forms one presentation."""

    lines: list = field(default_factory=list)
    ptype: str = "oral"
    page: int = 0
    col: int = 0


def _is_bold(fontname: str) -> bool:
    f = fontname.lower()
    return "bold" in f or "black" in f or f.endswith("bd")


def _is_italic(fontname: str) -> bool:
    f = fontname.lower()
    return "italic" in f or f.endswith("it")


def extract_lines(pdf_path: Path, n_columns: int) -> list:
    """PDF → ordered list of Line, columns emitted left to right within each page."""
    out: list = []
    with pdfplumber.open(pdf_path) as pdf:
        for pi, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["fontname", "size"])
            if not words:
                continue
            mid = page.width / 2
            cols: list = [[] for _ in range(n_columns)]
            for w in words:
                ci = (
                    0
                    if n_columns == 1
                    else (0 if (w["x0"] + w["x1"]) / 2 < mid else 1)
                )
                cols[ci].append(w)

            for ci, col_words in enumerate(cols):
                col_words.sort(key=lambda w: (round(w["top"], 1), w["x0"]))
                grouped: list = []
                cur: list = []
                cur_top = None
                for w in col_words:
                    if cur_top is None or abs(w["top"] - cur_top) < 4:
                        cur.append(w)
                        cur_top = w["top"] if cur_top is None else cur_top
                    else:
                        grouped.append(cur)
                        cur, cur_top = [w], w["top"]
                if cur:
                    grouped.append(cur)

                for li, g in enumerate(grouped):
                    g = sorted(g, key=lambda z: z["x0"])
                    text = " ".join(w["text"] for w in g).strip()
                    if not text:
                        continue
                    # 2021 prints a "#3" marker in the left gutter, set in the
                    # regular face. It lands on the same line as the title it
                    # numbers, so it must not get a vote on whether that line
                    # is bold; a two-word line would otherwise tie and lose.
                    voters = [
                        w for w in g if not HASH_ONLY_RE.match(w["text"])
                    ] or g
                    nb = sum(1 for w in voters if _is_bold(w["fontname"]))
                    ni = sum(1 for w in voters if _is_italic(w["fontname"]))
                    out.append(
                        Line(
                            text=text,
                            size=sum(w["size"] for w in g) / len(g),
                            x0=min(w["x0"] for w in g),
                            top=min(w["top"] for w in g),
                            bold=nb * 2 > len(voters),
                            italic=ni * 2 > len(voters),
                            page=pi,
                            col=ci,
                            new_flow=(li == 0),
                        )
                    )
    return out


def _is_boilerplate(text: str) -> bool:
    return any(p.search(text) for p in BOILERPLATE)


def _role(line: Line, cfg: dict) -> str | None:
    """Classify a line by font size into 'session', 'meta', 'content' or None."""
    tol = cfg["size_tol"]
    # Checked most specific first: 2023 shares one size between meta and content.
    for role, key in (
        ("session", "size_session"),
        ("content", "size_content"),
        ("meta", "size_meta"),
    ):
        if abs(line.size - cfg[key]) <= tol:
            return role
    return None


def _join_title(lines: list) -> str:
    """Join title lines, repairing the hyphenation the PDF introduces at wraps."""
    parts: list = []
    for ln in lines:
        t = ln.text.strip()
        if HASH_ONLY_RE.match(t):
            continue  # 2021 prints a bare "#3" marker in the left gutter
        t = HASH_RE.sub("", t)
        m = TIME_RE.match(t)
        if m:
            t = m.group("rest").strip()
        if not t:
            continue
        if parts and parts[-1].endswith("-"):
            parts[-1] = parts[-1] + t  # "Multi-" + "phase CTA" → "Multi-phase CTA"
        else:
            parts.append(t)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _split_block(block: Block, cfg: dict) -> dict | None:
    """Split one block's lines into title / authors / speaker."""
    lines = [ln for ln in block.lines if not HASH_ONLY_RE.match(ln.text.strip())]
    if not lines:
        return None

    # Program entries that are not papers (invited talks, panels).
    for ln in lines:
        t = ln.text.strip()
        if INVITED_RE.match(t) or PANEL_RE.match(t) or PANEL_RE.search(t):
            return None

    sp_re = re.compile(cfg["marker_speaker"]) if cfg.get("marker_speaker") else None
    au_re = re.compile(cfg["marker_authors"]) if cfg.get("marker_authors") else None

    speaker_lines: list = []
    rest = lines
    if sp_re:
        for i, ln in enumerate(lines):
            if sp_re.match(ln.text.strip()):
                speaker_lines = lines[i:]
                rest = lines[:i]
                break

    author_lines: list = []
    if au_re:
        for i, ln in enumerate(rest):
            if au_re.match(ln.text.strip()):
                author_lines = rest[i:]
                rest = rest[:i]
                break

    if cfg.get("title_is_bold"):
        # The title is the bold run; anything else left in the block is the
        # author or speaker line, whether or not it carried a prefix. This
        # catches 2023's speaker lines that omit "Speaker:" entirely and
        # 2021's author lines set in the regular rather than italic face.
        author_lines = author_lines + [ln for ln in rest if not ln.bold]
        rest = [ln for ln in rest if ln.bold]

    title = _join_title(rest)
    if not title:
        return None

    def _clean(lns: list, marker: "re.Pattern | None") -> str:
        txt = " ".join(ln.text.strip() for ln in lns)
        if marker:
            txt = marker.sub("", txt, count=1)
        return re.sub(r"\s+", " ", txt).strip()

    authors_txt = _clean(author_lines, au_re)
    speaker_txt = _clean(speaker_lines, sp_re)

    return {
        "type": block.ptype,
        "title_raw": title,
        "authors_raw": [a.strip() for a in authors_txt.split(",") if a.strip()]
        if authors_txt
        else [],
        "speaker": speaker_txt or None,
        "page": block.page,
        "column": block.col,
    }


class _ColState:
    """Parser state for one page column.

    Two-column years print each session down one column and continue it in the
    *same* column of the following page, so a single global "current session"
    mis-assigns every continuation to whichever session was seen most recently.
    State is therefore kept per column index.
    """

    def __init__(self) -> None:
        self.session: dict | None = None
        self.ptype = "oral"
        self.blocks: list = []
        self.block: Block | None = None
        self.prev_top: float | None = None
        self.prev_bold = False
        self.awaiting_theme = False


def parse_year(pdf_path: Path, cfg: dict) -> list:
    """Parse one year's PDF into a list of session dicts."""
    lines = extract_lines(pdf_path, cfg["n_columns"])
    session_re = re.compile(cfg["session_re"])
    has_types = cfg.get("has_types", False)
    block_gap = cfg["block_gap"]
    block_mode = cfg.get("block_mode", "gap")
    if block_mode not in ("time", "gap", "style"):
        raise ValueError(f"Unknown block_mode {block_mode!r} (time|gap|style)")

    sessions: list = []
    states: dict = {}

    def st(col: int) -> "_ColState":
        if col not in states:
            states[col] = _ColState()
        return states[col]

    def flush_block(s: "_ColState") -> None:
        if s.block is not None and s.block.lines:
            s.blocks.append(s.block)
        s.block = None

    def flush_session(s: "_ColState") -> None:
        if s.session is None:
            return
        pres: list = []
        for b in s.blocks:
            parsed = _split_block(b, cfg)
            if parsed:
                parsed["order"] = len(pres) + 1
                pres.append(parsed)
        s.session["presentations"] = pres
        sessions.append(s.session)
        s.blocks = []
        s.session = None

    for line in lines:
        s = st(line.col)
        text = line.text.strip()
        if _is_boilerplate(text):
            s.prev_top = None
            continue

        role = _role(line, cfg)
        if role is None:
            s.prev_top = None
            continue

        m = session_re.match(text)
        if m and role == "session":
            flush_block(s)
            flush_session(s)
            s.session = {
                "session_id": m.group("sid"),
                "session_title": (m.groupdict().get("theme") or "").strip(),
                "session_kind": "oral_spotlight"
                if re.search(r"&\s*Spotlight", text, re.I)
                else "oral",
                "day": None,
                "room": None,
                "chairs": [],
            }
            s.ptype = "oral"
            s.prev_top = None
            s.awaiting_theme = True
            continue

        if s.session is None:
            # Page furniture ahead of the first session heading in this column.
            s.prev_top = None
            continue

        if role == "session":
            # A line at heading size that is not itself a heading: either the
            # date strapline, the chairs, or the heading wrapping to a 2nd line.
            if DATE_TIME_RE.search(text):
                s.session["day"] = text
            elif CHAIRS_RE.match(text):
                v = CHAIRS_RE.sub("", text).strip()
                if v:
                    s.session["chairs"].append(v)
            elif not s.blocks and s.block is None:
                s.session["session_title"] = (
                    s.session["session_title"] + " " + text
                ).strip()
            s.prev_top = None
            continue

        if role == "meta" and cfg["size_meta"] != cfg["size_content"]:
            if s.awaiting_theme and not s.session["session_title"]:
                s.session["session_title"] = text
                s.awaiting_theme = False
            elif CHAIRS_RE.match(text):
                v = CHAIRS_RE.sub("", text).strip()
                if v:
                    s.session["chairs"].append(v)
            elif s.session["day"] is None:
                s.session["day"] = text
            elif s.session["room"] is None:
                s.session["room"] = text
            else:
                s.session["chairs"].append(text)
            s.prev_top = None
            continue

        # ---- content -------------------------------------------------------
        if TYPE_RE.match(text):
            flush_block(s)
            if has_types:
                s.ptype = text.split()[0].lower()
            s.prev_top = None
            continue

        if CHAIRS_RE.match(text):
            # 2022/2023 set the chairs line at content size.
            flush_block(s)
            v = CHAIRS_RE.sub("", text).strip()
            if v:
                s.session["chairs"].append(v)
            s.prev_top = None
            continue

        if block_mode == "time":
            starts_block = bool(TIME_RE.match(text))
        elif block_mode == "style":
            # A new presentation begins at a bold title line that follows the
            # previous presentation's non-bold author line.
            starts_block = s.block is None or (line.bold and not s.prev_bold)
        else:
            starts_block = (
                s.block is None
                or line.new_flow
                or s.prev_top is None
                or (line.top - s.prev_top) >= block_gap
            )

        if starts_block:
            flush_block(s)
            s.block = Block(lines=[], ptype=s.ptype, page=line.page, col=line.col)

        if s.block is None:
            # A stray content line before any block started (2022 venue strapline).
            s.prev_top = line.top
            continue

        s.block.lines.append(line)
        s.prev_top = line.top
        s.prev_bold = line.bold

    for s in states.values():
        flush_block(s)
        flush_session(s)

    # Emit in program order: numeric session id where the year uses one,
    # otherwise the order the sessions were encountered.
    if all(str(x["session_id"]).isdigit() for x in sessions):
        sessions.sort(key=lambda x: int(x["session_id"]))
    return sessions


def extract(year: int, cfg: dict, pdf_dir: Path) -> dict:
    pdf_path = pdf_dir / f"{year}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing schedule PDF: {pdf_path}")

    sessions = parse_year(pdf_path, cfg)
    n_pres = sum(len(s["presentations"]) for s in sessions)

    # Session numbers must be contiguous. A gap means a page of the program was
    # not included in the PDF, which would silently undercount the oral type.
    ids = [s["session_id"] for s in sessions]
    numeric = [int(i) for i in ids if str(i).isdigit()]
    if numeric:
        expected = list(range(1, max(numeric) + 1))
        missing = sorted(set(expected) - set(numeric))
        if missing:
            raise ValueError(
                f"{year}: session numbers not contiguous; missing {missing}. "
                f"The source PDF is probably missing a page."
            )
    exp = cfg.get("expected_sessions")
    if exp is not None and len(sessions) != exp:
        raise ValueError(
            f"{year}: parsed {len(sessions)} sessions, config expects {exp}. "
            f"Parsed ids: {ids}"
        )

    return {
        "year": year,
        "source_pdf": str(pdf_path),
        "n_sessions": len(sessions),
        "n_presentations": n_pres,
        "sessions": sessions,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, help="Single year (default: all configured)")
    ap.add_argument(
        "--dump",
        type=int,
        help="Print parsed blocks for a year, write nothing",
    )
    args = ap.parse_args()

    with open(CONFIG, encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)
    osched = cfg_all.get("oral_schedules") or {}
    pdf_dir = Path(osched.get("pdf_dir", "data/manually_downloaded/OralSchedules"))
    years = sorted(k for k in osched if isinstance(k, int))

    if args.dump:
        data = extract(args.dump, osched[args.dump], pdf_dir)
        for s in data["sessions"]:
            print(f"\n=== Session {s['session_id']}: {s['session_title']} "
                  f"[{s['session_kind']}] ({len(s['presentations'])}) ===")
            print(f"    day={s['day']!r} room={s['room']!r} chairs={s['chairs']}")
            for p in s["presentations"]:
                print(f"  [{p['type'][:4]}] {p['title_raw'][:88]}")
                if p["speaker"]:
                    print(f"         spk: {p['speaker'][:70]}")
        print(
            f"\nTOTAL sessions={data['n_sessions']} "
            f"presentations={data['n_presentations']}"
        )
        return 0

    targets = [args.year] if args.year else years
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grand = 0
    for yr in targets:
        if yr not in osched:
            logger.error(f"{yr}: no oral_schedules config block")
            return 1
        data = extract(yr, osched[yr], pdf_dir)
        out = OUT_DIR / f"orals_{yr}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        ptypes: dict = {}
        for s in data["sessions"]:
            for p in s["presentations"]:
                ptypes[p["type"]] = ptypes.get(p["type"], 0) + 1
        logger.info(
            f"{yr}: {data['n_sessions']} sessions, "
            f"{data['n_presentations']} presentations {ptypes} → {out}"
        )
        grand += data["n_presentations"]
    logger.info(f"Total presentations extracted: {grand}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
