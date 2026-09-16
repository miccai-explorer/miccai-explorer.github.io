#!/usr/bin/env python3
"""MICCAI 2026 program booklet -> data/raw/miccai_2026_program.json.

The booklet is two documents in one binding, and they need separate readers:

  pages  2-10   talk sessions, three columns, title plus presenter, no paper IDs
  pages 11-105  poster listing, one column, every accepted paper with its paper
                ID, board number, full author list and presenter

Every accepted paper gets a poster, so the second part is the complete paper
list and the first only says which papers got a talk. That is why the type join
happens inside this one document and needs nothing else.

2026 has no reviews, no abstracts and no subject areas, so its output is read
only by the 2026 year page and never enters miccai_all.json.

Usage:
    python analysis/extract_program_2026.py            # write the JSON
    python analysis/extract_program_2026.py --dump     # print a sample only
"""

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from author_names import build_lookup, to_last_first  # noqa: E402
from extract_orals import extract_lines  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PDF_PATH = Path("data/manually_downloaded/OralSchedules/2026.pdf")
OUT_PATH = Path("data/raw/miccai_2026_program.json")
FACTS_PATH = Path("data/miccai_2026_facts.yaml")
ALL_JSON = Path("data/processed/miccai_all.json")

POSTER_FIRST_PAGE = 11
SESSION_PAGES = range(2, 11)

# "16 1 M-PM-001 Automatic LV Localization..." -> id, session, board, title tail.
# extract_lines collapses runs of whitespace to a single space, so \s+ is enough
# and the form feed pdftotext emits at page breaks never appears.
POSTER_ROW = re.compile(r"^(\d+)\s+(\d+)\s+([MTW]-(?:AM|PM)-\d{3})\s*(.*)$")
# Tolerant on purpose. The booklet prints "Presenters:" for the one paper with
# two of them (W-AM-202), and the strict spelling let that line fall through
# into the author list, where it became the author "Ying, Nicolas Padoy
# Presenters: Kun Yuan & Haochao". Earlier years of this pipeline were bitten
# the same way: 2023 prints "Speakers:" and 2025 "Authors :" with a space
# before the colon. Match the family, not the literal.
AUTHORS_RE = re.compile(r"^Authors?\s*:\s*(.*)$", re.IGNORECASE)
PRESENTER_RE = re.compile(r"^(?:Presenters?|Speakers?)\s*:\s*(.*)$", re.IGNORECASE)

# The poster table repeats its column headers on every one of its 95 pages.
POSTER_NOISE = re.compile(
    r"^(Poster|Paper Poster|Board Paper Title|ID Session|Number|Paper Title"
    r"|Board|Paper ID)\s*$",
    re.IGNORECASE,
)

# A presenter reads "Name, Affiliation, Country" and its affiliation can wrap
# onto the next line. So can the country itself: "United\nKingdom" and
# "South\nKorea" both occur. The line after a presenter is therefore either the
# rest of that address or the first line of the NEXT record's title, and
# getting it wrong glues an address onto a title, which is the join key for
# this whole file.
#
# The two cannot be told apart by position: measured over all 1,166 records the
# vertical gaps overlap (13.3-14.8pt against 13.4-14.8pt) and both sit at
# x0=168.0 exactly. Font size does not separate them either; every line in the
# table is 11.0pt.
#
# Weight does, exactly. Titles are set bold and addresses are not. Checked
# against ground truth read from the table's own cell rulings, which is
# independent of anything textual: of the 816 ambiguous lines, all 680 title
# lines are bold and all 136 address lines are not, with no overlap.
#
# An earlier version of this tested whether the address ended in a known
# country. That needs a vocabulary that can never be closed, and the booklet
# settles the argument by printing "United Kingdon / Germany".


def _is_address_continuation(line) -> bool:
    return not line.bold


def _join_wrapped(parts: list) -> str:
    """Join wrapped lines. A trailing hyphen means the word itself is split,
    so "Motion-" plus "Aware Modeling" joins without a space."""
    out = ""
    for part in parts:
        p = part.strip()
        if not p:
            continue
        if not out:
            out = p
        elif out.endswith("-"):
            out += p
        else:
            out += " " + p
    return " ".join(out.split())


def _split_presenter(text: str) -> tuple:
    """'Han Wu, ShanghaiTech University, China' -> name, affiliation, country.

    Affiliations contain commas of their own, so the country is taken as the
    last field and everything between it and the name is the affiliation.
    """
    bits = [b.strip() for b in text.split(",") if b.strip()]
    if not bits:
        return "", "", ""
    if len(bits) == 1:
        return bits[0], "", ""
    if len(bits) == 2:
        return bits[0], bits[1], ""
    return bits[0], ", ".join(bits[1:-1]), bits[-1]


def parse_posters(pdf_path: Path, lookup: dict) -> list:
    """Pages 11 onward -> one record per accepted paper."""
    # Line objects, not their text: the presenter boundary is decided by font
    # weight, which is the only thing that separates a wrapped address from
    # the next record's title.
    lines = [
        line
        for line in extract_lines(pdf_path, n_columns=1)
        if line.page >= POSTER_FIRST_PAGE
    ]

    records: list = []
    pending: list = []  # title lines seen before this record's ID row
    cur: dict = {}
    field = ""

    def close() -> None:
        if not cur:
            return
        cur["title"] = _join_wrapped(cur.pop("_title"))
        raw = _join_wrapped(cur.pop("_authors"))
        cur["authors"] = [
            to_last_first(a, lookup)[0]
            for a in (x.strip() for x in raw.split(","))
            if a.strip()
        ]
        name, aff, country = _split_presenter(_join_wrapped(cur.pop("_presenter")))
        cur["presenter"] = to_last_first(name, lookup)[0]
        cur["presenter_affiliation"] = aff
        cur["presenter_country"] = country
        records.append(dict(cur))

    i = 0
    while i < len(lines):
        line = lines[i]
        text = line.text.strip()
        i += 1
        if not text or POSTER_NOISE.match(text):
            continue

        m = POSTER_ROW.match(text)
        if m:
            close()
            num, session, board, tail = m.groups()
            cur = {
                "paper_id": f"miccai-2026-Paper{int(num):04d}",
                "year": 2026,
                "poster_session": int(session),
                "board_number": board,
                # A table cell that wraps upward puts the title's first line
                # above its own ID row, so anything buffered belongs here.
                "_title": pending + [tail],
                "_authors": [],
                "_presenter": [],
            }
            pending, field = [], "title"
            continue

        if not cur:
            pending.append(text)
            continue

        ma = AUTHORS_RE.match(text)
        if ma:
            field = "authors"
            cur["_authors"].append(ma.group(1))
            continue

        mp = PRESENTER_RE.match(text)
        if mp:
            field = "presenter"
            cur["_presenter"].append(mp.group(1))
            # Take the following non-bold lines as the rest of this address.
            # The first bold line is the next record's title.
            while i < len(lines) and _is_address_continuation(lines[i]):
                t = lines[i].text.strip()
                if not t or POSTER_NOISE.match(t) or POSTER_ROW.match(t):
                    break
                cur["_presenter"].append(t)
                i += 1
            continue

        if field == "presenter":
            pending.append(text)
        else:
            cur[f"_{field}"].append(text)

    close()
    return records


SESSION_HEAD = re.compile(r"^Oral(?: & Spotlight)? Session\s+(\S+)", re.I)
ORAL_MARKER = re.compile(r"^Oral Presentations?\s*:?\s*$", re.I)
SPOT_MARKER = re.compile(r"^Spotlight Presentations?\s*:?\s*$", re.I)

# The session pages separate their parts by type size, which is what makes them
# readable without guessing:
#   18pt  document title, printed once across the top
#   14pt  date and time banner, spanning all three columns so each column sees
#         a fragment of it
#   12pt  session code, then the session name in bold (which can wrap: O6B is
#         "Multimodal Integration & Outcome" plus "Prediction")
#   11pt  room name and session chairs
#   10pt  the talks themselves, title in bold and presenter not
SIZE_BANNER = 13.0
SIZE_SESSION = 11.5
SIZE_BODY = 10.5


def parse_talks(pdf_path: Path) -> list:
    """Pages 2-10 -> one record per talk, with its type and session name.

    State is kept per COLUMN INDEX, not per page: a session's spotlight half
    continues in the same column of the following page, so page 3 belongs to
    the sessions opened on page 2. Keying this per page credits those talks to
    the wrong session, and keying it globally credits them to whichever column
    was read last.
    """
    lines = [
        line
        for line in extract_lines(pdf_path, n_columns=3, split="gaps")
        if line.page in SESSION_PAGES
    ]

    cols: dict = {}
    talks: list = []

    def state(col: int) -> dict:
        return cols.setdefault(
            col,
            {"session": "", "kind": "oral", "title": [], "after_presenter": False,
             "naming": False},
        )

    def flush(col: int) -> None:
        s = state(col)
        if s["title"]:
            talks.append(
                {
                    "title": _join_wrapped(s["title"]),
                    "presentation_type": s["kind"],
                    "oral_session": s["session"],
                }
            )
            s["title"] = []
        s["after_presenter"] = False

    for line in lines:
        text = line.text.strip()
        size = round(line.size, 1)
        if not text or size >= SIZE_BANNER:
            continue

        s = state(line.col)

        if size >= SIZE_SESSION:
            if SESSION_HEAD.match(text):
                flush(line.col)
                s.update(session="", kind="oral", naming=True)
            elif line.bold and s["naming"]:
                s["session"] = f"{s['session']} {text}".strip()
            # Anything else at this size is page furniture, such as the
            # "Revised 2026-09-08" stamp in the second column of page 2.
            continue

        if size >= SIZE_BODY:
            continue  # room name and chairs

        s["naming"] = False
        if ORAL_MARKER.match(text):
            flush(line.col)
            s["kind"] = "oral"
            continue
        if SPOT_MARKER.match(text):
            flush(line.col)
            s["kind"] = "spotlight"
            continue

        if line.bold:
            # A bold line after a presenter starts the next talk.
            if s["after_presenter"]:
                flush(line.col)
            s["title"].append(text)
        else:
            s["after_presenter"] = True

    for col in list(cols):
        flush(col)
    return talks


def _norm_title(title: str) -> str:
    """Fold case, strip accents and punctuation, for joining talks to papers."""
    t = unicodedata.normalize("NFKD", title)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", t.casefold()).strip()


def join_types(records: list, talks: list, facts: dict) -> list:
    """Attach presentation types to papers. Pure: no file reading.

    Split out from build() so the tests can exercise the join, the counts and
    the failure mode without re-reading the 105-page PDF for each one.
    """
    by_title: dict = {}
    for r in records:
        by_title.setdefault(_norm_title(r["title"]), []).append(r)

    unmatched = []
    for t in talks:
        hits = by_title.get(_norm_title(t["title"]), [])
        if len(hits) != 1:
            unmatched.append(f"{t['title']}  [{len(hits)} candidates]")
            continue
        hits[0]["presentation_type"] = t["presentation_type"]
        hits[0]["oral_session"] = t["oral_session"]

    if unmatched:
        raise SystemExit(
            f"{len(unmatched)} talks did not join to exactly one paper. A "
            "dropped talk biases every presentation-type number on the page "
            "with nothing looking wrong, so this is fatal. First five:\n  "
            + "\n  ".join(unmatched[:5])
        )

    for r in records:
        r.setdefault("presentation_type", "poster")
        r.setdefault("oral_session", "")

    n_oral = sum(1 for r in records if r["presentation_type"] == "oral")
    n_spot = sum(1 for r in records if r["presentation_type"] == "spotlight")
    if (n_oral, n_spot) != (facts["n_orals"], facts["n_spotlights"]):
        raise SystemExit(
            f"parsed {n_oral} orals and {n_spot} spotlights, but "
            f"data/miccai_2026_facts.yaml says {facts['n_orals']} and "
            f"{facts['n_spotlights']}. Fix the parser or the facts file; never "
            "publish counts that disagree with the program."
        )
    return records


def build(pdf_path: Path, facts: dict, lookup: dict) -> list:
    """Poster records plus presentation types. Unmatched talks are fatal."""
    return join_types(
        parse_posters(pdf_path, lookup), parse_talks(pdf_path), facts
    )


def main() -> int:
    import yaml

    facts = yaml.safe_load(FACTS_PATH.read_text())
    lookup = build_lookup(
        json.loads(ALL_JSON.read_text()) if ALL_JSON.exists() else []
    )
    records = build(PDF_PATH, facts, lookup)

    if "--dump" in sys.argv:
        for r in records[:25]:
            print(
                f"{r['paper_id']}  {r['presentation_type']:9}  "
                f"{r['title'][:62]}"
            )
        return 0

    records.sort(key=lambda r: r["paper_id"])
    OUT_PATH.write_text(json.dumps(records, indent=1, ensure_ascii=False))
    logger.info(f"Wrote {OUT_PATH} with {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
