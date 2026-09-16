"""Golden counts for the 2026 program, re-parsed from the source PDF.

Modelled on tests/test_orals_data.py: the committed JSON is not trusted on its
own, because a parser that silently drops records leaves no trace in its own
output. Every count here comes from the booklet, not from what the parser
happened to produce.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, "analysis")
import yaml  # noqa: E402
from extract_program_2026 import (  # noqa: E402
    join_types,
    parse_posters,
    parse_talks,
)

PDF = Path("data/manually_downloaded/OralSchedules/2026.pdf")
FACTS_PATH = Path("data/miccai_2026_facts.yaml")
pytestmark = pytest.mark.skipif(not PDF.exists(), reason="2026 program PDF absent")

FACTS = yaml.safe_load(FACTS_PATH.read_text()) if FACTS_PATH.exists() else {}


@pytest.fixture(scope="module")
def posters():
    return parse_posters(PDF, {})


@pytest.fixture(scope="module")
def talks():
    return parse_talks(PDF)


@pytest.fixture(scope="module")
def records(posters, talks):
    """Reuses the two parses above. Calling build() here would re-read the
    105-page PDF twice more; the whole file took 87s that way."""
    import copy

    return join_types(copy.deepcopy(posters), talks, FACTS)


def test_record_count(posters):
    assert len(posters) == 1166


def test_paper_ids_are_unique(posters):
    assert len({p["paper_id"] for p in posters}) == 1166


def test_board_numbers_are_unique(posters):
    assert len({p["board_number"] for p in posters}) == 1166


def test_every_record_has_a_title_and_authors(posters):
    assert all(p["title"].strip() for p in posters)
    assert all(p["authors"] for p in posters)


def test_authors_are_last_first(posters):
    """Every stored name is 'Last, First' or a single token, never more."""
    for p in posters:
        for a in p["authors"]:
            assert a.count(",") <= 1, f"{a!r} in {p['paper_id']}"


def test_no_table_header_leaked_in_as_a_title(posters):
    """The poster table repeats its column headers on every page."""
    for p in posters:
        assert "Paper Title" not in p["title"]
        assert "Board" not in p["title"].split()[:1]


def test_no_affiliation_bled_into_a_title(posters):
    """The bug this test exists for: a presenter's affiliation wraps onto a
    second line, and reading that line as the next record's title produced
    'Electronic Science and Technology of China, China PromptMedCT: A
    Training-Free LLM-Guided Framework...'. Every count still came out at
    1,166, so only looking at the titles could catch it.

    A real title never contains a country name preceded by a comma."""
    import re

    leak = re.compile(
        r",\s*(China|United States|Germany|South Korea|United Kingdom|France|"
        r"Canada|Australia|Japan|Singapore|Italy|India|Switzerland)\b"
    )
    offenders = [p for p in posters if leak.search(p["title"])]
    assert not offenders, (
        f"{len(offenders)} titles carry an address fragment, first: "
        f"{offenders[0]['board_number']} {offenders[0]['title'][:90]!r}"
    )


def test_titles_are_a_plausible_length(posters):
    """A title with an address glued on runs far longer than any real one."""
    longest = max(posters, key=lambda p: len(p["title"].split()))
    assert len(longest["title"].split()) <= 30, (
        f"{longest['board_number']} has {len(longest['title'].split())} words: "
        f"{longest['title'][:110]!r}"
    )


def test_known_record_parses_exactly(posters):
    """M-PM-002's title starts on the line ABOVE its ID row and wraps on a
    hyphen, which is the case a naive line reader gets wrong twice over."""
    rec = next(p for p in posters if p["board_number"] == "M-PM-002")
    assert rec["title"] == (
        "Weakly-Supervised Coronary Artery Segmentation from DSA Sequence "
        "via Motion-Aware Modeling"
    )
    assert rec["paper_id"] == "miccai-2026-Paper0033"
    assert rec["authors"][0] == "Wu, Han"
    assert rec["authors"][-1] == "Shen, Dinggang"
    assert rec["presenter_country"] == "China"


def test_first_record_parses_exactly(posters):
    rec = next(p for p in posters if p["board_number"] == "M-PM-001")
    assert rec["title"] == (
        "Automatic LV Localization and Short-Axis Plane Estimation from "
        "Arbitrary CMR Slice"
    )
    assert len(rec["authors"]) == 7
    assert rec["presenter"] == "Xue, Yuan"


def test_plural_presenters_marker_is_handled(posters):
    """W-AM-202 is the one paper the booklet gives two presenters, and it
    writes 'Presenters:'. Matching the literal 'Presenter:' let that line fall
    into the author list as 'Ying, Nicolas Padoy Presenters: Kun Yuan &
    Haochao'. Same family of bug as 2023's 'Speakers:' and 2025's 'Authors :'."""
    rec = next(p for p in posters if p["board_number"] == "W-AM-202")
    assert rec["authors"] == [
        "Hu, Yaojun",
        "Yuan, Kun",
        "Navab, Nassir",
        "Ying, Haochao",
        "Wu, Jian",
        "Padoy, Nicolas",
    ]
    assert rec["presenter_country"] == "France"


def test_no_author_name_contains_a_marker_word(posters):
    """A marker the parser failed to recognise ends up inside a name."""
    for p in posters:
        for a in p["authors"]:
            low = a.casefold()
            assert "presenter" not in low and "author" not in low, (
                f"{a!r} in {p['board_number']}"
            )


def test_no_author_name_carries_an_address(posters):
    """A swallowed marker drags the presenter's affiliation in with it, so the
    corrupted entries were 'Strasbourg, University of' and 'France'.

    Length does not discriminate here and a test using it would be theatre:
    the corrupted name was 49 characters and the longest genuine name in five
    years of existing data is 51 ('Consortium, Simons Variation in Individuals
    Project'), with real personal names reaching 47."""
    import re

    address = re.compile(
        r"\b(University|Institute|Hospital|Laborator|Academy|"
        r"China|France|Germany|United States|South Korea)\b",
        re.IGNORECASE,
    )
    offenders = [
        (p["board_number"], a)
        for p in posters
        for a in p["authors"]
        if address.search(a)
    ]
    assert not offenders, f"{len(offenders)} names carry an address: {offenders[:3]}"


def test_poster_sessions_are_one_to_five(posters):
    assert {p["poster_session"] for p in posters} == {1, 2, 3, 4, 5}


def test_year_is_set(posters):
    assert all(p["year"] == 2026 for p in posters)


def test_agrees_with_the_ruled_table(posters):
    """Cross-check against a method that shares no logic with the parser.

    The poster listing is a real table with drawn cell borders, so pdfplumber
    can read each record straight out of its own cell. That is derived from
    the page's rulings rather than from any text pattern, which makes it real
    evidence rather than the parser agreeing with itself.

    It cannot replace the line parser: find_tables() drops any row whose cell
    straddles a page break, which is the first and last row of most pages, so
    it sees 1,072 of the 1,166. But on those 1,072 it is authoritative.

    This check earned its place. Every other test in this file passed while
    three titles still carried an address fragment ('Singapore PhysOCT: ...',
    'Kingdon / Germany E-MRL: ...'), and this is what found them.
    """
    import re

    import pdfplumber
    from extract_program_2026 import _join_wrapped

    mine = {p["board_number"]: p for p in posters}
    rows = []
    with pdfplumber.open(PDF) as pdf:
        for pno in range(11, 106):
            for t in pdf.pages[pno - 1].find_tables():
                for r in t.extract():
                    if (
                        r
                        and len(r) >= 4
                        and r[0]
                        and re.fullmatch(r"\d+", (r[0] or "").strip())
                    ):
                        rows.append(r)

    assert len(rows) > 1000, f"only {len(rows)} table rows; check pdfplumber"

    mismatches = []
    checked = 0
    for r in rows:
        board = r[2].strip()
        if board not in mine:
            continue
        checked += 1
        cell = r[3] or ""
        title = _join_wrapped(
            re.split(r"\n(?=Authors?\s*:)", cell, maxsplit=1)[0].split("\n")
        )
        if title != mine[board]["title"]:
            mismatches.append((board, title, mine[board]["title"]))

    assert not mismatches, (
        f"{len(mismatches)} of {checked} titles disagree with the ruled table, "
        f"first: {mismatches[0][0]}\n  table: {mismatches[0][1][:90]!r}\n"
        f"  parser: {mismatches[0][2][:90]!r}"
    )


def test_committed_json_used_the_name_lookup(posters):
    """The committed file is built with the canonical names from the existing
    five years, so compound surnames that no rule can guess come out right.

    Checked against the committed artifact rather than by re-parsing the PDF
    with a lookup, which cost 11 seconds for the same assurance.
    """
    import json

    out = Path("data/raw/miccai_2026_program.json")
    if not out.exists():
        pytest.skip("run analysis/extract_program_2026.py first")
    built = {r["board_number"]: r for r in json.loads(out.read_text())}
    assert len(built) == len(posters)

    rule_only = {p["board_number"]: p for p in posters}
    differing = [
        (b, rule_only[b]["authors"], built[b]["authors"])
        for b in built
        if built[b]["authors"] != rule_only[b]["authors"]
    ]
    # The lookup only overrides where the particle rule gets a compound
    # surname wrong, which is rare: 17 of 7,359 author slots.
    assert differing, "the committed file shows no sign of the name lookup"
    flat_built = {a for _, _, names in differing for a in names}
    assert "Ben Bashat, Dafna" in flat_built or "De Sousa Ribeiro, Fabio" in flat_built


# ── Talk sessions (pages 2-10) ──────────────────────────────────────────


def test_talk_counts_match_the_facts_file(talks):
    """These are the counts MICCAI announced. The parser agreeing with them is
    the whole safety net for the session reader, which is the least certain
    code in this pipeline."""
    orals = [t for t in talks if t["presentation_type"] == "oral"]
    spots = [t for t in talks if t["presentation_type"] == "spotlight"]
    assert len(orals) == FACTS["n_orals"] == 99
    assert len(spots) == FACTS["n_spotlights"] == 54


def test_every_talk_has_a_session(talks):
    assert all(t["oral_session"].strip() for t in talks)


def test_eighteen_sessions(talks):
    """Six slots of three parallel sessions: O1A through O6C."""
    assert len({t["oral_session"] for t in talks}) == 18


def test_wrapped_session_name_is_joined(talks):
    """O6B's name is set over two bold lines, 'Multimodal Integration &
    Outcome' and 'Prediction'. Taking only the first line truncates it."""
    names = {t["oral_session"] for t in talks}
    assert "Multimodal Integration & Outcome Prediction" in names
    assert "Image Registration & Computational Anatomy" in names


def test_no_page_furniture_became_a_talk(talks):
    """The date banner spans all three columns so each column sees a fragment,
    and page 2 carries a 'Revised' stamp at session-name size."""
    for t in talks:
        assert "Oral Session" not in t["title"]
        assert "Session Chairs" not in t["title"]
        assert "Revised" not in t["title"]
        assert "MICCAI 2026" not in t["title"]


def test_spotlights_only_in_combined_sessions(talks):
    """Sessions 2, 3 and 5 are headed 'Oral Session' and hold no spotlights;
    1, 4 and 6 are 'Oral & Spotlight Session' and hold both."""
    spot_sessions = {
        t["oral_session"] for t in talks if t["presentation_type"] == "spotlight"
    }
    oral_only = {"Computer-Aided Diagnosis II", "Computational Pathology"}
    assert not (spot_sessions & oral_only)


# ── The join (talks onto papers) ────────────────────────────────────────


def test_build_assigns_a_type_to_every_paper(records):
    assert len(records) == 1166
    counts = {}
    for r in records:
        counts[r["presentation_type"]] = counts.get(r["presentation_type"], 0) + 1
    assert counts == {
        "poster": 1166 - FACTS["n_orals"] - FACTS["n_spotlights"],
        "oral": FACTS["n_orals"],
        "spotlight": FACTS["n_spotlights"],
    }


def test_posters_carry_no_session(records):
    assert all(
        not r["oral_session"]
        for r in records
        if r["presentation_type"] == "poster"
    )


def test_talks_carry_a_session(records):
    assert all(
        r["oral_session"]
        for r in records
        if r["presentation_type"] in ("oral", "spotlight")
    )


def test_a_mismatch_in_counts_is_fatal(posters, talks):
    """A dropped talk must stop the build, not warn. Silently losing one
    biases every presentation-type number with nothing looking wrong."""
    import copy

    wrong = dict(FACTS)
    wrong["n_orals"] = FACTS["n_orals"] + 1
    with pytest.raises(SystemExit, match="disagree with the program"):
        join_types(copy.deepcopy(posters), talks, wrong)
