#!/usr/bin/env python3
"""
Join extracted oral/spotlight presentations to proceedings papers by title
→ data/processed/orals.json

Input:  data/raw/orals_YYYY.json      (analysis/extract_orals.py)
        data/processed/miccai_all.json
        data/oral_title_overrides.yaml   (optional, hand-edited)

The program book and the proceedings site are two independent publications of
the same paper, so titles agree most of the time but not always: camera-ready
retitling, typography (en dashes, ligatures, smart quotes) and line-wrap
hyphenation all introduce differences. Matching therefore runs as a cascade,
most-certain first, and anything it cannot settle is reported rather than
guessed:

  1. exact   ; identical after normalization
  2. fuzzy   ; similarity >= FUZZY_ACCEPT and clearly ahead of the runner-up
  3. speaker ; similarity >= FUZZY_CONSIDER and the presenting speaker's
                surname appears in the candidate's author list
  4. override; listed by hand in data/oral_title_overrides.yaml
                (a null value marks a deliberate non-paper, e.g. an invited talk)

Anything left unmatched is a hard error. Silently dropping a presentation would
bias every downstream statistic toward whatever kind of title fails to parse,
which is exactly the sort of invisible error this analysis cannot afford.

Usage:
    python analysis/match_orals.py
    python analysis/match_orals.py --report   # write unmatched to a YAML stub
"""

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import yaml
from rapidfuzz import fuzz, process

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
ALL_JSON = Path("data/processed/miccai_all.json")
OUT_JSON = Path("data/processed/orals.json")
OVERRIDES = Path("data/oral_title_overrides.yaml")

FUZZY_ACCEPT = 92.0     # accept outright at or above this similarity
FUZZY_CONSIDER = 82.0   # below this, do not even consider a speaker tie-break
FUZZY_MARGIN = 3.0      # best must beat runner-up by this much to be unique


def normalize(title: str) -> str:
    """Fold a title to a comparable form: accents, case and punctuation removed."""
    s = unicodedata.normalize("NFKD", title)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"[‐-―]", "-", s)  # all dash flavours → hyphen
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _speaker_surname(speaker: str | None) -> str | None:
    """'Lei Ma, Tongji University, China' → 'ma' (name precedes the affiliation)."""
    if not speaker:
        return None
    name = speaker.split(",")[0].strip()
    parts = [p for p in name.split() if p]
    if not parts:
        return None
    return normalize(parts[-1]) or None


def _author_surnames(paper: dict) -> set:
    """Proceedings authors are stored 'Last, First'."""
    out = set()
    for a in paper.get("authors", []):
        surname = a.split(",")[0].strip()
        n = normalize(surname)
        if n:
            out.add(n)
    return out


def load_papers() -> dict:
    with open(ALL_JSON, encoding="utf-8") as f:
        papers = json.load(f)
    by_year: dict = {}
    for p in papers:
        by_year.setdefault(p["year"], []).append(p)
    return by_year


def load_overrides() -> dict:
    if not OVERRIDES.exists():
        return {}
    with open(OVERRIDES, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Normalize the keys so the file can be written the way the PDF prints it.
    return {
        int(yr): {normalize(k): v for k, v in (block or {}).items()}
        for yr, block in data.items()
        if str(yr).isdigit()
    }


def match_year(year: int, presentations: list, papers: list, overrides: dict):
    """Return (matches, unmatched, skipped) for one year."""
    norm_to_papers: dict = {}
    for p in papers:
        norm_to_papers.setdefault(normalize(p["title"]), []).append(p)
    choices = list(norm_to_papers)

    matches: dict = {}
    unmatched: list = []
    skipped: list = []
    ov = overrides.get(year, {})

    for pres in presentations:
        raw = pres["title_raw"]
        nt = normalize(raw)

        # -- 0. hand-written override ------------------------------------
        if nt in ov:
            target = ov[nt]
            if target is None:
                skipped.append(raw)
                continue
            matches[target] = dict(pres, match="override")
            continue

        # -- 1. exact ----------------------------------------------------
        if nt in norm_to_papers and len(norm_to_papers[nt]) == 1:
            matches[norm_to_papers[nt][0]["paper_id"]] = dict(pres, match="exact")
            continue

        # -- 2/3. fuzzy, then speaker tie-break --------------------------
        ranked = process.extract(nt, choices, scorer=fuzz.ratio, limit=3)
        if not ranked:
            unmatched.append((raw, None, 0.0))
            continue
        best_key, best_score, _ = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_score >= FUZZY_ACCEPT and (best_score - second) >= FUZZY_MARGIN:
            cands = norm_to_papers[best_key]
            if len(cands) == 1:
                matches[cands[0]["paper_id"]] = dict(pres, match="fuzzy")
                continue

        if best_score >= FUZZY_CONSIDER:
            surname = _speaker_surname(pres.get("speaker"))
            resolved = None
            if surname:
                for key, score, _ in ranked:
                    if score < FUZZY_CONSIDER:
                        continue
                    for cand in norm_to_papers[key]:
                        if surname in _author_surnames(cand):
                            resolved = cand
                            break
                    if resolved:
                        break
            if resolved is not None:
                matches[resolved["paper_id"]] = dict(pres, match="speaker")
                continue

        unmatched.append((raw, best_key, best_score))

    return matches, unmatched, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report",
        action="store_true",
        help="Write unmatched titles to data/oral_title_overrides.todo.yaml",
    )
    args = ap.parse_args()

    by_year = load_papers()
    overrides = load_overrides()

    raw_files = sorted(RAW_DIR.glob("orals_*.json"))
    if not raw_files:
        logger.error(f"No orals_*.json in {RAW_DIR}; run extract_orals.py first")
        return 1

    out: dict = {"generated_from": {}, "summary": {}, "papers": {}}
    all_unmatched: dict = {}
    ok = True

    for rf in raw_files:
        with open(rf, encoding="utf-8") as f:
            data = json.load(f)
        year = data["year"]
        papers = by_year.get(year, [])
        if not papers:
            logger.error(f"{year}: no papers in {ALL_JSON}")
            return 1

        presentations = [
            dict(
                p,
                session_id=s["session_id"],
                session_title=s["session_title"],
                day=s["day"],
            )
            for s in data["sessions"]
            for p in s["presentations"]
        ]

        matches, unmatched, skipped = match_year(
            year, presentations, papers, overrides
        )

        ptypes: dict = {}
        methods: dict = {}
        for pid, m in matches.items():
            ptypes[m["type"]] = ptypes.get(m["type"], 0) + 1
            methods[m["match"]] = methods.get(m["match"], 0) + 1
            out["papers"][pid] = {
                "type": m["type"],
                "session_id": m["session_id"],
                "session_title": m["session_title"],
                "day": m["day"],
                "match": m["match"],
            }

        n_extracted = len(presentations)
        n_matched = len(matches)
        rate = 100.0 * n_matched / max(n_extracted - len(skipped), 1)
        out["generated_from"][str(year)] = str(rf)
        out["summary"][str(year)] = {
            "extracted": n_extracted,
            "matched": n_matched,
            "skipped": len(skipped),
            "unmatched": len(unmatched),
            "match_rate_pct": round(rate, 1),
            "types": ptypes,
            "methods": methods,
        }

        level = logger.info if not unmatched else logger.warning
        level(
            f"{year}: {n_matched}/{n_extracted} matched ({rate:.1f}%) "
            f"{ptypes} via {methods}"
            + (f"; {len(unmatched)} UNMATCHED" if unmatched else "")
        )
        if unmatched:
            ok = False
            all_unmatched[year] = unmatched
            for raw, best, score in unmatched:
                logger.warning(f"    [{score:5.1f}] {raw[:70]}")
                if best:
                    logger.warning(f"             best guess: {best[:70]}")

    if all_unmatched and args.report:
        stub = Path("data/oral_title_overrides.todo.yaml")
        lines = [
            "# Unmatched program titles. Map each to a paper_id, or to null",
            "# if it is deliberately not a proceedings paper (e.g. invited talk).",
            "# Then merge the entries into data/oral_title_overrides.yaml.",
            "",
        ]
        for year, items in sorted(all_unmatched.items()):
            lines.append(f"{year}:")
            for raw, best, score in items:
                lines.append(f"  # closest match ({score:.1f}): {best}")
                lines.append(f"  {json.dumps(raw, ensure_ascii=False)}: null")
            lines.append("")
        stub.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Wrote unmatched stub → {stub}")

    if not ok:
        logger.error(
            "Unresolved presentations remain. Add them to "
            f"{OVERRIDES} (run again with --report to generate a stub)."
        )
        return 1

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    total = len(out["papers"])
    logger.info(f"Wrote {total} matched presentations → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
