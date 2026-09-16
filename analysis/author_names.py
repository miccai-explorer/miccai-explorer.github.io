#!/usr/bin/env python3
"""Turn a program booklet's "First Last" into the site's "Last, First".

This module formats names. It does NOT identify people, and nothing here can.
Two researchers who share a name are indistinguishable in this data: there is
no ORCID, no email, and no affiliation for anyone but the presenting author.
Measured on the existing five years, 38 of the 40 authors who reach the 2025
co-authorship network carry a two-word name, which is the shape most prone to
collision, so this is not a rare edge case. The site's author counts are string
matches and the year pages say so.
"""

# Surname particles, so "Marleen de Bruijne" keeps "de Bruijne" together.
# Splitting on the last space instead mangles 204 of the 12,307 author names
# already in the data.
PARTICLES = {
    "van",
    "von",
    "de",
    "der",
    "den",
    "di",
    "da",
    "das",
    "dos",
    "del",
    "della",
    "la",
    "le",
    "du",
    "bin",
    "ibn",
    "al",
    "el",
    "ter",
    "ten",
}


def build_lookup(papers: list) -> dict:
    """Canonical names from existing data: "first last" -> "Last, First".

    Many 2026 authors have published at MICCAI before, and their existing
    entry already records which tokens are the surname, which is exactly what
    no rule can reliably guess. A key that maps to two different canonical
    names is dropped rather than guessed at.
    """
    seen: dict = {}
    clashing: set = set()
    for p in papers:
        for a in p.get("authors", []):
            if a.count(",") != 1:
                continue
            last, first = (x.strip() for x in a.split(","))
            if not last or not first:
                continue
            key = f"{first} {last}".casefold()
            if key in seen and seen[key] != a:
                clashing.add(key)
            seen[key] = a
    for k in clashing:
        seen.pop(k, None)
    return seen


def to_last_first(name: str, lookup: dict) -> tuple:
    """Return (canonical name, which rule produced it).

    Most certain first, the same shape as the cascade in match_orals.py: look
    the name up in what is already known, then fall back to a particle-aware
    split. The stage is returned so a caller can report how much of its input
    rested on the fallback rather than on evidence.
    """
    n = " ".join(name.split())
    if not n:
        return "", "empty"

    hit = lookup.get(n.casefold())
    if hit:
        return hit, "lookup"

    tokens = n.split()
    if len(tokens) == 1:
        return tokens[0], "single"

    # Walk left from the last token while the token before it is a particle,
    # so "Jan van der Meer" yields the surname "van der Meer". Stop at index 1
    # so there is always something left to be the given name.
    i = len(tokens) - 1
    while i > 1 and tokens[i - 1].casefold().strip(".") in PARTICLES:
        i -= 1
    return f"{' '.join(tokens[i:])}, {' '.join(tokens[:i])}", "particle"
