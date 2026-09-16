"""Turning a program booklet's "First Last" into the site's "Last, First".

Splitting on the last space is the obvious rule and it mangles 204 of the
12,307 author names already in the data, including "Abou El-Ghar, Mohamed".
Looking a name up in the existing data fixes those; only one string in 12,307
is genuinely ambiguous.

These tests cover name FORMATTING only. Deciding whether two identical names
are the same person is not possible from this data and is not attempted.
"""

import sys

sys.path.insert(0, "analysis")
from author_names import build_lookup, to_last_first  # noqa: E402


def test_simple_two_word_name():
    assert to_last_first("Yi Yu", {}) == ("Yu, Yi", "particle")


def test_two_word_given_name():
    """'Won Hwa Kim' must not become 'Hwa Kim, Won'."""
    assert to_last_first("Won Hwa Kim", {}) == ("Kim, Won Hwa", "particle")
    assert to_last_first("S. Kevin Zhou", {}) == ("Zhou, S. Kevin", "particle")


def test_particle_surname():
    """The naive last-space rule gives 'Bruijne, Marleen de', which is wrong."""
    assert to_last_first("Marleen de Bruijne", {}) == (
        "de Bruijne, Marleen",
        "particle",
    )
    assert to_last_first("Jan van der Meer", {}) == (
        "van der Meer, Jan",
        "particle",
    )


def test_lookup_beats_the_rule():
    """'Abou El-Ghar' is a two-word surname no rule can guess, but the
    existing data already records which tokens are the surname."""
    lookup = build_lookup(
        [{"authors": ["Abou El-Ghar, Mohamed", "Shen, Dinggang"]}]
    )
    assert to_last_first("Mohamed Abou El-Ghar", lookup) == (
        "Abou El-Ghar, Mohamed",
        "lookup",
    )
    assert to_last_first("Dinggang Shen", lookup) == ("Shen, Dinggang", "lookup")


def test_single_token_name():
    assert to_last_first("Madonna", {}) == ("Madonna", "single")


def test_empty_name():
    assert to_last_first("   ", {}) == ("", "empty")


def test_lookup_drops_ambiguous_keys():
    """If one 'First Last' maps to two different canonical names the lookup
    must not pick one; it drops the key and lets the rule decide. This is the
    single real collision in the existing 12,307 names."""
    lookup = build_lookup(
        [{"authors": ["M. Shama, Deeksha", "Shama, Deeksha M."]}]
    )
    assert "deeksha m. shama" not in lookup


def test_lookup_is_case_insensitive():
    lookup = build_lookup([{"authors": ["Shen, Dinggang"]}])
    assert to_last_first("DINGGANG SHEN", lookup) == ("Shen, Dinggang", "lookup")


def test_whitespace_is_collapsed():
    assert to_last_first("  Yi   Yu  ", {}) == ("Yu, Yi", "particle")


def test_lookup_ignores_malformed_entries():
    """Entries with no comma, or an empty half, cannot say which token is the
    surname, so they must not become lookup keys."""
    lookup = build_lookup([{"authors": ["Madonna", "Shen,", ", Dinggang"]}])
    assert lookup == {}
