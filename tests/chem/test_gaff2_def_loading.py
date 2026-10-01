"""Tests for ATOMTYPE_GFF2.DEF loading, pinning, and fail-loud error paths.

Deliberately does NOT import rdkit and carries no skip markers: everything
here exercises `gaff2.py`'s pin/path/digest/parse logic, none of which
needs RDKit. Per this repo's SECURITY RULE, the real ATOMTYPE_GFF2.DEF is
never fetched, stubbed, or fabricated here -- every test either points
`PROXIDE_GAFF2_DEF`/`def_path` at a synthetic tmp file, or points it at a
deliberately nonexistent path.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from proxide.chem import gaff2
from proxide.chem.gaff2 import (
    Gaff2DefInvalidError,
    Gaff2DefMissingError,
    load_gaff2_rules,
)

_REPO_ROOT = Path(__file__).parent.parent.parent
_PIN_PATH = _REPO_ROOT / "src" / "proxide" / "assets" / "gaff" / "ATOMTYPE_GFF2.pin.toml"
_CI_YML_PATH = _REPO_ROOT / ".github" / "workflows" / "ci.yml"

# One syntactically valid ATD row (same grammar as
# tests/test_gaff2_golden.py's TestGaff2Grammar / rules_loader.rs's
# fake_parse_ok fixtures) -- NOT real DEF content, just enough for
# parse_gaff2_rules() to recognize a single rule if it ever got that far.
_ONE_VALID_ATD_LINE = "Definition begin.\nATD c3 * 6 4 *&\nDefinition end.\n"


@pytest.fixture(autouse=True)
def _reset_default_rules_cache():
    """Reset gaff2's module-level default-rules cache before and after each test.

    Without this, one test's successful (or failed-but-cached, which
    should never happen) default load would leak into the next test via
    the module-global `_default_rules`/`_default_wildatom` cache.
    """
    gaff2._default_rules = None
    gaff2._default_wildatom = None
    yield
    gaff2._default_rules = None
    gaff2._default_wildatom = None


def test_missing_default_def_raises_missing_error(tmp_path, monkeypatch) -> None:
    nonexistent = tmp_path / "does-not-exist" / "ATOMTYPE_GFF2.DEF"
    monkeypatch.setenv("PROXIDE_GAFF2_DEF", str(nonexistent))

    with pytest.raises(Gaff2DefMissingError) as excinfo:
        load_gaff2_rules()

    err = excinfo.value
    assert isinstance(err, FileNotFoundError)
    message = str(err)
    assert str(nonexistent) in message
    assert "PROXIDE_GAFF2_DEF" in message
    assert "fetch_amber_assets" in message

    pin = tomllib.loads(_PIN_PATH.read_text())
    assert pin["sha256"] in message


def test_digest_mismatch_on_default_path_raises_invalid_error(tmp_path, monkeypatch) -> None:
    altered = tmp_path / "ATOMTYPE_GFF2.DEF"
    altered.write_text(_ONE_VALID_ATD_LINE)
    monkeypatch.setenv("PROXIDE_GAFF2_DEF", str(altered))

    pin = tomllib.loads(_PIN_PATH.read_text())
    import hashlib

    actual_digest = hashlib.sha256(altered.read_bytes()).hexdigest()
    assert actual_digest != pin["sha256"]  # sanity: our synthetic file really differs

    with pytest.raises(Gaff2DefInvalidError) as excinfo:
        load_gaff2_rules()

    message = str(excinfo.value)
    assert pin["sha256"] in message
    assert actual_digest in message


def test_explicit_empty_def_path_raises_invalid_error(tmp_path) -> None:
    empty = tmp_path / "empty.DEF"
    empty.write_text("")

    with pytest.raises(Gaff2DefInvalidError):
        load_gaff2_rules(def_path=empty)


def test_explicit_missing_def_path_raises_missing_error(tmp_path) -> None:
    missing = tmp_path / "nope.DEF"

    with pytest.raises(Gaff2DefMissingError):
        load_gaff2_rules(def_path=missing)


def test_failure_is_not_cached(tmp_path, monkeypatch) -> None:
    nonexistent = tmp_path / "does-not-exist" / "ATOMTYPE_GFF2.DEF"
    monkeypatch.setenv("PROXIDE_GAFF2_DEF", str(nonexistent))

    with pytest.raises(Gaff2DefMissingError):
        load_gaff2_rules()

    # A raising call must never populate the module cache.
    assert gaff2._default_rules is None
    assert gaff2._default_wildatom is None

    # And a second call must raise again, not silently return an empty ruleset.
    with pytest.raises(Gaff2DefMissingError):
        load_gaff2_rules()

    assert gaff2._default_rules is None
    assert gaff2._default_wildatom is None


def test_pin_file_parses_and_has_required_keys() -> None:
    pin = tomllib.loads(_PIN_PATH.read_text())

    for key in ("repo", "ref", "upstream_path", "url", "sha256", "license_note"):
        assert key in pin, f"pin file missing required key: {key}"
        assert isinstance(pin[key], str) and pin[key], f"pin key {key!r} is empty"

    sha256 = pin["sha256"]
    assert len(sha256) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", sha256), f"sha256 is not 64 lowercase hex chars: {sha256!r}"


def test_ci_yml_literals_match_pin() -> None:
    """ci.yml's rust-checks job hardcodes ref + sha256 (see that job's own
    comment on why it can't just call fetch_amber_assets.py) -- those
    literals must never silently drift from the pin file.
    """
    pin = tomllib.loads(_PIN_PATH.read_text())
    ci_text = _CI_YML_PATH.read_text()

    curl_match = re.search(
        r"Amber-MD/AmberClassic/([0-9a-f]{40})/dat/antechamber/ATOMTYPE_GFF2\.DEF",
        ci_text,
    )
    assert curl_match, "could not find the AmberClassic curl URL in ci.yml"
    assert curl_match.group(1) == pin["ref"]

    sha_match = re.search(r"([0-9a-f]{64})\s+src/proxide/assets/gaff/dat/ATOMTYPE_GFF2\.DEF", ci_text)
    assert sha_match, "could not find the sha256sum -c line in ci.yml"
    assert sha_match.group(1) == pin["sha256"]


# Debt #2361: a malformed ATD line used to vanish silently while the rest of
# the file loaded. Synthetic content only (see module docstring).
_DEF_HEADER = "Defination begin\n------------------\n"


@pytest.mark.parametrize(
    "bad_line",
    [
        "ATD  c3    *   6   4",  # no terminating "&"
        "ATD  c3    *   &",  # fewer than 3 fields
        "ATD  c3    &",  # one token, but not the DU catch-all
        "ATD  bogus *   NOTANUM 4 &",  # atomic number does not parse
    ],
)
def test_malformed_atd_line_is_an_error(tmp_path, bad_line):
    def_path = tmp_path / "bad.DEF"
    def_path.write_text(f"{_DEF_HEADER}ATD  c3 * 6 4 &\n{bad_line}\n")
    with pytest.raises(Gaff2DefInvalidError, match="1 malformed ATD rule line"):
        gaff2.parse_gaff2_rules(def_path)


def test_du_catch_all_is_the_one_deliberate_skip(tmp_path):
    def_path = tmp_path / "du.DEF"
    def_path.write_text(f"{_DEF_HEADER}ATD  c3 * 6 4 &\nATD  DU    &\n")
    rules, _ = gaff2.parse_gaff2_rules(def_path)
    assert [r.atom_type for r in rules] == ["c3"]


# Debt #2363: malformed f8/f9 used to be silently dropped. Now they raise.


@pytest.mark.parametrize(
    "bad_f8",
    [
        "[unclosed",  # f8 with unclosed "["
        "[invalid-token]",  # f8 with non-matching token (invalid-token doesn't match pattern)
        "[2]",  # f8 with bare count (count without word)
    ],
)
def test_malformed_f8_bracket_is_an_error(tmp_path, bad_f8):
    """Test that malformed f8 bracket bodies raise Gaff2DefInvalidError (debt #2363)."""
    def_path = tmp_path / "bad_f8.DEF"
    # Insert f8 before f9 (simulated by putting it as 5th+ field): "ATD c3 * 6 4 * bad_f8"
    def_path.write_text(f"{_DEF_HEADER}ATD  c3 * 6 4 * {bad_f8} &\n")
    with pytest.raises(Gaff2DefInvalidError, match="1 malformed ATD rule line"):
        gaff2.parse_gaff2_rules(def_path)


@pytest.mark.parametrize(
    "bad_f9",
    [
        "(unclosed",  # f9 with unmatched opening "("
        "(N3)extra",  # f9 with trailing text after closing ")"
        "(XX[AR1,junk!])",  # bad token inside a neighbour's [...] (Python/Rust parity)
        "(N3(N3))",  # properly balanced nested pattern (should parse)
    ],
)
def test_malformed_f9_pattern_is_an_error(tmp_path, bad_f9):
    """Test that malformed f9 patterns raise Gaff2DefInvalidError (debt #2363)."""
    def_path = tmp_path / "bad_f9.DEF"
    # Insert f9 after f8: "ATD c3 * 6 4 * * bad_f9"
    def_path.write_text(f"{_DEF_HEADER}ATD  c3 * 6 4 * * {bad_f9} &\n")
    if bad_f9 == "(N3(N3))":
        # This one should actually parse fine
        rules, _ = gaff2.parse_gaff2_rules(def_path)
        assert len(rules) == 1
    else:
        with pytest.raises(Gaff2DefInvalidError, match="1 malformed ATD rule line"):
            gaff2.parse_gaff2_rules(def_path)


def test_text_after_f9_is_malformed(tmp_path):
    """Test that non-parenthesized text after f8 is flagged as malformed (debt #2363)."""
    def_path = tmp_path / "bad_text.DEF"
    # "JUNK" is not a valid f9 pattern (doesn't start with "(" or "*")
    def_path.write_text(f"{_DEF_HEADER}ATD  c3 * 6 4 * * JUNK &\n")
    with pytest.raises(Gaff2DefInvalidError, match="1 malformed ATD rule line"):
        gaff2.parse_gaff2_rules(def_path)
