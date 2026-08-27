"""Conformance: proxide's alphabet constants match the ecosystem's declarations.

WHY PROXIDE MATTERS MOST HERE

proxide is the origin. `chem/residues.py:623`'s `restypes` is the AlphaFold-derived ordering
that proteinsmc copied verbatim, and `chem/conversion.py:16-17` is what aminx duplicated
byte-for-byte. An ecosystem census found three base orderings under five names across four
repos, and a silent data-corruption bug that shipped from exactly that confusion.

proxide also has **three** alphabet declaration sites, not the two an earlier census recorded:
`chem/conversion.py`, `io/parsing/mappings.py`, and `chem/residues.py`. The third is the one
the others copy from, and it was missed because the first census regexed string literals and
`restypes` is a list.

Phase 0 (decision D4) scoped `alphex` to **dev-dependency-only**: nothing under `src/` imported
the library, and these tests only checked proxide's hand-rolled tables against it. That scoping
is superseded for proxide specifically — see
`.praxia/docs/decisions/260827_alphex-runtime-dependency-d4-superseded.md`. `alphex` is now a
**runtime** dependency: `chem/conversion.py` and `io/parsing/mappings.py` import it directly at
module scope and delegate their `af_to_mpnn`/`mpnn_to_af` permutation-table construction to
`alphex.perm`, rather than building those tables from local literals. Consequently, the tests in
this file no longer gate *whether* delegation should happen (that decision is made); they
validate the *result* of that delegation — that the alphabet constants proxide exposes still
match the ecosystem's canonical declarations now that they're derived through `alphex` rather
than hand-rolled.

If a test here fails, do NOT edit it to match. One of the two sides is wrong, and which one is
the finding.
"""

from __future__ import annotations

import pytest

alphex = pytest.importorskip(
  "alphex",
  reason="alphabet conformance library not installed; add it to the dev group",
)
known = alphex.known
SpecialKind = alphex.SpecialKind


def test_restypes_is_the_alphafold_ordering() -> None:
  """The origin declaration. proteinsmc's `restypes` is a copy of this one."""
  from proxide.chem.residues import restypes

  assert "".join(restypes) == known.AF_20.symbols


def test_conversion_declares_both_base_orderings_with_x_at_20() -> None:
  from proxide.chem.conversion import AF_ALPHABET, MPNN_ALPHABET

  assert MPNN_ALPHABET[:20] == known.MPNN_X_21.symbols
  assert AF_ALPHABET[:20] == known.AF_X_21.symbols
  assert MPNN_ALPHABET[20] == AF_ALPHABET[20] == "X"
  assert known.MPNN_X_21.specials[SpecialKind.UNKNOWN] == 20
  assert known.AF_X_21.specials[SpecialKind.UNKNOWN] == 20


def test_the_second_internal_declaration_agrees_with_the_first() -> None:
  """proxide declares the same pair twice, in two modules.

  Finding F3 of the census. If these ever diverge, one copy has been edited in isolation --
  which is the failure mode a single declaration exists to prevent. Until they are unified,
  this test is what keeps the duplication harmless.
  """
  from proxide.chem import conversion
  from proxide.io.parsing import mappings

  assert conversion.MPNN_ALPHABET == mappings.MPNN_ALPHABET
  assert conversion.AF_ALPHABET == mappings.AF_ALPHABET


def test_hhblits_value_ordering_is_proteinmpnn_with_x_then_gap() -> None:
  """`ID_TO_HHBLITS_AA` in id order is ProteinMPNN-ordered with X at 20 and gap at 21.

  Note this is the OPPOSITE sentinel order to proteinsmc's 22-wide space, which puts gap at 20
  and stop/unknown at 21. Two q=22 alphabets whose sentinels are swapped is an easy and silent
  mistake, so both are declared separately and pinned here.
  """
  from proxide.chem.residues import ID_TO_HHBLITS_AA

  in_id_order = "".join(ID_TO_HHBLITS_AA[i] for i in range(len(ID_TO_HHBLITS_AA)))
  assert in_id_order[:20] == known.MPNN_X_GAP_22.symbols
  assert in_id_order[20] == "X"
  assert in_id_order[21] == "-"
  assert known.MPNN_X_GAP_22.specials[SpecialKind.UNKNOWN] == 20
  assert known.MPNN_X_GAP_22.specials[SpecialKind.GAP] == 21


def test_hhblits_handles_degenerate_and_nonstandard_symbols() -> None:
  """Pins the fact that voided two of the library's refusals.

  The library's abstraction ceiling refused to model degenerate symbols (B, Z, J) and
  nonstandard residues (U, O) on the grounds that the ecosystem had zero instances. proxide has
  had them all along, here, with a stated resolution policy: B and Z fold onto D and E, U onto
  C, and J and O onto X. Asserting it means the count can never silently return to zero.

  These mappings are many-to-one, so the library cannot express them until `aliases` lands
  (v0.2). Until then this site is knowingly outside the system, which is why it is pinned.
  """
  from proxide.chem.residues import HHBLITS_AA_TO_ID, ID_TO_HHBLITS_AA

  assert ID_TO_HHBLITS_AA[HHBLITS_AA_TO_ID["B"]] == "D"
  assert ID_TO_HHBLITS_AA[HHBLITS_AA_TO_ID["Z"]] == "E"
  assert ID_TO_HHBLITS_AA[HHBLITS_AA_TO_ID["U"]] == "C"
  assert ID_TO_HHBLITS_AA[HHBLITS_AA_TO_ID["J"]] == "X"
  assert ID_TO_HHBLITS_AA[HHBLITS_AA_TO_ID["O"]] == "X"


def test_restypes_with_x_and_gap_is_alphafold_ordered() -> None:
  """The eighth alphabet, absent from the library because the AST census cannot see it.

  `restypes_with_x_and_gap = [*restypes, "X", "-"]` is built by a starred expression, so
  literal analysis misses it and the library deliberately ships no declaration for it rather
  than guessing. This test is the substitute: it pins the ordering in place so that when a
  declaration is finally added, there is something to check it against.
  """
  from proxide.chem.residues import restypes_with_x_and_gap

  assert "".join(restypes_with_x_and_gap[:20]) == known.AF_20.symbols
  assert restypes_with_x_and_gap[20] == "X"
  assert restypes_with_x_and_gap[21] == "-"


def test_af_mpnn_perm_tables_match_hand_rolled_literals_exhaustively() -> None:
  """Exhaustive 21-symbol round-trip parity gate for the alphex migration (Work Package 1).

  Work Package 1 (260827_alphex-ecosystem-migration-plan) replaces `chem/conversion.py`'s and
  `io/parsing/mappings.py`'s hand-rolled `MPNN_ALPHABET`/`AF_ALPHABET` literals and the
  `[...].index(k) for k in ...]`-built permutation tables with values derived from
  `alphex.known` and `alphex.perm`.

  "Expected" is deliberately hardcoded to the pre-migration literal strings rather than read
  off `conversion.AF_ALPHABET`/`mappings.AF_ALPHABET` at test time: those module attributes
  are exactly what the migration replaces with alphex-derived values, so deriving "expected"
  from them would make this test check the migrated implementation against itself
  (tautological) once the migration lands, rather than against the ordering it must preserve.
  This is what keeps this test a real regression gate for steps 3-4 of the migration, as
  distinct from `test_conversion_declares_both_base_orderings_with_x_at_20` above, which pins
  the module attribute against `alphex`'s declaration and necessarily becomes tautological
  itself once that attribute is constructed from `alphex.known` directly.

  Run against the unmodified hand-rolled implementation, this passes because it is then
  checking each literal against itself -- establishing the pre-migration baseline this test
  guards.
  """
  import numpy as np

  from proxide.chem import conversion
  from proxide.io.parsing import mappings

  # Hand-rolled literals exactly as they read immediately before this migration. Hardcoded
  # (not imported) so that migrating conversion.py/mappings.py to derive these from `alphex`
  # cannot silently make this assertion vacuous.
  hand_rolled_mpnn = "ACDEFGHIKLMNPQRSTVWYX"
  hand_rolled_af = "ARNDCQEGHILKMFPSTWYVX"

  expected_af_to_mpnn = [hand_rolled_mpnn.index(k) for k in hand_rolled_af]
  expected_mpnn_to_af = [hand_rolled_af.index(k) for k in hand_rolled_mpnn]

  for site in (conversion, mappings):
    # The module's public alphabet strings must still equal the pre-migration literals,
    # whether they are still literals (pre-migration) or now derived from `alphex.known`
    # (post-migration) -- either way, byte-for-byte, or the migration changed the ordering.
    assert site.AF_ALPHABET == hand_rolled_af
    assert site.MPNN_ALPHABET == hand_rolled_mpnn

    got_af_to_mpnn = np.asarray(site.af_to_mpnn(np.arange(21)))
    got_mpnn_to_af = np.asarray(site.mpnn_to_af(np.arange(21)))

    np.testing.assert_array_equal(got_af_to_mpnn, expected_af_to_mpnn)
    np.testing.assert_array_equal(got_mpnn_to_af, expected_mpnn_to_af)
