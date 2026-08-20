"""GAFF2 atom type rule parsing and assignment.

This module implements GAFF2 (General Amber Force Field 2) atom type assignment
without requiring AmberTools. It parses the ATOMTYPE_GFF2.DEF rules and applies them
to RDKit molecules.
"""

from __future__ import annotations

import math as _math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rdkit import Chem

try:
    from rdkit import Chem
except ImportError:
    Chem = None


@dataclass
class Gaff2WildAtomDef:
    """WILDATOM definition for pattern matching.

    WILDATOM defines symbol shortcuts for sets of atom types that can be
    used in rule matching (e.g., XX = C,N,O,S,P).
    """

    symbol: str
    elements: list[str]


# ---------------------------------------------------------------------------
# f8 (atomic-property bracket) grammar: "[sb,db,AR2]", "[AR1,1RG6]", "[2DL]", ...
#
# Per ATOMTYPE_GFF2.DEF's own footer ("Field descriptions" / "Specific symbols" /
# "Predefined words" / "Miscellaneous"): comma = AND, "." = OR (ring/aromaticity
# descriptions); a leading digit is an exact-count prefix (e.g. "2DL" = exactly 2
# delocalized bonds, "1RG6" = exactly 1 six-membered ring); uppercase bond words
# (SB/DB/TB/AB/DL) are exact bond-type identity, lowercase (sb/db/tb) are the
# inclusive unions the footer describes ("sb" includes aromatic-single + delocalized,
# "db" includes aromatic-double).
# ---------------------------------------------------------------------------

_PROP_TOKEN_RE = re.compile(r"^(\d*)([A-Za-z]+\d*)$")


@dataclass
class PropToken:
    word: str          # e.g. "RG3", "AR1", "SB", "sb", "NR", "DL"
    count: int | None  # leading digit prefix, e.g. 2 in "2DL"; None = presence-only


@dataclass
class AtomicPropExpr:
    op: str  # "AND" (comma) or "OR" (dot)
    tokens: list[PropToken]


def _tokenize_prop_list(raw: str) -> tuple[str, list[str]]:
    """Split a bracket body on its top-level separator, returning (op, raw_tokens)."""
    raw = raw.strip()
    if "." in raw:
        return "OR", [t.strip() for t in raw.split(".") if t.strip()]
    return "AND", [t.strip() for t in raw.split(",") if t.strip()]


def parse_atomic_prop(raw: str) -> AtomicPropExpr | None:
    """Parse an f8 bracket body (without the surrounding `[...]`) into an AST."""
    raw = raw.strip()
    if not raw or raw == "*":
        return None

    op, raw_tokens = _tokenize_prop_list(raw)
    tokens: list[PropToken] = []
    for raw_tok in raw_tokens:
        m = _PROP_TOKEN_RE.match(raw_tok)
        if not m:
            continue
        count_str, word = m.group(1), m.group(2)
        tokens.append(PropToken(word=word, count=int(count_str) if count_str else None))
    return AtomicPropExpr(op=op, tokens=tokens) if tokens else None


def atomic_prop_matches(expr: AtomicPropExpr | None, facts: AtomBondFacts) -> bool:
    """Evaluate a parsed f8 expression against one atom's precomputed bond/ring facts."""
    if expr is None:
        return True

    results = [_prop_token_matches(tok, facts) for tok in expr.tokens]
    return any(results) if expr.op == "OR" else all(results)


# ---------------------------------------------------------------------------
# f9 (chemical-environment) grammar: "(N3,N3,N3)", "(XX[AR1],XX[AR1],XX[AR1])",
# "(C3(C3))", "(XD3[sb',db])", ...
#
# Top-level comma-separated entries = AND of existential, pairwise-DISTINCT
# neighbor requirements. Each entry is an element-or-WILDATOM token, optionally
# followed by a digit (that neighbor's own attached-atom count), an optional
# `[...]` bracket (that neighbor's own f8-style properties -- a bond-type word
# suffixed `'`/`''` constrains the SPECIFIC edge to the predecessor, positively
# or negatively; an unsuffixed bond word constrains some other bond of that
# neighbor), and an optional nested `(...)` recursing one more hop out.
# ---------------------------------------------------------------------------


@dataclass
class NeighborSpec:
    elem_or_wild: str
    attached_count: int | None = None
    own_props: AtomicPropExpr | None = None
    edge_bond_reqs: list[tuple[str, bool]] | None = None  # (bond_word, must_have)
    nested: ChemEnvExpr | None = None


@dataclass
class ChemEnvExpr:
    neighbors: list[NeighborSpec]


class _TokenStream:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.pos = 0

    def peek(self) -> str:
        return self.raw[self.pos] if self.pos < len(self.raw) else ""

    def advance(self) -> str:
        ch = self.peek()
        self.pos += 1
        return ch

    def eof(self) -> bool:
        return self.pos >= len(self.raw)


def parse_chem_env(raw: str) -> ChemEnvExpr | None:
    """Parse an f9 pattern (including its outer parens) into an AST."""
    raw = raw.strip()
    if not raw or raw == "*":
        return None
    if not raw.startswith("("):
        return None

    stream = _TokenStream(raw)
    expr = _parse_paren_group(stream)
    return expr


def _parse_paren_group(stream: _TokenStream) -> ChemEnvExpr | None:
    if stream.peek() != "(":
        return None
    stream.advance()  # consume "("

    neighbors: list[NeighborSpec | None] = []
    current = []
    depth = 0
    while not stream.eof():
        ch = stream.peek()
        if ch == "(" or ch == "[":
            depth += 1 if ch == "(" else 0
            current.append(stream.advance())
            if ch == "[":
                # consume through matching "]" verbatim (no nesting of [] within [])
                while not stream.eof() and stream.peek() != "]":
                    current.append(stream.advance())
                if not stream.eof():
                    current.append(stream.advance())
            continue
        if ch == ")":
            if depth == 0:
                stream.advance()  # consume closing ")"
                break
            depth -= 1
            current.append(stream.advance())
            continue
        if ch == "," and depth == 0:
            neighbors.append(_parse_neighbor_spec("".join(current)))
            current = []
            stream.advance()
            continue
        current.append(stream.advance())

    if current:
        neighbors.append(_parse_neighbor_spec("".join(current)))

    resolved_neighbors = [n for n in neighbors if n is not None]
    return ChemEnvExpr(neighbors=resolved_neighbors) if resolved_neighbors else None


_NEIGHBOR_HEAD_RE = re.compile(r"^([A-Za-z]+)(\d*)")


def _parse_neighbor_spec(raw: str) -> NeighborSpec | None:
    raw = raw.strip()
    if not raw:
        return None

    m = _NEIGHBOR_HEAD_RE.match(raw)
    if not m:
        return None
    elem_or_wild = m.group(1)
    attached_count = int(m.group(2)) if m.group(2) else None
    rest = raw[m.end():]

    own_props: AtomicPropExpr | None = None
    edge_bond_reqs: list[tuple[str, bool]] = []
    nested: ChemEnvExpr | None = None

    if rest.startswith("["):
        close = rest.find("]")
        if close != -1:
            bracket_body = rest[1:close]
            rest = rest[close + 1:]
            plain_tokens = []
            for raw_tok in bracket_body.replace(".", ",").split(","):
                raw_tok = raw_tok.strip()
                if not raw_tok:
                    continue
                if raw_tok.endswith("''"):
                    edge_bond_reqs.append((raw_tok[:-2], False))
                elif raw_tok.endswith("'"):
                    edge_bond_reqs.append((raw_tok[:-1], True))
                else:
                    plain_tokens.append(raw_tok)
            if plain_tokens:
                own_props = parse_atomic_prop(",".join(plain_tokens))

    if rest.startswith("("):
        nested = _parse_paren_group(_TokenStream(rest))

    return NeighborSpec(
        elem_or_wild=elem_or_wild,
        attached_count=attached_count,
        own_props=own_props,
        edge_bond_reqs=edge_bond_reqs or None,
        nested=nested,
    )


def _resolve_wildatom(token: str, wildatom_map: dict[str, list[str]]) -> list[str]:
    return wildatom_map.get(token, [token])


def chem_env_matches(
    expr: ChemEnvExpr | None,
    atom,
    wildatom_map: dict[str, list[str]],
    predecessor=None,
) -> bool:
    """Check whether `atom`'s real neighbor graph satisfies a parsed f9 pattern.

    Existential, pairwise-distinct matching: each entry in `expr.neighbors` must
    bind to a different actual neighbor of `atom` (excluding `predecessor`
    only for the purpose of *counting this atom itself* -- predecessor loop-back as a
    neighbor candidate is permitted, matching how conjugated-ring patterns like
    `cc`'s `(C3(C3))` are meant to walk back through a ring).

    Candidates include H neighbors, not just heavy atoms: `hw`'s pattern
    `(O(H1))` is the only f9 pattern anywhere in ATOMTYPE_GFF2.DEF that
    targets H as an explicit element (confirmed by grep), and it needs to see
    H neighbors to match at all. This is safe for every other rule: no
    WILDATOM macro (XX/XA/XB/XC/XD) or plain-element spec anywhere else
    resolves to "H", so `_neighbor_matches`'s element gate
    (`cand.GetSymbol() not in resolved`) already rejects an H candidate for
    every spec except `hw`'s -- including H atoms in the candidate pool only
    adds (harmless, small) extra iteration for those other rules.
    """
    if expr is None:
        return True

    candidates = list(atom.GetNeighbors())
    return _match_neighbor_specs(expr.neighbors, candidates, atom, wildatom_map)


def _match_neighbor_specs(specs, candidates, predecessor, wildatom_map) -> bool:
    if not specs:
        return True
    spec, rest = specs[0], specs[1:]
    for i, cand in enumerate(candidates):
        if _neighbor_matches(spec, cand, predecessor, wildatom_map):
            remaining = candidates[:i] + candidates[i + 1:]
            if _match_neighbor_specs(rest, remaining, predecessor, wildatom_map):
                return True
    return False


def _neighbor_matches(
    spec: NeighborSpec, cand, predecessor, wildatom_map: dict[str, list[str]]
) -> bool:
    resolved = _resolve_wildatom(spec.elem_or_wild, wildatom_map)
    if cand.GetSymbol() not in resolved:
        return False

    if spec.attached_count is not None:
        if _attached_count(cand) != spec.attached_count:
            return False

    if spec.edge_bond_reqs:
        bond = cand.GetOwningMol().GetBondBetweenAtoms(cand.GetIdx(), predecessor.GetIdx())
        if bond is None:
            return False
        edge_facts = _bond_category_facts(bond)
        for bond_word, must_have in spec.edge_bond_reqs:
            has_it = edge_facts.get(bond_word, False)
            if has_it != must_have:
                return False

    if spec.own_props is not None:
        cand_facts = _atom_bond_facts(cand)
        if not atomic_prop_matches(spec.own_props, cand_facts):
            return False

    if spec.nested is not None:
        if not chem_env_matches(spec.nested, cand, wildatom_map, predecessor=cand):
            return False

    return True


# ---------------------------------------------------------------------------
# Per-atom bond/ring fact extraction shared by f8 and f9 matching.
# ---------------------------------------------------------------------------


@dataclass
class AtomBondFacts:
    in_ring: bool
    ring_counts_by_size: dict[int, int]
    aromaticity_class: str | None  # "AR1"/"AR2"/"AR3"/"AR4"/"AR5", best-effort (see note)
    bond_counts: dict[str, int]


def _attached_count(atom) -> int:
    """H-inclusive attached-atom count (matches DEF f5 convention)."""
    return atom.GetDegree() + atom.GetNumImplicitHs()


def _bond_category_facts(bond) -> dict[str, bool]:
    """Per-bond category flags, used both for tallying and for `'`/`''` edge checks.

    Requires `bond`'s owning molecule to have been Kekulized with
    `clearAromaticFlags=False` (see `assign_gaff2_atom_types`) -- otherwise every
    ring bond in an aromatic system reports `GetBondType() == AROMATIC` rather
    than its true Kekule SINGLE/DOUBLE identity, and the sb/db distinction below
    collapses to always-False for ring bonds.

    Per ATOMTYPE_GFF2.DEF's footer: SB/DB are *exact, non-aromatic* bond identity
    ("Single bond" / "Double bond", no qualifier); sb/db are the *inclusive*
    unions ("Single bond, including aromatic single..." / "Double bond, including
    aromatic double"). With Kekulization applied, `bt.name` already reflects the
    true per-bond Kekule single/double assignment regardless of aromaticity, so
    sb/db read directly off it; SB/DB additionally require `not is_aromatic`.

    DL (delocalized) detection convention: a bond is treated as delocalized iff it
    is aromatic. This is a deliberate, documented simplification (the DEF file's
    "9 in AM1-BCC" delocalized bond-order code has no equivalent in RDKit's bond
    model) -- it correctly covers ring-embedded delocalized systems (most aromatic
    heterocycles) but will NOT flag isolated, non-ring resonance systems (e.g. an
    acyclic amide or carboxylate) as delocalized. Those instead present as a plain
    Kekule double bond and match the DEF's "[1DB,0DL]" variant, which is the
    common case for isolated carbonyls.
    """
    is_aromatic = bond.GetIsAromatic()
    bt = bond.GetBondType()
    is_single = bt.name == "SINGLE"
    is_double = bt.name == "DOUBLE"
    is_triple = bt.name == "TRIPLE"
    if is_aromatic and bt.name not in ("SINGLE", "DOUBLE"):
        raise AssertionError(
            f"_bond_category_facts precondition violated: an aromatic bond "
            f"reports GetBondType()={bt.name!r}, not SINGLE/DOUBLE -- the "
            f"owning molecule was not Kekulized with clearAromaticFlags=False "
            f"before this call (see assign_gaff2_atom_types). sb/db would "
            f"silently collapse to always-False for this bond."
        )
    return {
        "SB": is_single and not is_aromatic,
        "DB": is_double and not is_aromatic,
        "TB": is_triple,
        "AB": is_aromatic,
        "DL": is_aromatic,
        "sb": is_single,
        "db": is_double,
        "tb": is_triple,
    }


def _ring_has_exocyclic_multibond(mol: Chem.Mol, ring: tuple[int, ...]) -> bool:
    """True if any atom in `ring` has a post-Kekulization DOUBLE/TRIPLE bond
    to a neighbor with NO ring membership at all (AmberTools' real AR3
    "planar rings formed 'outside' double bonds" test, ring.c -- its actual
    condition is the OTHER atom's total ring-membership count being zero,
    `arom[bond[j].bondj].rg[0] == 0`, not merely "not in this specific
    ring": a fused-ring bridgehead's neighbor in the OTHER ring of the fused
    system is still ring-connected and must NOT count as exocyclic, or
    naphthalene/anthracene's bridgeheads would be wrongly demoted). `mol`
    must already be Kekulized -- see `_atom_bond_facts`'s caller.
    """
    ring_info = mol.GetRingInfo()
    for ring_idx in ring:
        for bond in mol.GetAtomWithIdx(ring_idx).GetBonds():
            other = bond.GetOtherAtomIdx(ring_idx)
            if ring_info.NumAtomRings(other) > 0:
                continue
            if bond.GetBondType().name in ("DOUBLE", "TRIPLE"):
                return True
    return False


def _atom_bond_facts(atom) -> AtomBondFacts:
    mol = atom.GetOwningMol()
    ring_info = mol.GetRingInfo()
    idx = atom.GetIdx()

    ring_counts_by_size: dict[int, int] = {}
    in_ring = False
    for ring in ring_info.AtomRings():
        if idx in ring:
            in_ring = True
            ring_counts_by_size[len(ring)] = ring_counts_by_size.get(len(ring), 0) + 1

    # AR1-AR5 classification: the DEF footer's prose ("AR1 Pure aromatic atom
    # (such as benzene and pyridine)"; "AR2 Atom in a planar ring, usually the
    # ring has two continous single bonds and at least two double bonds"; "AR3
    # ... one or several double bonds formed between non-ring atoms and the ring
    # atoms") gives no direct algorithm, but its own canonical examples and a
    # real external reference (antechamber/GAFFTemplateGenerator, run 260820 on
    # benzene/pyridine/naphthalene/toluene/biphenyl/anthracene vs.
    # furan/pyrrole/thiophene) converge on a clean, checkable proxy: every AR1
    # example is a 6-membered aromatic ring (matching AR1's own "benzene and
    # pyridine" citation, and confirmed for naphthalene's doubly-ring-membered
    # bridgeheads too -- ring COUNT doesn't affect AR1 eligibility, only ring
    # SIZE does); every AR2/AR3 example in scope is a 5-membered heteroaromatic,
    # whose Kekule structure necessarily has the heteroatom's two ring bonds as
    # "continuous single bonds" per AR2's own description (the heteroatom
    # contributes a lone pair, not a pi bond, breaking full delocalization the
    # way a 6-membered ring's alternating pattern doesn't). AR2 vs AR3 aren't
    # separately distinguished here because DEF-file rules that read either
    # token always OR them together for the same output type (e.g. cc's lines
    # 37-58) -- no case in the current benchmark set needs to tell them apart.
    # non-aromatic ring atom that is sp3 carbon -> AR5; other ring atom -> AR4.
    #
    # FIXED 260820 (post-merge PARITY audit): the ring-SIZE-only proxy above is
    # necessary but not sufficient -- confirmed wrong by a real reference on
    # 2-pyranone (O=C1C=CC=CO1), whose 6-membered ring RDKit marks aromatic but
    # real antechamber types as the cc/cd family, not ca. Root cause, found in
    # AmberTools' own ring-classification source (ring.c): its AR3 test
    # ("planar rings formed 'outside' double bonds") is checked, and can fire,
    # BEFORE the AR1 "pure aromatic ring" test -- a ring where some ring atom
    # has an exocyclic double/triple bond to a NON-ring neighbor is AR3
    # regardless of size or RDKit's own aromaticity perception (matches
    # pyranone's carbonyl carbon exactly: its ring-internal bonds are
    # single/aromatic, but its bond to the exocyclic carbonyl O is double).
    # Ported as a targeted addition (not the full initarom/threshold system in
    # ring.c, which has its own AR2/AR4/AR5 machinery this module doesn't need
    # given the cc/cd-family rules never distinguish AR2 from AR3): a
    # 6-membered otherwise-AR1-eligible ring is demoted to AR23 if ANY of its
    # atoms has a post-Kekulization DOUBLE/TRIPLE bond to a neighbor outside
    # the ring. Verified NOT to over-fire on benzaldehyde/styrene (exocyclic
    # double bond one atom removed from the ring, via a single ring-to-
    # substituent bond) -- both stay `ca`, matching real antechamber exactly.
    aromaticity_class: str | None = None
    if atom.GetIsAromatic():
        # An atom can belong to more than one 6-ring (e.g. naphthalene's
        # bridgeheads); AR1 fires if AT LEAST ONE of its 6-rings is "clean"
        # (matches real antechamber's per-ring, independent-counter
        # semantics -- the AR1 token check is a plain "was this atom ever
        # tagged AR1 by any ring", not a single resolved class per atom).
        six_rings = [r for r in ring_info.AtomRings() if idx in r and len(r) == 6]
        has_clean_six_ring = any(
            not _ring_has_exocyclic_multibond(mol, ring) for ring in six_rings
        )
        aromaticity_class = "AR1" if has_clean_six_ring else "AR23"
    elif in_ring and atom.GetSymbol() == "C" and atom.GetHybridization().name == "SP3":
        aromaticity_class = "AR5"
    elif in_ring:
        aromaticity_class = "AR4"

    # Bond-type tallies include bonds to hydrogen (unlike f9's neighbor-pattern
    # matching, which is heavy-atom-only): a terminal alkyne carbon's only
    # single bond is to H (cg's "[sb,tb]" requires it to be counted), and the
    # DEF file gives no indication f8's bond-count facts should be H-exclusive.
    bond_counts = {"SB": 0, "DB": 0, "TB": 0, "AB": 0, "DL": 0, "sb": 0, "db": 0, "tb": 0}
    for bond in atom.GetBonds():
        facts = _bond_category_facts(bond)
        for key, present in facts.items():
            if present:
                bond_counts[key] += 1

    return AtomBondFacts(
        in_ring=in_ring,
        ring_counts_by_size=ring_counts_by_size,
        aromaticity_class=aromaticity_class,
        bond_counts=bond_counts,
    )


def _matches_aromaticity_token(word: str, aromaticity_class: str | None) -> bool:
    if aromaticity_class is None:
        return False
    if word == "AR4" or word == "AR5":
        return aromaticity_class == word
    if word == "AR1":
        return aromaticity_class == "AR1"
    if word in ("AR2", "AR3"):
        return aromaticity_class == "AR23"
    return False


def _prop_token_matches(tok: PropToken, facts: AtomBondFacts) -> bool:
    word = tok.word

    if word == "NR":
        return not facts.in_ring
    if word == "RG":
        return facts.in_ring
    if word.startswith("RG") and word[2:].isdigit():
        size = int(word[2:])
        count = facts.ring_counts_by_size.get(size, 0)
        if tok.count is not None:
            return count == tok.count
        return count > 0
    if word.startswith("AR") and word[2:].isdigit():
        return _matches_aromaticity_token(word, facts.aromaticity_class)

    if word in facts.bond_counts:
        count = facts.bond_counts[word]
        if tok.count is not None:
            return count == tok.count
        return count > 0

    return False


# f7 (H-only electron-withdrawing-neighbor count) grammar note: ATOMTYPE_GFF2.DEF's
# footer defines f7 ("For hydrogen, number of the electron-withdrawal atoms
# connected to the atom that the hydrogen attached") and the term "EW" ("Electron-
# withdraw atom") but never actually enumerates which elements count as EW -- "EW"
# is never used as a token in any real f8/f9 rule pattern anywhere in the file
# (confirmed by grep). FIXED 260820 (post-merge PARITY audit): this used to be
# {N, O, F, Cl, Br, I} -- a plausible-looking guess that omitted sulfur and was
# never actually checked against a live reference before landing. Independently
# confirmed via AmberTools' real source (`aromatic()`,
# Amber-MD/AmberClassic/src/antechamber/aromatic.c): its `ewd` (electron-
# withdrawing) flag is set for exactly atomic numbers 7/8/16/9/17/35/53
# (N/O/S/F/Cl/Br/I) and nothing else -- S is explicitly commented "S is
# considered electron withdraw group". The prior omission caused a real,
# reproducible bug: thiophene's ring H's (beta to the ring sulfur) mistyped as
# `ha` instead of `h4` (confirmed against real antechamber output).
_EW_ATOMS = frozenset({"N", "O", "S", "F", "Cl", "Br", "I"})


def _h_ew_neighbor_count(h_atom) -> int:
    """Count electron-withdrawing atoms bonded to h_atom's own heavy attachment.

    f7 is meaningful only for H atoms (per the DEF footer); an H atom has
    exactly one real neighbor (its attachment atom), whose OTHER neighbors
    (not the H itself) are what f7 counts.
    """
    neighbors = list(h_atom.GetNeighbors())
    if len(neighbors) != 1:
        return 0
    attachment = neighbors[0]
    return sum(
        1
        for nb in attachment.GetNeighbors()
        if nb.GetIdx() != h_atom.GetIdx() and nb.GetSymbol() in _EW_ATOMS
    )


@dataclass
class Gaff2Rule:
    """A single GAFF2 atom type assignment rule.

    Each rule maps molecular properties to a GAFF2 atom type.
    Fields correspond to ATD line format (per ATOMTYPE_GFF2.DEF's own footer):
    - f2: atom_type - the GAFF2 type to assign (e.g., c3, ca, n)
    - f3: residue - residue name filter (* means any)
    - f4: atomic_num - atomic number
    - f5: num_attached - number of attached atoms (heavy + H, per the footer)
    - f6: num_h - number of hydrogen attachments
    - f7: h_ew_count - H-only electron-withdrawing-neighbor count (meaningful
      only when atomic_num == 1; enforced via _h_ew_neighbor_count/_EW_ATOMS,
      see that helper's docstring for the EW-element judgment call)
    - f8: atomic_prop - bracketed atomic-property expression (ring/aromaticity/
      bond-type-count facts about the atom itself)
    - f9: chem_env - parenthesized chemical-environment neighbor pattern
    """

    atom_type: str
    residue: str
    atomic_num: int
    num_attached: int | None
    num_h: int | None
    h_ew_count: int | None
    atomic_prop: AtomicPropExpr | None
    chem_env: ChemEnvExpr | None

    def matches(self, atom, wildatom_map: dict[str, list[str]]) -> bool:
        """Check if this rule matches the given RDKit atom."""
        if self.atomic_num != atom.GetAtomicNum():
            return False

        if self.num_attached is not None and self.num_attached != _attached_count(atom):
            return False

        # includeNeighbors=True: after AllChem.AddHs, H atoms are explicit graph
        # nodes, so the default (implicit/explicit-count-only) form would return 0.
        num_h = atom.GetTotalNumHs(includeNeighbors=True)
        if self.num_h is not None and self.num_h != num_h:
            return False

        if self.h_ew_count is not None and self.h_ew_count != _h_ew_neighbor_count(atom):
            return False

        if self.atomic_prop is not None:
            facts = _atom_bond_facts(atom)
            if not atomic_prop_matches(self.atomic_prop, facts):
                return False

        if self.chem_env is not None:
            if not chem_env_matches(self.chem_env, atom, wildatom_map):
                return False

        return True


def parse_wildatom_defs(lines: list[str]) -> dict[str, list[str]]:
    """Parse WILDATOM definitions from the rule file header.

    Lines are space-separated element lists, e.g. "WILDATOM XX C N O S P" ->
    {"XX": ["C","N","O","S","P"]} -- not comma-joined, so a single `parts[2]`
    with a comma-split previously silently captured only the first element.
    """
    wildatom_map: dict[str, list[str]] = {}

    for line in lines:
        line = line.strip()
        if not line.startswith("WILDATOM"):
            continue

        parts = line.split()
        if len(parts) >= 3:
            symbol = parts[1]
            elements = parts[2:]
            wildatom_map[symbol] = elements

    return wildatom_map


def parse_gaff2_rules(def_path: str | Path) -> tuple[list[Gaff2Rule], dict[str, list[str]]]:
    """Parse ATOMTYPE_GFF2.DEF file.

    Args:
        def_path: Path to ATOMTYPE_GFF2.DEF file

    Returns:
        Tuple of (list of Gaff2Rule objects, WILDATOM map)
    """
    path = Path(def_path)
    content = path.read_text()
    lines = content.split("\n")

    wildatom_map = parse_wildatom_defs(lines)

    rules: list[Gaff2Rule] = []
    in_definition = False

    for line in lines:
        line = line.strip()

        if "efination begin" in line.lower():
            in_definition = True
            continue

        if not in_definition or not line.startswith("ATD"):
            continue

        if "&" not in line:
            continue

        line = line.removesuffix("&").strip()
        if line.startswith("ATD"):
            line = line[3:].strip()

        parts = line.split()
        # Minimum valid row is "<atom_type> <residue> <atomic_num>" (3 tokens) --
        # bare single-element rules like "ATD f * 9 &" have no further fields.
        # (This was previously "< 4", which silently dropped every such rule --
        # roughly half the file, mostly halogens/metals/late-periodic-table
        # fallback types -- meaning those elements always fell through to the
        # generic "x" placeholder instead of their real GAFF2 type.)
        if len(parts) < 3:
            continue

        try:
            atom_type = parts[0]
            residue = parts[1] if parts[1] != "*" else "*"
            atomic_num = int(parts[2])

            num_attached = None
            num_h = None
            h_ew_count = None

            idx = 3
            if idx < len(parts) and parts[idx] != "*":  # f5
                num_attached = int(parts[idx])
            idx += 1

            if idx < len(parts) and parts[idx] != "*":  # f6
                num_h = int(parts[idx])
            idx += 1

            remaining = " ".join(parts[idx:]) if idx < len(parts) else ""
            remaining = remaining.lstrip()

            # f7: h_ew -- only meaningful for hydrogen rows (atomic_num == 1); a
            # bare integer, or "*" for every non-H row. Consume exactly one token
            # positionally rather than silently discarding it.
            if remaining.startswith("*"):
                remaining = remaining[1:].strip()
            else:
                f7_match = re.match(r"(\d+)", remaining)
                if f7_match:
                    h_ew_count = int(f7_match.group(1))
                    remaining = remaining[f7_match.end():].strip()

            # f8: bracketed atomic-property expression, "[...]" or "*" or absent.
            atomic_prop: AtomicPropExpr | None = None
            if remaining.startswith("["):
                close = remaining.find("]")
                if close != -1:
                    atomic_prop = parse_atomic_prop(remaining[1:close])
                    remaining = remaining[close + 1:].strip()
            elif remaining.startswith("*"):
                remaining = remaining[1:].strip()

            # f9: parenthesized chemical-environment neighbor pattern, "(...)" or
            # "*" or absent (end of line before "&"). There is no true f10 --
            # anything left after this is a malformed line, not a further field.
            chem_env: ChemEnvExpr | None = None
            if remaining.startswith("("):
                chem_env = parse_chem_env(remaining)

            rule = Gaff2Rule(
                atom_type=atom_type,
                residue=residue,
                atomic_num=atomic_num,
                num_attached=num_attached,
                num_h=num_h,
                h_ew_count=h_ew_count,
                atomic_prop=atomic_prop,
                chem_env=chem_env,
            )
            rules.append(rule)

        except (ValueError, IndexError):
            continue

    return rules, wildatom_map


def extract_atom_features(
    mol: Chem.Mol,
) -> list[dict]:
    """Extract features from RDKit molecule needed for GAFF2 typing.

    Args:
        mol: RDKit molecule

    Returns:
        List of dicts, one per atom with features for matching
    """
    if Chem is None:
        raise ImportError("RDKit is required. Install with: pip install rdkit")

    features: list[dict] = []

    rings = mol.GetRingInfo()

    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()

        # Skip hydrogens for GAFF2 typing (they get typed based on their heavy atom neighbor)
        if atomic_num == 1:
            continue

        # After AllChem.AddHs, H atoms are explicit graph nodes, so
        # GetNumExplicitHs() returns 0 (H is a real atom, not a valence annotation).
        # Walk bonds to count actual H-atom neighbors; add GetNumImplicitHs for
        # molecules that still carry implicit H (pre-AddHs path).
        num_implicit_h = atom.GetNumImplicitHs()
        num_h = (
            sum(
                1 for bond in atom.GetBonds()
                if mol.GetAtomWithIdx(bond.GetOtherAtomIdx(atom.GetIdx())).GetAtomicNum() == 1
            )
            + num_implicit_h
        )

        # Count non-hydrogen bonds (standard degree)
        # Note: RDKit GetDegree() counts AROMATIC bonds as degree 1 each
        # For benzene C: GetDegree() = 3 (2 aromatic bonds + 1 H), which matches GAFF "attached"
        heavy_degree = atom.GetDegree()

        # Also compute effective degree including implicit hydrogens
        # This is what GAFF "attached" field represents
        attached_with_implicit_h = heavy_degree + num_implicit_h

        is_aromatic = atom.GetIsAromatic()

        ring_info = atom.IsInRing()
        ring_size = None
        if ring_info:
            for size in range(3, 10):
                if rings.IsAtomInRingOfSize(atom.GetIdx(), size):
                    ring_size = size
                    break

        bond_types: list[tuple[int, str]] = []
        neighbor_elements: list[str] = []

        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(atom.GetIdx())
            other_atom = mol.GetAtomWithIdx(other_idx)

            # Only consider bonds to non-hydrogen neighbors
            if other_atom.GetAtomicNum() == 1:
                continue

            neighbor_elements.append(other_atom.GetSymbol())

            bt = bond.GetBondType()
            if bt == 1:
                bond_types.append((1, "SB"))
            elif bt == 2:
                bond_types.append((2, "DB"))
            elif bt == 3:
                bond_types.append((3, "TB"))
            elif bt.name == "AROMATIC":
                bond_types.append((1, "AR"))

        # Also compute total H count based on expected valency
        # For C: 4 bonds total, N: 3 bonds, O: 2 bonds, etc.
        expected_valence = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 3}
        elem = atom.GetSymbol()
        expected = expected_valence.get(elem, 4)
        # Count all bonds (including to H)
        actual_bonds = atom.GetDegree() + atom.GetNumImplicitHs()
        total_h = max(0, expected - actual_bonds)

        # Also compute total H count based on expected valency
        # For C: 4 bonds total, N: 3 bonds, O: 2 bonds, etc.
        expected_valence = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 3}
        elem = atom.GetSymbol()
        expected = expected_valence.get(elem, 4)
        actual_bonds = atom.GetDegree() + atom.GetNumImplicitHs()
        total_h = max(0, expected - actual_bonds)

        feature = {
            "atomic_num": atomic_num,
            "num_heavy_neighbors": heavy_degree,
            "attached_with_implicit_h": attached_with_implicit_h,
            "num_h": num_h,
            "total_h": total_h,
            "is_aromatic": is_aromatic,
            "ring_size": ring_size,
            "bond_types": bond_types,
            "neighbor_elements": neighbor_elements,
        }
        features.append(feature)

    return features


# cc/cd (and analogous) ring/chain-alternation bookkeeping: AMBER's bonded-
# torsion-parameter tables need conjugated-family atoms split into two
# alternating labels so a formally-single-bonded pair and a formally-double-
# bonded pair within the same delocalized system don't share one torsion
# term. There is no corresponding ATD rule anywhere in ATOMTYPE_GFF2.DEF for
# the "primed" half of each pair (`cd` never appears as a rule's atom_type
# field; confirmed by grep) -- this can't come from rule-matching, it's a
# separate graph-coloring post-process. Ported directly from AmberTools'
# real algorithm (`atadjust()`/`cpadjust()`, src/antechamber/atomtype.c,
# Amber-MD/AmberClassic) rather than reverse-engineered from output alone:
# verified line-for-line against real antechamber output on 7 molecules
# (furan/pyrrole/thiophene/imidazole needing the split; 1,3-butadiene/
# divinyl-ketone/acrolein/biphenyl correctly needing none).
#
# atadjust() vs cpadjust() are NOT structurally identical, despite sharing the
# same coloring rule (confirmed by re-reading both functions independently,
# post-merge PARITY audit 260820): atadjust()'s propagation loop has a
# `flag`-gated reseed line that picks up one new never-before-seen connected
# component per outer pass (bounded by the total family-atom count -- more
# than enough passes for any realistic molecule), so it colors EVERY
# disconnected same-family subgraph in the molecule. cpadjust() has no such
# reseed: it seeds exactly ONE atom overall (the first "cp" atom found) and
# only ever colors that one component -- any other disconnected `cp` system
# in the same molecule is left untouched (stays `cp`, never becomes `cq`).
# `single_seed_only=True` on the cp/cq call below reproduces this real,
# intentional asymmetry rather than treating both calls the same way.
_CONJUGATED_ALTERNATION_PAIRS = {
    "cc": "cd", "ce": "cf", "cg": "ch",
    "pc": "pd", "pe": "pf", "nc": "nd", "ne": "nf",
}
_BIPHENYL_ALTERNATION_PAIR = {"cp": "cq"}


def _relabel_conjugated_alternation(
    mol: Chem.Mol,
    atom_types: list[str],
    pairs: dict[str, str],
    *,
    single_seed_only: bool = False,
) -> None:
    """Mutate `atom_types` in place: 2-color each connected subgraph of
    same-"unprimed"-family atoms by Kekule bond parity (single bond = same
    label, double/triple = flipped label), relabeling flipped atoms to their
    paired ("primed") type.

    `mol` must already be Kekulized (see `assign_gaff2_atom_types`) -- an
    un-Kekulized aromatic bond reports `AROMATIC`, which this function
    treats as "not SINGLE" (a flip), silently corrupting the coloring for
    ring systems exactly like the sb/db precondition this mirrors.

    `single_seed_only`: when True, color only the FIRST connected component
    found (by atom index) and leave every other same-family atom untouched --
    matches real AmberTools' `cpadjust()` (see the module-level comment above
    `_BIPHENYL_ALTERNATION_PAIR`). When False (default), color every
    connected component independently -- matches real `atadjust()`.
    """
    n = mol.GetNumAtoms()
    visited = [False] * n
    for start in range(n):
        if visited[start] or atom_types[start] not in pairs:
            continue
        sign = {start: 1}
        visited[start] = True
        stack = [start]
        while stack:
            cur = stack.pop()
            for bond in mol.GetAtomWithIdx(cur).GetBonds():
                other = bond.GetOtherAtomIdx(cur)
                if visited[other] or atom_types[other] not in pairs:
                    continue
                same = bond.GetBondType().name == "SINGLE"
                sign[other] = sign[cur] if same else -sign[cur]
                visited[other] = True
                stack.append(other)
        for idx, s in sign.items():
            if s == -1:
                atom_types[idx] = pairs[atom_types[idx]]
        if single_seed_only:
            return


def assign_gaff2_atom_types(
    mol: Chem.Mol,
    rules: list[Gaff2Rule] | None = None,
    wildatom_map: dict[str, list[str]] | None = None,
) -> list[str]:
    """Assign GAFF2 atom types to an RDKit molecule.

    Args:
        mol: RDKit molecule
        rules: Pre-parsed GAFF2 rules (optional, will use default if not provided)
        wildatom_map: Pre-parsed WILDATOM map

    Returns:
        List of GAFF2 atom type strings, ONE PER ATOM, in `mol.GetAtoms()` index
        order -- including H atoms if `mol` has explicit hydrogens (e.g. after
        `Chem.AddHs`). This is an intentional, index-aligned contract (changed
        260820): previously H atoms were skipped entirely, so the returned list
        was shorter than `mol.GetNumAtoms()` and callers had to reconstruct
        index alignment themselves by re-walking `mol.GetAtoms()` and filtering
        `GetAtomicNum() != 1` in lockstep -- see `build_gaff2_ffxml`, now
        simplified to rely on this contract directly.
    """
    if Chem is None:
        raise ImportError("RDKit is required. Install with: pip install rdkit")

    if rules is None:
        rules, wildatom_map = _get_default_rules()

    if wildatom_map is None:
        wildatom_map = {}

    atom_types: list[str] = []

    # Kekulize a local copy before matching: f8/f9's lowercase sb/db bond-category
    # tokens require the true per-bond Kekule single/double identity (see
    # _bond_category_facts), which RDKit only exposes once aromatic ring bonds
    # are Kekulized -- by default they report GetBondType() == AROMATIC, not
    # SINGLE/DOUBLE. clearAromaticFlags=False keeps GetIsAromatic() intact so AB/DL
    # and aromaticity-class matching are unaffected. Never mutates the caller's
    # mol. A molecule that has already passed Chem.SanitizeMol should always
    # Kekulize successfully (sanitization Kekulizes internally to validate) --
    # deliberately NOT silently falling back to the un-Kekulized molecule on
    # failure: that would make every downstream sb/db-dependent match silently
    # wrong for an aromatic bond (masked by _bond_category_facts's own
    # precondition assertion firing at an unrelated call site instead, with a
    # confusing stack trace far from the real cause). Fail loudly here instead,
    # at the actual point of failure, with an explicit warning first.
    mol_for_matching = Chem.Mol(mol)
    try:
        Chem.Kekulize(mol_for_matching, clearAromaticFlags=False)
    except Chem.KekulizeException as exc:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "assign_gaff2_atom_types: Kekulize failed (%s); refusing to "
            "silently degrade sb/db bond-category matching for this molecule",
            exc,
        )
        raise

    # Precedence: first-match-in-file-order (parse_gaff2_rules preserves DEF-file
    # declaration order), per ATOMTYPE_GFF2.DEF's own "defination order is crucial"
    # rule. H atoms go through the SAME rule loop as heavy atoms (260820 -- the 13
    # H-rules, ATOMTYPE_GFF2.DEF lines 79-91, are parsed like any other ATD row;
    # they just never ran before this fix).
    for atom in mol_for_matching.GetAtoms():
        assigned = False
        for rule in rules:
            if rule.matches(atom, wildatom_map):
                atom_types.append(rule.atom_type)
                assigned = True
                break

        if not assigned:
            atomic_num = atom.GetAtomicNum()
            if atomic_num == 1:
                atom_types.append("ha")  # DEF line 91: unconstrained H catch-all
            elif atomic_num == 6:
                atom_types.append("c3")
            elif atomic_num == 7:
                atom_types.append("n3")
            elif atomic_num == 8:
                atom_types.append("oh")
            elif atomic_num == 16:
                atom_types.append("s")
            else:
                atom_types.append("x")

    # cc/cd-family ring/chain-alternation bookkeeping (see the two constants'
    # docstring comment above) -- two independent passes, matching
    # AmberTools' real atadjust()/cpadjust() structure: cc/ce/cg/pc/pe/nc/ne
    # and cp are disjoint families that must never cross-propagate through
    # each other even if adjacent. single_seed_only=True on the cp/cq call
    # reproduces cpadjust()'s real (and structurally different from
    # atadjust()'s) single-component-only behavior -- see that constant's
    # module-level comment.
    _relabel_conjugated_alternation(mol_for_matching, atom_types, _CONJUGATED_ALTERNATION_PAIRS)
    _relabel_conjugated_alternation(
        mol_for_matching, atom_types, _BIPHENYL_ALTERNATION_PAIR, single_seed_only=True
    )

    return atom_types


_default_rules: list[Gaff2Rule] | None = None
_default_wildatom: dict[str, list[str]] | None = None


def _get_default_rules() -> tuple[list[Gaff2Rule], dict[str, list[str]]]:
    """Get default GAFF2 rules (cached)."""
    global _default_rules, _default_wildatom

    if _default_rules is None:
        # Use fixed relative path from project root
        rules_path = Path(__file__).parent.parent / "assets" / "gaff" / "dat" / "ATOMTYPE_GFF2.DEF"

        if rules_path.exists():
            _default_rules, _default_wildatom = parse_gaff2_rules(rules_path)
        else:
            _default_rules = []
            _default_wildatom = {}

    return (
        _default_rules if _default_rules is not None else [],
        _default_wildatom if _default_wildatom is not None else {},
    )


def load_gaff2_rules(
    def_path: str | Path | None = None,
) -> tuple[list[Gaff2Rule], dict[str, list[str]]]:
    """Load GAFF2 rules from file.

    Args:
        def_path: Path to ATOMTYPE_GFF2.DEF. If None, uses default bundled.

    Returns:
        Tuple of (rules list, wildatom map)
    """
    if def_path is None:
        return _get_default_rules()

    return parse_gaff2_rules(def_path)


def load_gaff2_parameters(dat_path: str | Path | None = None) -> dict:
    """Load GAFF2 parameter tables from .dat file.

    Args:
        dat_path: Path to GAFF2 .dat file (e.g., gaff-2.2.20.dat).
                  If None, uses default bundled.

    Returns:
        Dict with 'masses', 'bonds', 'angles', 'torsions', 'impropers', 'vdw'.
        'vdw' maps atom_type -> (rmin_half_angstrom, epsilon_kcal_mol).
    """
    if dat_path is None:
        dat_path = Path(__file__).parent.parent / "assets" / "gaff" / "dat" / "gaff-2.2.20.dat"

    params: dict = {
        'masses': {},
        'bonds': {},
        'angles': {},
        'torsions': {},
        'impropers': {},
        'vdw': {},
    }

    content = Path(dat_path).read_text()
    lines = content.split('\n')

    in_vdw = False
    for line in lines:
        line_stripped = line.strip()

        # Detect VdW section (MOD4 RE header)
        if line_stripped.startswith('MOD4'):
            in_vdw = True
            continue

        if in_vdw:
            if not line_stripped:
                continue
            parts = line_stripped.split()
            if len(parts) >= 3:
                try:
                    params['vdw'][parts[0]] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    pass
            continue

        if not line_stripped:
            continue

        parts = line_stripped.split()
        if len(parts) < 2:
            continue

        # Handle AMBER .dat quirk: single-char types are padded, so "c -o kb r0"
        # splits as ["c", "-o", "kb", "r0"]. Reconstruct the dash-joined key.
        if '-' not in parts[0] and len(parts) >= 2 and parts[1].startswith('-'):
            parts = [parts[0] + parts[1]] + list(parts[2:])

        first = parts[0]

        if '-' in first:
            dash_count = first.count('-')

            # Parse bond: type1-type2  kb  r0
            if dash_count == 1:
                t1, t2 = first.split('-')
                if len(t1) <= 3 and len(t2) <= 3:
                    try:
                        kb = float(parts[1])
                        r0 = float(parts[2])
                        if kb > 0:
                            params['bonds'][(t1, t2)] = (kb, r0)
                    except (ValueError, IndexError):
                        pass

            # Parse angle: type1-type2-type3  kt  t0
            elif dash_count == 2:
                t1, rest = first.split('-', 1)
                t2, t3 = rest.split('-', 1)
                if len(t1) <= 3 and len(t2) <= 3 and len(t3) <= 3:
                    try:
                        kt = float(parts[1])
                        t0 = float(parts[2])
                        if kt > 0:
                            params['angles'][(t1, t2, t3)] = (kt, t0)
                    except (ValueError, IndexError):
                        pass

            # Parse torsion or improper: type1-type2-type3-type4  ...
            elif dash_count == 3:
                t1, rest = first.split('-', 1)
                t2, rest2 = rest.split('-', 1)
                t3, t4 = rest2.split('-', 1)
                if len(t1) <= 3 and len(t2) <= 3 and len(t3) <= 3 and len(t4) <= 3:
                    try:
                        periodicity = int(parts[1])
                        kt = float(parts[2])
                        phase = float(parts[3])
                        # Torsion: has 4+ terms per line (may have multiple periodicity)
                        if len(parts) >= 5:
                            # This is a torsion
                            key = (t1, t2, t3, t4)
                            if key not in params['torsions']:
                                params['torsions'][key] = []
                            params['torsions'][key].append((periodicity, kt, phase))
                        else:
                            # This could be improper
                            try:
                                kt_imp = float(parts[2])
                                phase_imp = float(parts[3])
                                if kt_imp > 0:
                                    params['impropers'][(t1, t2, t3, t4)] = (kt_imp, phase_imp)
                            except (ValueError, IndexError):
                                pass
                    except (ValueError, IndexError):
                        pass

        # Parse mass: type  mass
        elif len(first) <= 3 and first.replace('+', '').islower():
            try:
                params['masses'][first] = float(parts[1])
            except (ValueError, IndexError):
                pass

    return params


# ---------------------------------------------------------------------------
# H-type assignment and OpenMM FFXML builder
# ---------------------------------------------------------------------------

_KCAL_TO_KJ = 4.184
_BOND_K_CONV = 2.0 * _KCAL_TO_KJ / (0.1 ** 2)   # kcal/mol/Å² → kJ/mol/nm²
_BOND_R_CONV = 0.1                                  # Å → nm
_ANGLE_K_CONV = 2.0 * _KCAL_TO_KJ                  # kcal/mol/rad² → kJ/mol/rad²
_ANGLE_T_CONV = _math.pi / 180.0                    # deg → rad
_TORSION_K_CONV = _KCAL_TO_KJ                       # kcal/mol → kJ/mol
_TORSION_P_CONV = _math.pi / 180.0                  # deg → rad
_LJ_SIGMA_CONV = 2.0 * (2.0 ** (-1.0 / 6.0)) * 0.1  # Rmin/2 (Å) → σ (nm)
_LJ_EPS_CONV = _KCAL_TO_KJ                          # kcal/mol → kJ/mol

_ELEM_MASS_DEFAULT = {
    "C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007,
    "S": 32.06, "P": 30.974, "F": 18.998, "Cl": 35.453,
    "Br": 79.904, "I": 126.904,
}

_VDW_ELEM_DEFAULT = {
    "C": (1.9080, 0.0860), "O": (1.6612, 0.2100),
    "H": (1.4593, 0.0208), "N": (1.8240, 0.1700),
    "S": (2.0000, 0.2500), "P": (2.1000, 0.2000),
}

_BOND_TYPE_SUB = {
    "cx": "c3", "cy": "c3", "c5": "c3", "c6": "c3",
    "cs": "c2", "cz": "c2", "ca": "c2", "cc": "c2", "cd": "c2",
    "ce": "c2", "cf": "c2", "cp": "c2", "cq": "c2",
    "cg": "c1", "ch": "c1",
}


def assign_pdb_atom_names(rdmol: Chem.Mol) -> list[str]:
    """Assign unique PDB atom names to rdmol in-place and return them.

    If an atom already has a non-empty MonomerInfo name it is kept as-is.
    New names are element+counter (e.g. C1, C2, O1, H1 …).
    """
    from rdkit.Chem import AtomPDBResidueInfo

    counts: dict[str, int] = {}
    names: list[str] = []
    for atom in rdmol.GetAtoms():
        mi = atom.GetMonomerInfo()
        existing = mi.GetName().strip() if mi else ""
        if existing:
            names.append(existing)
        else:
            elem = atom.GetSymbol()
            counts[elem] = counts.get(elem, 0) + 1
            name = f"{elem}{counts[elem]}"
            if mi is None:
                mi = AtomPDBResidueInfo()
            mi.SetName(f"{name:<4s}")
            atom.SetMonomerInfo(mi)
            names.append(name)
    return names


def _lookup_bond_params(
    ti: str, tj: str, params: dict
) -> tuple[float, float]:
    """Return (kb, r0) for a bond, trying substitutions on miss."""
    for a, b in [(ti, tj), (tj, ti)]:
        if (a, b) in params['bonds']:
            return params['bonds'][(a, b)]
    # substitution fallback
    ti_s = _BOND_TYPE_SUB.get(ti, ti)
    tj_s = _BOND_TYPE_SUB.get(tj, tj)
    for a, b in [(ti_s, tj_s), (tj_s, ti_s)]:
        if (a, b) in params['bonds']:
            return params['bonds'][(a, b)]
    return (0.0, 1.50)


def _lookup_angle_params(
    ti: str, tj: str, tk: str, params: dict
) -> tuple[float, float]:
    """Return (kt, t0) for an angle, trying substitutions on miss."""
    for a, b, c in [(ti, tj, tk), (tk, tj, ti)]:
        if (a, b, c) in params['angles']:
            return params['angles'][(a, b, c)]
    ti_s = _BOND_TYPE_SUB.get(ti, ti)
    tj_s = _BOND_TYPE_SUB.get(tj, tj)
    tk_s = _BOND_TYPE_SUB.get(tk, tk)
    for a, b, c in [(ti_s, tj_s, tk_s), (tk_s, tj_s, ti_s)]:
        if (a, b, c) in params['angles']:
            return params['angles'][(a, b, c)]
    return (0.0, 120.0)


def build_gaff2_ffxml(
    rdmol: Chem.Mol,
    resname: str,
    charges: list[float],
    *,
    gaff_version: str = "gaff-2.2.20",
) -> str:
    """Build a complete OpenMM FFXML string for a single small-molecule residue.

    Args:
        rdmol: RDKit Mol with explicit H atoms.
        resname: PDB residue name (e.g. "OHP").
        charges: Partial charges, one per atom in mol atom-index order (including H).
        gaff_version: GAFF version string used to select the .dat file.

    Returns:
        FFXML string suitable for openmm.app.ForceField.loadFile(StringIO(xml)).
    """
    if Chem is None:
        raise ImportError("RDKit is required.")

    atom_names = assign_pdb_atom_names(rdmol)
    n_atoms = rdmol.GetNumAtoms()

    if len(charges) != n_atoms:
        raise ValueError(
            f"charges length {len(charges)} != mol atom count {n_atoms}"
        )

    # --- GAFF2 type assignment ---
    # assign_gaff2_atom_types returns one type per atom index (including H,
    # via the real DEF rule engine -- see its docstring for the 260820
    # index-alignment contract change; this used to be a two-pass
    # heavy-then-heuristic-H construction via a now-deleted
    # _H_TYPE_BY_HEAVY/_H_TYPE_ELEMENT_DEFAULT lookup).
    all_types = assign_gaff2_atom_types(rdmol)
    idx_to_type: dict[int, str] = {
        atom.GetIdx(): t for atom, t in zip(rdmol.GetAtoms(), all_types, strict=True)
    }

    params = load_gaff2_parameters(
        Path(__file__).parent.parent / "assets" / "gaff" / "dat" / f"{gaff_version}.dat"
    )

    lines: list[str] = ['<?xml version="1.0" encoding="utf-8"?>', "<ForceField>"]

    # --- AtomTypes ---
    lines.append("  <AtomTypes>")
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        gaff_type = idx_to_type[idx]
        elem = atom.GetSymbol()
        mass = params['masses'].get(gaff_type, _ELEM_MASS_DEFAULT.get(elem, 12.011))
        lines.append(
            f'    <Type name="{resname}_{idx}" class="{gaff_type}"'
            f' element="{elem}" mass="{mass:.4f}"/>'
        )
    lines.append("  </AtomTypes>")

    # --- Residues ---
    lines.append("  <Residues>")
    lines.append(f'    <Residue name="{resname}">')
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        name = atom_names[idx]
        q = charges[idx]
        lines.append(
            f'      <Atom name="{name}" type="{resname}_{idx}" charge="{q:.6f}"/>'
        )
    for bond in rdmol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        lines.append(
            f'      <Bond atomName1="{atom_names[i]}" atomName2="{atom_names[j]}"/>'
        )
    lines.append("    </Residue>")
    lines.append("  </Residues>")

    # --- HarmonicBondForce ---
    lines.append("  <HarmonicBondForce>")
    missing_bonds = 0
    for bond in rdmol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        ti = idx_to_type[i]
        tj = idx_to_type[j]
        kb, r0 = _lookup_bond_params(ti, tj, params)
        if kb == 0.0:
            missing_bonds += 1
        r0_nm = r0 * _BOND_R_CONV
        k_kj = kb * _BOND_K_CONV
        lines.append(
            f'    <Bond type1="{resname}_{i}" type2="{resname}_{j}"'
            f' length="{r0_nm:.6f}" k="{k_kj:.2f}"/>'
        )
    if missing_bonds:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "build_gaff2_ffxml: %d bonds missing GAFF2 parameters (kb=0)", missing_bonds
        )
    lines.append("  </HarmonicBondForce>")

    # --- HarmonicAngleForce ---
    lines.append("  <HarmonicAngleForce>")
    for atom in rdmol.GetAtoms():
        j = atom.GetIdx()
        bonds_j = list(atom.GetBonds())
        for b1_idx in range(len(bonds_j)):
            i = bonds_j[b1_idx].GetOtherAtomIdx(j)
            for b2_idx in range(b1_idx + 1, len(bonds_j)):
                k = bonds_j[b2_idx].GetOtherAtomIdx(j)
                ti = idx_to_type[i]
                tj = idx_to_type[j]
                tk = idx_to_type[k]
                kt, t0 = _lookup_angle_params(ti, tj, tk, params)
                t0_rad = t0 * _ANGLE_T_CONV
                k_kj = kt * _ANGLE_K_CONV
                lines.append(
                    f'    <Angle type1="{resname}_{i}" type2="{resname}_{j}"'
                    f' type3="{resname}_{k}" angle="{t0_rad:.6f}" k="{k_kj:.2f}"/>'
                )
    lines.append("  </HarmonicAngleForce>")

    # --- PeriodicTorsionForce ---
    lines.append("  <PeriodicTorsionForce>")
    seen_torsions: set[tuple] = set()
    for bond in rdmol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()
        atom_j = rdmol.GetAtomWithIdx(j)
        atom_k = rdmol.GetAtomWithIdx(k)
        for bond_i in atom_j.GetBonds():
            i = bond_i.GetOtherAtomIdx(j)
            if i == k:
                continue
            for bond_l in atom_k.GetBonds():
                ll = bond_l.GetOtherAtomIdx(k)
                if ll == j or ll == i:
                    continue
                key = tuple(sorted([(i, j, k, ll), (ll, k, j, i)])[0])
                if key in seen_torsions:
                    continue
                seen_torsions.add(key)
                ti = idx_to_type[i]
                tj = idx_to_type[j]
                tk_ = idx_to_type[k]
                tl = idx_to_type[ll]
                torsion_key = (ti, tj, tk_, tl)
                torsion_params = params['torsions'].get(torsion_key, [])
                if not torsion_params:
                    rev_key = (tl, tk_, tj, ti)
                    torsion_params = params['torsions'].get(rev_key, [])
                if not torsion_params:
                    ti_s = _BOND_TYPE_SUB.get(ti, ti)
                    tj_s = _BOND_TYPE_SUB.get(tj, tj)
                    tk_s = _BOND_TYPE_SUB.get(tk_, tk_)
                    tl_s = _BOND_TYPE_SUB.get(tl, tl)
                    for sub_key in [(ti_s, tj_s, tk_s, tl_s), (tl_s, tk_s, tj_s, ti_s)]:
                        torsion_params = params['torsions'].get(sub_key, [])
                        if torsion_params:
                            break
                if not torsion_params:
                    continue
                attrs = (
                    f'type1="{resname}_{i}" type2="{resname}_{j}"'
                    f' type3="{resname}_{k}" type4="{resname}_{ll}"'
                )
                for term_idx, (n, kt, phase) in enumerate(torsion_params, 1):
                    phase_rad = phase * _TORSION_P_CONV
                    k_kj = kt * _TORSION_K_CONV
                    attrs += (
                        f' periodicity{term_idx}="{n}"'
                        f' phase{term_idx}="{phase_rad:.6f}"'
                        f' k{term_idx}="{k_kj:.4f}"'
                    )
                lines.append(f"    <Proper {attrs}/>")
    lines.append("  </PeriodicTorsionForce>")

    # --- NonbondedForce ---
    lines.append('  <NonbondedForce coulomb14scale="0.8333333333" lj14scale="0.5">')
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        gaff_type = idx_to_type[idx]
        elem = atom.GetSymbol()
        q = charges[idx]
        if gaff_type in params['vdw']:
            rmin_half, epsilon = params['vdw'][gaff_type]
        else:
            rmin_half, epsilon = _VDW_ELEM_DEFAULT.get(elem, (1.9080, 0.0860))
        sigma_nm = rmin_half * _LJ_SIGMA_CONV
        eps_kj = epsilon * _LJ_EPS_CONV
        lines.append(
            f'    <Atom type="{resname}_{idx}" charge="{q:.6f}"'
            f' sigma="{sigma_nm:.8f}" epsilon="{eps_kj:.6f}"/>'
        )
    lines.append("  </NonbondedForce>")
    lines.append("</ForceField>")
    return "\n".join(lines)


def _get_espaloma_charges(mol: Chem.Mol) -> list[float]:
    """Compute partial charges using expaloma or fallback to Gasteiger.

    Tries native Rust expaloma first, then RDKit Gasteiger as fallback.
    Returns zero charges if nothing is available.
    """
    from rdkit import Chem

    mol_copy = Chem.Mol(mol)
    Chem.SanitizeMol(mol_copy)

    try:
        from proxide._proxider import (
            assign_espaloma_charges as assign_rust_charges,
        )
    except ImportError:
        assign_rust_charges = None

    try:
        from expaloma.featurize import from_rdkit_mol
    except ImportError:
        from_rdkit_mol = None

    if assign_rust_charges and from_rdkit_mol:
        try:
            g = from_rdkit_mol(mol_copy)
            h0 = np.ascontiguousarray(g.h0, dtype=np.float32)
            senders = np.ascontiguousarray(g.senders, dtype=np.uint32)
            receivers = np.ascontiguousarray(g.receivers, dtype=np.uint32)
            q_ref = np.ascontiguousarray(g.q_ref, dtype=np.float32)
            total_charge = float(q_ref.sum())

            q_rust = assign_rust_charges(
                h0,
                senders,
                receivers,
                np.zeros(h0.shape[0], dtype=np.uint32),
                1,
                [total_charge],
            )
            return list(q_rust)
        except Exception:
            pass

    try:
        mol_copy.ComputeGasteigerCharges()
        charges = []
        for atom in mol_copy.GetAtoms():
            charge = atom.GetDoubleProp("_GasteigerCharge")
            if charge == float("inf") or charge == float("-inf") or abs(charge) > 10:
                charge = 0.0
            charges.append(charge)
        return charges
    except Exception:
        return [0.0] * mol.GetNumAtoms()


def parameterize_gaff_with_rdkit(
    mol: Chem.Mol,
    gaff_version: str = "gaff-2.2.20",
) -> dict:
    """Assign GAFF2 parameters to an RDKit molecule.

    This function assigns GAFF2 atom types and looks up force field parameters
    without requiring AmberTools.

    Args:
        mol: RDKit molecule (should have explicit hydrogens)
        gaff_version: GAFF version string (default: gaff-2.2.20)

    Returns:
        Dict with keys:
        - atom_types: list of atom type strings
        - masses: dict of atom type -> mass
        - bonds: dict of (type1, type2) -> (kb, r0)
        - angles: dict of (type1, type2, type3) -> (kt, t0)
        - torsions: list of torsion parameters
    """
    if Chem is None:
        raise ImportError("RDKit is required. Install with: pip install rdkit")

    # Load parameters
    params = load_gaff2_parameters()

    # Assign atom types
    atom_types = assign_gaff2_atom_types(mol)

    # Extract molecule topology
    n_atoms = mol.GetNumAtoms()

    # Build bond list
    bonds = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetOtherAtomIdx(i)
        # Get atom types
        t_i = atom_types[i] if i < len(atom_types) else "x"
        t_j = atom_types[j] if j < len(atom_types) else "x"

        # Get bond order
        bo = bond.GetBondTypeAsDouble()

        # Determine bond type string
        if bo >= 1.9:
            bond_type = "tb"  # triple
        elif bo >= 1.4:
            bond_type = "db"  # double
        else:
            bond_type = "sb"  # single

        bonds.append({
            'i': i,
            'j': j,
            'type': bond_type,
            'order': bo,
            'gaff_type_i': t_i,
            'gaff_type_j': t_j,
        })

    # Build angle list (1-2-3 connections)
    angles = []
    for i in range(n_atoms):
        atom_i = mol.GetAtomWithIdx(i)
        for bond in atom_i.GetBonds():
            j = bond.GetOtherAtomIdx(i)
            if j <= i:
                continue
            for bond2 in mol.GetAtomWithIdx(j).GetBonds():
                k = bond2.GetOtherAtomIdx(j)
                if k <= j or k == i:
                    continue
                t_i = atom_types[i] if i < len(atom_types) else "x"
                t_j = atom_types[j] if j < len(atom_types) else "x"
                t_k = atom_types[k] if k < len(atom_types) else "x"
                angles.append({
                    'i': i, 'j': j, 'k': k,
                    'types': (t_i, t_j, t_k),
                })

    charges = _get_espaloma_charges(mol)

    used_types = set(atom_types)
    masses = {at: params['masses'].get(at, 0.0) for at in used_types}

    # Look up bond parameters
    for b in bonds:
        t1, t2 = b['gaff_type_i'], b['gaff_type_j']
        key = tuple(sorted([t1, t2]))
        if key in params['bonds']:
            kb, r0 = params['bonds'][key]
            b['kb'] = kb
            b['r0'] = r0
        else:
            b['kb'] = 0.0
            b['r0'] = 0.0

    # Look up angle parameters
    for a in angles:
        t1, t2, t3 = a['types']  # ty: ignore[not-iterable]
        key = (t1, t2, t3)
        if key in params['angles']:
            kt, t0 = params['angles'][key]
            a['kt'] = kt
            a['t0'] = t0
        else:
            a['kt'] = 0.0
            a['t0'] = 0.0

    # Build torsion list (1-2-3-4 connections)
    torsions = []

    # Substitutions for atom type looking (cx->c3, etc.)
    type_substitutions = {
        'cx': 'c3', 'cy': 'c3', 'c5': 'c3', 'c6': 'c3',
        'n7': 'n3', 'n8': 'n3', 'nx': 'n3',
        'ny': 'n3', 'ni': 'n', 'nu': 'n3', 'nv': 'n3',
    }

    def _substitute_type(t: str) -> str:
        if t == 'x':
            return 'hc'  # default H type
        return type_substitutions.get(t, t)

    for i in range(n_atoms):
        atom_i = mol.GetAtomWithIdx(i)
        for bond1 in atom_i.GetBonds():
            j = bond1.GetOtherAtomIdx(i)
            if j <= i:
                continue
            for bond2 in mol.GetAtomWithIdx(j).GetBonds():
                k = bond2.GetOtherAtomIdx(j)
                if k <= j or k == i:
                    continue
                for bond3 in mol.GetAtomWithIdx(k).GetBonds():
                    neighbor_idx = bond3.GetOtherAtomIdx(k)
                    if neighbor_idx <= k or neighbor_idx == j:
                        continue
                    l_idx = neighbor_idx
                    t_i = atom_types[i] if i < len(atom_types) else "x"
                    t_j = atom_types[j] if j < len(atom_types) else "x"
                    t_k = atom_types[k] if k < len(atom_types) else "x"
                    t_l = atom_types[neighbor_idx] if neighbor_idx < len(atom_types) else "x"

                    key = (t_i, t_j, t_k, t_l)

                    # Try exact match first, then with substitutions
                    torsion_params = params['torsions'].get(key, [])
                    if not torsion_params:
                        # Try with substitutions (cx->c3, etc.)
                        key_sub = tuple(_substitute_type(x) for x in key)
                        torsion_params = params['torsions'].get(key_sub, [])

                    torsions.append({
                        'i': i, 'j': j, 'k': k, 'l': l_idx,
                        'types': key,
                        'params': torsion_params,
                    })

    return {
        'atom_types': atom_types,
        'charges': charges,
        'masses': masses,
        'bonds': bonds,
        'angles': angles,
        'torsions': torsions,
    }
