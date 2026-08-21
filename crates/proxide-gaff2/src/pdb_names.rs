//! PDB atom name assignment.
//!
//! Ports `assign_pdb_atom_names` (gaff2.py:1421-1445): assigns unique PDB atom
//! names (element symbol + per-element counter, e.g. `C1`, `O2`, `H3`) to atoms
//! that don't already carry one.
//!
//! Faithful-port note: this preserves a known Python quirk rather than fixing
//! it. Atoms that already carry a name are **not** counted against the
//! per-element counter used for un-named atoms, so a freshly generated name
//! can collide with a pre-existing one (e.g. an atom pre-named `"C1"` plus two
//! un-named carbons both yields `"C1"` for the first un-named carbon too).
//! See `assign_pdb_atom_names` in `gaff2.py` — this is one of the residual
//! mismatches tracked against real AmberTools output and is explicitly out of
//! scope to fix here.

/// Assign unique PDB atom names to a set of atoms.
///
/// Mirrors the Python `assign_pdb_atom_names(rdmol)`, minus the RDKit
/// `AtomPDBResidueInfo` mutation-in-place side effect: RDKit's `Chem.Mol`
/// carries per-atom `MonomerInfo` state directly, so the Python version reads
/// and writes that state on the same object it iterates. This owned Rust
/// representation has no such implicit store, so the pre-existing name for
/// each atom (if any) is passed in explicitly via `existing_names` and the
/// full, ordered set of names is returned — the caller is responsible for
/// writing them back into whatever atom representation it maintains, exactly
/// as the Python caller would read the returned `list[str]` alongside the
/// mutated `rdmol`.
///
/// # Arguments
/// * `elements` - Element symbol per atom (e.g. `"C"`, `"O"`, `"Cl"`), in atom
///   index order. Corresponds to `atom.GetSymbol()` in the Python loop, and to
///   `MolGraph.elements`.
/// * `existing_names` - Pre-existing PDB name per atom, in the same atom
///   order, or `None` if the atom has no monomer info at all. Corresponds to
///   `atom.GetMonomerInfo()` — a name that is present but all-whitespace is
///   treated identically to `None`, exactly as `mi.GetName().strip()` being
///   empty is in Python (both fall through to name generation).
///
/// # Returns
/// One name per atom, in atom index order: the trimmed existing name where
/// present and non-empty, otherwise a freshly generated `"{element}{counter}"`
/// name. The per-element counter is 1-indexed and increments only for atoms
/// that receive a freshly generated name (see the module-level quirk note).
///
/// # Panics
/// If `elements.len() != existing_names.len()`. The Python source has no
/// equivalent failure mode because both facts come from the same RDKit atom
/// object per iteration step; this length check exists only because the two
/// facts are separate parameters here, not because of any behavioral
/// difference from Python.
pub fn assign_pdb_atom_names(
    elements: &[String],
    existing_names: &[Option<String>],
) -> Vec<String> {
    assert_eq!(
        elements.len(),
        existing_names.len(),
        "elements and existing_names must have the same length (one entry per atom)"
    );

    // Iteration order does not matter for this counter -- names are only used
    // for output, never re-looked-up by key (see recon risk notes). A plain
    // HashMap would be fine; a Vec-backed linear counter keeps things
    // dependency-free and is just as simple for the small key set (<20
    // elements) involved here.
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut names: Vec<String> = Vec::with_capacity(elements.len());

    for (elem, existing) in elements.iter().zip(existing_names.iter()) {
        let trimmed = existing.as_deref().map(str::trim).unwrap_or("");
        if !trimmed.is_empty() {
            names.push(trimmed.to_string());
        } else {
            let count = counts.entry(elem.clone()).or_insert(0);
            *count += 1;
            names.push(format!("{elem}{count}"));
        }
    }

    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elems(symbols: &[&str]) -> Vec<String> {
        symbols.iter().map(|s| s.to_string()).collect()
    }

    fn none_names(n: usize) -> Vec<Option<String>> {
        vec![None; n]
    }

    /// No atom has a pre-existing name: plain element+counter assignment,
    /// matching a water molecule + methane-shaped small set of atoms.
    #[test]
    fn all_unnamed_assigns_sequential_counters() {
        let elements = elems(&["C", "C", "O", "H", "H", "H", "H"]);
        let existing = none_names(elements.len());
        let names = assign_pdb_atom_names(&elements, &existing);
        assert_eq!(
            names,
            vec!["C1", "C2", "O1", "H1", "H2", "H3", "H4"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }

    /// A pre-existing non-empty name is kept verbatim (after trimming) and
    /// does NOT participate in that element's counter.
    #[test]
    fn existing_name_is_kept_and_not_counted() {
        let elements = elems(&["C", "C", "O"]);
        let existing = vec![Some("CA".to_string()), None, None];
        let names = assign_pdb_atom_names(&elements, &existing);
        assert_eq!(names, vec!["CA", "C1", "O1"]);
    }

    /// Ported quirk: because a pre-named atom's element isn't counted, a
    /// freshly generated name can collide with a pre-existing one elsewhere
    /// in the same molecule. This is a known residual mismatch against real
    /// AmberTools output and must NOT be "fixed" by this port.
    #[test]
    fn preserves_known_counter_collision_quirk() {
        let elements = elems(&["C", "C"]);
        // Atom 0 already carries the name "C1" (e.g. read from an input PDB).
        // Atom 1 has no name and becomes the *first* un-named carbon, so it
        // also gets "C1" -- a genuine duplicate, faithfully reproduced.
        let existing = vec![Some("C1".to_string()), None];
        let names = assign_pdb_atom_names(&elements, &existing);
        assert_eq!(names, vec!["C1", "C1"]);
    }

    /// An all-whitespace existing name is treated the same as no monomer
    /// info at all (`mi.GetName().strip()` empty falls through in Python).
    #[test]
    fn whitespace_only_existing_name_is_treated_as_absent() {
        let elements = elems(&["N"]);
        let existing = vec![Some("   ".to_string())];
        let names = assign_pdb_atom_names(&elements, &existing);
        assert_eq!(names, vec!["N1"]);
    }

    /// Multi-character element symbols (Cl, Br) concatenate correctly and
    /// are unaffected by counters for a different element.
    #[test]
    fn multi_char_element_symbols() {
        let elements = elems(&["Cl", "Br", "Cl", "C"]);
        let existing = none_names(elements.len());
        let names = assign_pdb_atom_names(&elements, &existing);
        assert_eq!(names, vec!["Cl1", "Br1", "Cl2", "C1"]);
    }

    /// Existing names surrounded by whitespace are trimmed before being kept,
    /// matching `mi.GetName().strip()`.
    #[test]
    fn existing_name_is_trimmed() {
        let elements = elems(&["O"]);
        let existing = vec![Some("  OW  ".to_string())];
        let names = assign_pdb_atom_names(&elements, &existing);
        assert_eq!(names, vec!["OW"]);
    }

    /// Empty input produces empty output.
    #[test]
    fn empty_input() {
        let elements: Vec<String> = vec![];
        let existing: Vec<Option<String>> = vec![];
        let names = assign_pdb_atom_names(&elements, &existing);
        assert!(names.is_empty());
    }

    /// Every atom pre-named: no counters are touched at all, names pass
    /// through unchanged (after trimming).
    #[test]
    fn all_pre_named_passthrough() {
        let elements = elems(&["C", "N", "O"]);
        let existing = vec![
            Some("CA".to_string()),
            Some("N".to_string()),
            Some("O".to_string()),
        ];
        let names = assign_pdb_atom_names(&elements, &existing);
        assert_eq!(names, vec!["CA", "N", "O"]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn mismatched_lengths_panics() {
        let elements = elems(&["C", "O"]);
        let existing = none_names(1);
        let _ = assign_pdb_atom_names(&elements, &existing);
    }
}
