//! PQR file format parser
//!
//! Parses PQR files which contain atom coordinates along with
//! partial charges and radii (used for electrostatics calculations).
//!
//! Format: Similar to PDB but with charge and radius instead of occupancy/B-factor
//! `ATOM serial name resName chainID resSeq x y z charge radius`
//!
//! Sprint 25 (task 260922_autonomous-loop, track a, debt #1919, decisions
//! a-c/l) replaced the original whitespace-split parser -- which silently
//! dropped a line on a short/garbage record (`.ok()?`), zero-filled a
//! garbage `res_seq` (`unwrap_or(0)`), and silently accepted any line with
//! >=11 tokens (ignoring extras) -- with a fail-loud one: every malformed
//! line is a hard, structured [`PqrFieldError`] naming the line, token
//! index, field, and kind, and the caller (`parse_pqr_file`) aborts the
//! whole parse on the first one rather than silently skipping it.

#![allow(dead_code)]

use crate::formats::field_parse::{self, truncate_raw_str, TokenFieldErrorKind, IO_MALFORMED_RECORD_CODE};
use proxide_core::chem::masses::infer_element;
use proxide_core::structure::{AtomRecord, RawAtomData};
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A structured, non-silent PQR parse failure: the line, which
/// whitespace-delimited token (0-based) was involved, the field name, and
/// what specifically was wrong. See the module docs for why this replaces
/// every `.ok()?`/`unwrap_or` silent-drop path the original parser had.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PqrFieldError {
    /// 1-based line number within the file/reader.
    pub line: usize,
    /// 0-based whitespace-delimited token index the problem was found at.
    pub token_index: usize,
    /// The field name, e.g. "serial", "res_seq", "x", "charge".
    pub field: &'static str,
    /// The offending line, truncated (see [`truncate_raw_str`]).
    pub raw: String,
    pub kind: TokenFieldErrorKind,
}

impl PqrFieldError {
    fn new(line: usize, token_index: usize, field: &'static str, raw_line: &str, kind: TokenFieldErrorKind) -> Self {
        Self {
            line,
            token_index,
            field,
            raw: truncate_raw_str(raw_line),
            kind,
        }
    }
}

impl fmt::Display for PqrFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] line {}: field '{}' (token {}): {} -- raw: {:?}",
            IO_MALFORMED_RECORD_CODE, self.line, self.field, self.token_index, self.kind, self.raw
        )
    }
}

impl std::error::Error for PqrFieldError {}

/// Parses PQR/PDB-style residue-sequence-number grammar (decision b): an
/// optional leading `-`, one or more ASCII digits, and at most one trailing
/// ASCII letter (insertion code). Returns `(value, insertion_code)`, where
/// insertion code is `' '` if absent. `None` for anything else -- `"A52"`
/// (leading letter), `""` (empty), `"5-2"` (embedded `-`), `"52AB"` (more
/// than one trailing letter).
fn parse_res_seq_token(s: &str) -> Option<(i32, char)> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let mut i = 0;
    if bytes[0] == b'-' {
        i = 1;
    }
    let digits_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == digits_start {
        return None;
    }
    let digits_end = i;
    let mut icode = ' ';
    if i < bytes.len() && bytes[i].is_ascii_alphabetic() {
        icode = bytes[i] as char;
        i += 1;
    }
    if i != bytes.len() {
        return None;
    }
    let numeric_text = &s[..digits_end];
    // Every byte in `numeric_text` was verified ASCII-digit (+ optional
    // leading '-') by the loop above; the only way `parse` fails here is
    // i32 overflow (too many digits), which correctly propagates as `None`
    // to the caller -- not a silent default, so no `// documented-default:`
    // tag applies to this line.
    let value = match numeric_text.parse::<i32>() {
        Ok(v) => v,
        Err(_) => return None,
    };
    Some((value, icode))
}

/// Decision a: a first token like `HETATM10000`/`ATOM123` merges the record
/// keyword and the serial number together -- previously this failed the
/// exact `"ATOM"`/`"HETATM"` check and was silently ignored like a genuinely
/// unrelated record (`TER`, `END`, ...). Detect that specific shape so it can
/// be reported instead.
fn merged_record_serial(token: &str) -> bool {
    for prefix in ["HETATM", "ATOM"] {
        if let Some(rest) = token.strip_prefix(prefix) {
            if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
                return true;
            }
        }
    }
    false
}

/// Parse a PQR ATOM/HETATM line (decisions a-c, l).
///
/// Returns `Ok(None)` for a line that is not an ATOM/HETATM record at all
/// (e.g. `TER`, `END`, `REMARK`) -- matching `pdb.rs`/`mmcif.rs`'s existing
/// convention of ignoring non-atom records, NOT silently dropping a
/// malformed ATOM/HETATM record (that is now `Err`).
fn parse_pqr_line(line: &str, line_no: usize) -> Result<Option<AtomRecord>, PqrFieldError> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.is_empty() {
        return Ok(None);
    }

    let record_type = tokens[0];
    if record_type != "ATOM" && record_type != "HETATM" {
        if merged_record_serial(record_type) {
            return Err(PqrFieldError::new(
                line_no,
                0,
                "record+serial",
                line,
                TokenFieldErrorKind::TokenCount,
            ));
        }
        // Not an ATOM/HETATM record at all (TER, END, REMARK, ...) --
        // proxide does not parse it, matching today's behaviour.
        return Ok(None);
    }

    let n = tokens.len();
    let (chain_idx, res_seq_idx, coord_start): (Option<usize>, usize, usize) = match n {
        11 => (Some(4), 5, 6),
        10 => {
            // decision a: no chain column. token 3 (res_name) must not look
            // like a residue name fused with the chain letter, and token 4
            // (res_seq) must match the res_seq grammar -- either failure
            // means the file's columns were merged, not that res_seq itself
            // is garbage, so both report TokenCount rather than Unparseable.
            if tokens[3].len() > 3 {
                return Err(PqrFieldError::new(
                    line_no,
                    3,
                    "res_name",
                    line,
                    TokenFieldErrorKind::TokenCount,
                ));
            }
            if parse_res_seq_token(tokens[4]).is_none() {
                return Err(PqrFieldError::new(
                    line_no,
                    4,
                    "res_seq",
                    line,
                    TokenFieldErrorKind::TokenCount,
                ));
            }
            (None, 4, 5)
        }
        _ => {
            return Err(PqrFieldError::new(
                line_no,
                n,
                "<tokens>",
                line,
                TokenFieldErrorKind::TokenCount,
            ));
        }
    };

    let serial = field_parse::parse_decimal_i32(tokens[1])
        .ok_or_else(|| PqrFieldError::new(line_no, 1, "serial", line, TokenFieldErrorKind::Unparseable))?;
    let atom_name = tokens[2].to_string();
    let res_name = tokens[3].to_string();
    let chain_id = chain_idx.map(|i| tokens[i].to_string()).unwrap_or_default(); // documented-default: decision a -- 10-token line has no chain column, chain = ""

    let (res_seq, i_code) = parse_res_seq_token(tokens[res_seq_idx]).ok_or_else(|| {
        PqrFieldError::new(line_no, res_seq_idx, "res_seq", line, TokenFieldErrorKind::Unparseable)
    })?;

    let x = parse_field_f32(tokens[coord_start], line_no, coord_start, "x", line)?;
    let y = parse_field_f32(tokens[coord_start + 1], line_no, coord_start + 1, "y", line)?;
    let z = parse_field_f32(tokens[coord_start + 2], line_no, coord_start + 2, "z", line)?;
    let charge = parse_field_f32(tokens[coord_start + 3], line_no, coord_start + 3, "charge", line)?;
    let radius = parse_field_f32(tokens[coord_start + 4], line_no, coord_start + 4, "radius", line)?;

    // Infer element from atom name. PQR has no dedicated element column at
    // all (unlike PDB's optional columns 77-78) -- this is *always* a
    // name-based inference, so it must be the same two-letter-aware
    // `infer_element` used by mass assignment and (after the pdb.rs fix,
    // backlog #5052 in prolix) the PDB parser's own fallback, not a naive
    // first-character slice. The naive version previously here mis-
    // elementized "CL" (chloride) as "C" (carbon), "NA" (sodium) as "N"
    // (nitrogen), "ZN"/"MG"/"FE"/"CU"/"MN"/"SE" similarly wrong or outright
    // invalid single-letter symbols ("Z", "M"...) -- found during the
    // proxide-brittle-parsing sweep (task 260909_dhfr_gap_tranche2) that
    // also found and fixed the identical pattern in `pdb.rs`.
    let element = infer_element(&atom_name).to_string();

    Ok(Some(AtomRecord {
        serial,
        atom_name,
        alt_loc: ' ', // documented-default: PQR has no alt_loc column at all
        res_name,
        chain_id,
        res_seq,
        i_code,
        x,
        y,
        z,
        occupancy: 1.0,   // documented-default: PQR has no occupancy column at all
        temp_factor: 0.0, // documented-default: PQR has no B-factor column at all
        element,
        charge: Some(charge),
        radius: Some(radius),
        is_hetatm: record_type == "HETATM",
    }))
}

/// decision c: serial/x/y/z/charge/radius unparseable -> Unparseable,
/// nan/inf -> NonFinite, naming the field. No range checks on charge/radius
/// (0.0 radius is legitimate for some force-field hydrogens).
fn parse_field_f32(
    token: &str,
    line_no: usize,
    token_index: usize,
    field: &'static str,
    raw_line: &str,
) -> Result<f32, PqrFieldError> {
    field_parse::parse_finite_f32(token)
        .map_err(|kind| PqrFieldError::new(line_no, token_index, field, raw_line, kind))
}

/// Parse PQR file and return raw atom data with charges and radii. The first
/// malformed ATOM/HETATM line anywhere in the file aborts the whole parse
/// with a [`PqrFieldError`] -- there is no partial result and no silent drop.
pub fn parse_pqr_file<P: AsRef<Path>>(path: P) -> Result<RawAtomData, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut raw_data = RawAtomData::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let line_no = idx + 1;
        if let Some(atom) = parse_pqr_line(&line, line_no)? {
            raw_data.add_atom(atom);
        }
    }

    if raw_data.num_atoms == 0 {
        return Err("No atoms found in PQR file".into());
    }

    Ok(raw_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_one(line: &str) -> Result<Option<AtomRecord>, PqrFieldError> {
        parse_pqr_line(line, 1)
    }

    fn err_kind_field(line: &str) -> (TokenFieldErrorKind, &'static str) {
        let err = parse_one(line).unwrap_err();
        (err.kind, err.field)
    }

    #[test]
    fn test_parse_pqr_line() {
        let line = "ATOM      1  N   MET A   1      20.154  29.699   5.276  -0.4157  1.8240";
        let atom = parse_one(line).unwrap().unwrap();
        assert_eq!(atom.serial, 1);
        assert_eq!(atom.atom_name, "N");
        assert_eq!(atom.res_name, "MET");
        assert_eq!(atom.chain_id, "A");
        assert!((atom.charge.unwrap() - (-0.4157)).abs() < 0.001);
        assert!((atom.radius.unwrap() - 1.8240).abs() < 0.001);
    }

    #[test]
    fn test_parse_hetatm() {
        let line = "HETATM 2242  O   HOH A 301      24.243  16.452  10.158  -0.8340  1.5000";
        let atom = parse_one(line).unwrap().unwrap();
        assert!(atom.is_hetatm);
        assert_eq!(atom.res_name, "HOH");
    }

    #[test]
    fn test_parse_insertion_code() {
        let line = "ATOM      5  N   ALA A  52A     14.000  24.000  34.000  -0.500   1.850";
        let atom = parse_one(line).unwrap().unwrap();
        assert_eq!(atom.res_seq, 52);
        assert_eq!(atom.i_code, 'A');
    }

    #[test]
    fn test_two_letter_element_fallback_not_truncated() {
        // Same anti-pattern as backlog #5052 (prolix)'s `pdb.rs` bug: PQR has
        // no dedicated element column at all, so this is *always* a name-
        // based inference. A naive first-character-only slice previously
        // here mis-elementized "CL" (chloride) as "C" (carbon) and "NA"
        // (sodium) as "N" (nitrogen) -- and would do the same, or produce an
        // outright invalid single-letter symbol, for Br/Mg/Zn/Fe/Cu/Mn/Se.
        let cl_line = "HETATM    1  CL  CL  A 500      12.000   3.000   4.000  -1.0000  1.9350";
        let atom = parse_one(cl_line).unwrap().unwrap();
        assert_eq!(atom.atom_name, "CL");
        assert_eq!(atom.element, "Cl");

        let na_line = "HETATM    2  NA  NA  A 501      15.000   3.000   4.000   1.0000  1.8680";
        let atom = parse_one(na_line).unwrap().unwrap();
        assert_eq!(atom.atom_name, "NA");
        assert_eq!(atom.element, "Na");
    }

    // -------------------------------------------------------------
    // Sprint 25 fail-loud tests (decisions a-c). Every error case asserts
    // BOTH `kind` and `field` via downcast-equivalent direct struct access
    // (no bare `is_err()`).
    // -------------------------------------------------------------

    #[test]
    fn garbage_res_seq_is_unparseable() {
        // 11 tokens (separate chain "A" and res_seq "XXXXX" tokens), so this
        // exercises decision b's general Unparseable path, not decision a's
        // 10-token "merged columns" TokenCount special case.
        let line = "ATOM      1  N   MET A XXXXX      20.154  29.699   5.276  -0.4157  1.8240";
        let (kind, field) = err_kind_field(line);
        assert_eq!(kind, TokenFieldErrorKind::Unparseable);
        assert_eq!(field, "res_seq");
    }

    #[test]
    fn a52_res_seq_is_unparseable() {
        let line = "ATOM      1  N   MET A A52      20.154  29.699   5.276  -0.4157  1.8240";
        let (kind, field) = err_kind_field(line);
        assert_eq!(kind, TokenFieldErrorKind::Unparseable);
        assert_eq!(field, "res_seq");
    }

    #[test]
    fn fifty_two_a_res_seq_parses_ok_with_icode_a() {
        let line = "ATOM      1  N   MET A 52A      20.154  29.699   5.276  -0.4157  1.8240";
        let atom = parse_one(line).unwrap().unwrap();
        assert_eq!(atom.res_seq, 52);
        assert_eq!(atom.i_code, 'A');
    }

    #[test]
    fn bad_x_is_unparseable() {
        let line = "ATOM      1  N   MET A   1      xx.xxx  29.699   5.276  -0.4157  1.8240";
        let (kind, field) = err_kind_field(line);
        assert_eq!(kind, TokenFieldErrorKind::Unparseable);
        assert_eq!(field, "x");
    }

    #[test]
    fn nan_charge_is_nonfinite() {
        let line = "ATOM      1  N   MET A   1      20.154  29.699   5.276  nan  1.8240";
        let (kind, field) = err_kind_field(line);
        assert_eq!(kind, TokenFieldErrorKind::NonFinite);
        assert_eq!(field, "charge");
    }

    #[test]
    fn inf_radius_is_nonfinite() {
        let line = "ATOM      1  N   MET A   1      20.154  29.699   5.276  -0.4157  inf";
        let (kind, field) = err_kind_field(line);
        assert_eq!(kind, TokenFieldErrorKind::NonFinite);
        assert_eq!(field, "radius");
    }

    #[test]
    fn nine_tokens_is_token_count() {
        let line = "ATOM      1  N   MET   1      20.154  29.699   5.276";
        let (kind, field) = err_kind_field(line);
        assert_eq!(kind, TokenFieldErrorKind::TokenCount);
        assert_eq!(field, "<tokens>");
    }

    #[test]
    fn twelve_tokens_is_token_count() {
        // A DELIBERATE change (decision a): today extras are silently
        // ignored; now a 12-token line is a hard TokenCount error.
        let line =
            "ATOM      1  N   MET A   1      20.154  29.699   5.276  -0.4157  1.8240  EXTRA";
        let (kind, field) = err_kind_field(line);
        assert_eq!(kind, TokenFieldErrorKind::TokenCount);
        assert_eq!(field, "<tokens>");
    }

    #[test]
    fn ten_token_no_chain_line_parses_ok_with_empty_chain() {
        let line = "ATOM      1  N   MET   1      20.154  29.699   5.276  -0.4157  1.8240";
        let atom = parse_one(line).unwrap().unwrap();
        assert_eq!(atom.chain_id, "");
        assert_eq!(atom.res_seq, 1);
    }

    #[test]
    fn digit_chain_on_eleven_tokens_parses_ok() {
        // No grammar test on the chain token for an 11-token line -- a
        // digit chain like "1" is accepted as-is.
        let line = "ATOM      1  N   MET 1   52      20.154  29.699   5.276  -0.4157  1.8240";
        let atom = parse_one(line).unwrap().unwrap();
        assert_eq!(atom.chain_id, "1");
        assert_eq!(atom.res_seq, 52);
    }

    #[test]
    fn merged_res_name_and_chain_on_ten_tokens_is_token_count() {
        // "TIP3A" = "TIP3" (a 4-char residue name) fused with chain "A" --
        // decision a: res_name token longer than 3 chars on a 10-token line.
        let line = "ATOM      1  OH2 TIP3A   1      20.154  29.699   5.276  -0.4157  1.8240";
        let (kind, field) = err_kind_field(line);
        assert_eq!(kind, TokenFieldErrorKind::TokenCount);
        assert_eq!(field, "res_name");
    }

    #[test]
    fn merged_chain_and_number_on_ten_tokens_is_token_count() {
        // "A1000" = chain "A" fused with res_seq "1000" -- fails the
        // res_seq grammar on a 10-token line, reported as TokenCount
        // ("merged columns"), not Unparseable.
        let line = "ATOM      1  N   MET A1000      20.154  29.699   5.276  -0.4157  1.8240";
        let (kind, field) = err_kind_field(line);
        assert_eq!(kind, TokenFieldErrorKind::TokenCount);
        assert_eq!(field, "res_seq");
    }

    #[test]
    fn merged_record_and_serial_is_token_count() {
        let line = "HETATM10000  O   HOH A 301      24.243  16.452  10.158  -0.8340  1.5000";
        let err = parse_one(line).unwrap_err();
        assert_eq!(err.kind, TokenFieldErrorKind::TokenCount);
        assert_eq!(err.field, "record+serial");
    }

    #[test]
    fn merged_coordinates_is_token_count() {
        // "-12.345-100.456" merges what should be two separate x/y tokens
        // into one, dropping an otherwise-valid 11-token line to 10 tokens.
        // The 10-token path then reads token[4] (still the chain letter "A"
        // here, since the merge did not touch the chain/res_seq columns) as
        // the would-be res_seq -- "A" fails the res_seq grammar, so this is
        // reported as TokenCount naming "res_seq" (decision a's "merged
        // columns" case), never silently parsed as a 3-coordinate atom.
        let line = "ATOM      1  N   MET A   1      -12.345-100.456   5.276  -0.4157  1.8240";
        let err = parse_one(line).unwrap_err();
        assert_eq!(err.kind, TokenFieldErrorKind::TokenCount);
    }

    #[test]
    fn ignored_non_atom_record_is_ok_none() {
        assert!(parse_one("TER").unwrap().is_none());
        assert!(parse_one("END").unwrap().is_none());
        assert!(parse_one("REMARK  some text").unwrap().is_none());
    }

    #[test]
    fn empty_line_is_ok_none() {
        assert!(parse_one("").unwrap().is_none());
        assert!(parse_one("   ").unwrap().is_none());
    }
}
