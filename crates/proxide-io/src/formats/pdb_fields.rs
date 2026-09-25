//! Shared, fail-loud fixed-column PDB field reader (sprint 24, task
//! 260922_autonomous-loop, track a, debt #1776+#1883, OBS-105 option a).
//!
//! `pdb.rs`'s original `parse_atom_line` silently dropped an atom on a short
//! line, a non-decimal serial/res_seq, or a bad coordinate; it substituted
//! `1.0`/`0.0` for a malformed (not just blank) occupancy/B-factor; and it
//! byte-sliced a `&str` at fixed offsets, which panics if the line contains a
//! multibyte UTF-8 character before column 78 (backlog #1776). This module
//! replaces all of that with one reader that:
//!
//! - reads lines as raw bytes (`BufRead::read_until`), so a fixed-column
//!   slice can never land mid-character and panic (decision f);
//! - requires ATOM/HETATM/MODEL content to be ASCII up to column 78, and
//!   errors with line+column otherwise, rather than silently accepting or
//!   panicking on it (decision f);
//! - treats a non-decimal serial/res_seq, a non-finite coordinate/occupancy/
//!   B-factor, or a too-short ATOM/HETATM line as a hard, structured error
//!   instead of a silent drop or a substituted sentinel (decisions a, c, e);
//! - reads a trailing field (occupancy, B-factor, element) from whatever
//!   partial slice the line actually has, treating it as blank (default)
//!   only when that slice is empty after trimming -- not merely because the
//!   line as a whole is shorter than the field's nominal width (decision c);
//! - errors when a (chain, res_seq, insertion code) key reappears
//!   non-contiguously within one model+chain, which would otherwise silently
//!   merge two distinct residues in `ProcessedStructure::from_raw` (decision
//!   d) -- this is what catches an OpenMM hex/hybrid-36 residue-number
//!   wraparound before it corrupts downstream residue grouping.
//!
//! Every numeric field is parsed independently from its own field text with
//! `str::parse` -- coordinates are parsed as both `f32` (parity with
//! `AtomRecord`, bit-identical with "today") and `f64` (for
//! `proxide-confind`'s `coords.rs`, debt #1885), never by casting one to the
//! other, which can double-round a decimal literal to a different bit
//! pattern than parsing it directly in the target width would.

use proxide_core::chem::masses::infer_element;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::BufRead;

/// Literal error code for this error family. Intended for the OBS-110 error
/// registry once it exists in code (it does not yet -- recon 260924 found no
/// PROX- code registry implemented anywhere); until then this constant is the
/// single source of truth for the code string. Filed as a debt (see the
/// fixer's sprint 24 track a report) to register it once OBS-110 lands.
pub const PDB_MALFORMED_RECORD_CODE: &str = "PROX-IO-MALFORMED-RECORD";

/// What kind of malformation `PdbFieldError` reports.
///
/// `#[non_exhaustive]`: sprint 25 (debt, decision b) is expected to route
/// PQR and mmCIF through this same reader, which may need kinds beyond the
/// six sprint 24 identifies. `Io` is one already-necessary addition beyond
/// sprint 24's decision list -- `parse_pdb_records`'s signature returns
/// `Result<_, PdbFieldError>`, and `BufRead::read_until` can fail at the OS
/// level (not just on malformed content); there is nowhere else to put that
/// error.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PdbFieldErrorKind {
    /// A byte at or before column 78 of an ATOM/HETATM/MODEL line is not
    /// ASCII (decision f).
    NonAscii,
    /// An ATOM/HETATM line is shorter than the 54 bytes needed for the
    /// mandatory serial/name/res/coordinate fields (decision c).
    LineTooShort,
    /// A field's text does not parse as the expected numeric type (decision
    /// a for serial/res_seq; general for coordinates/occupancy/B-factor).
    Unparseable,
    /// A field parsed successfully but is NaN or +/-infinity (decision e).
    NonFinite,
    /// A MODEL record has no numeric token after `MODEL`, or a
    /// non-numeric one.
    InvalidModel,
    /// A (chain, res_seq, insertion code) key reappeared after a different
    /// key was already seen in the same model+chain (decision d).
    ResidueReappears,
    /// An I/O error while reading a line (not a content problem).
    Io,
}

impl fmt::Display for PdbFieldErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PdbFieldErrorKind::NonAscii => "non-ASCII byte",
            PdbFieldErrorKind::LineTooShort => "line too short",
            PdbFieldErrorKind::Unparseable => "unparseable value",
            PdbFieldErrorKind::NonFinite => "non-finite value (NaN/infinity)",
            PdbFieldErrorKind::InvalidModel => "invalid MODEL serial number",
            PdbFieldErrorKind::ResidueReappears => {
                "residue key reappeared non-contiguously in the same model+chain"
            }
            PdbFieldErrorKind::Io => "I/O error",
        };
        f.write_str(s)
    }
}

/// A structured, non-silent PDB parse failure: exactly where in the file, and
/// what specifically was wrong. See the module docs for why this replaces
/// every silent-drop / sentinel-substitution path that `pdb.rs` used to have.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PdbFieldError {
    /// 1-based line number within the file/reader.
    pub line: usize,
    /// The record type text ("ATOM", "HETATM", "MODEL", ...).
    pub record: String,
    /// The field name, e.g. "serial", "x", "occupancy", "res_seq+i_code".
    pub field: &'static str,
    /// 1-based, inclusive PDB column range for `field`.
    pub columns: (usize, usize),
    /// The offending line, lossily decoded and truncated (see
    /// [`truncate_raw`]) so a single malformed line cannot blow up an error
    /// message or log.
    pub raw: String,
    pub kind: PdbFieldErrorKind,
}

/// Truncate a lossily-decoded line to at most 80 characters, appending a
/// `…(+N bytes)` marker naming exactly how many additional raw bytes were not
/// shown -- ledger B9: a truncated error must say that it truncated, not
/// silently present a partial line as if it were the whole one.
fn truncate_raw(bytes: &[u8]) -> String {
    let full = String::from_utf8_lossy(bytes);
    if full.chars().count() <= 80 {
        return full.into_owned();
    }
    let shown: String = full.chars().take(80).collect();
    let shown_bytes = shown.len();
    let remaining = bytes.len().saturating_sub(shown_bytes);
    format!("{shown}\u{2026}(+{remaining} bytes)")
}

impl PdbFieldError {
    fn new(
        line: usize,
        record: &str,
        field: &'static str,
        columns: (usize, usize),
        raw_line: &[u8],
        kind: PdbFieldErrorKind,
    ) -> Self {
        Self {
            line,
            record: record.to_string(),
            field,
            columns,
            raw: truncate_raw(raw_line),
            kind,
        }
    }

    fn io(line: usize, err: &std::io::Error) -> Self {
        Self {
            line,
            record: String::new(),
            field: "<line>",
            columns: (0, 0),
            raw: err.to_string(),
            kind: PdbFieldErrorKind::Io,
        }
    }
}

impl fmt::Display for PdbFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] line {}: {} field '{}' (columns {}-{}): {} -- raw: {:?}",
            PDB_MALFORMED_RECORD_CODE,
            self.line,
            self.record,
            self.field,
            self.columns.0,
            self.columns.1,
            self.kind,
            self.raw
        )
    }
}

impl std::error::Error for PdbFieldError {}

/// One parsed ATOM/HETATM record: every field `pdb.rs`'s original
/// `AtomRecord` carried, plus the source line number and f64-precision
/// coordinates (parsed independently from the same field text, not cast from
/// the f32 value -- see module docs).
#[derive(Debug, Clone)]
pub struct PdbAtomRecord {
    /// 1-based source line number.
    pub line: usize,
    /// MODEL serial number in effect when this atom was read (1 if the file
    /// has no MODEL record, matching `pdb.rs`'s existing default).
    pub model: usize,
    pub serial: i32,
    pub atom_name: String,
    pub alt_loc: char,
    pub res_name: String,
    pub chain_id: String,
    pub res_seq: i32,
    pub i_code: char,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub x64: f64,
    pub y64: f64,
    pub z64: f64,
    pub occupancy: f32,
    pub temp_factor: f32,
    pub element: String,
    pub charge: Option<f32>,
    pub radius: Option<f32>,
    pub is_hetatm: bool,
}

// ---------------------------------------------------------------------
// Byte/str helpers with partial-slice semantics (decision c).
// ---------------------------------------------------------------------

/// `bytes[start..min(end, bytes.len())]`, or `&[]` if `start` is already past
/// the end -- never panics, unlike a bare `bytes[start..end]`.
fn slice_field(bytes: &[u8], start: usize, end: usize) -> &[u8] {
    let len = bytes.len();
    if start >= len {
        return &[];
    }
    &bytes[start..end.min(len)]
}

/// Same clamping for an ASCII-verified `&str` (byte indices == char indices).
fn partial_str(s: &str, start: usize, end: usize) -> &str {
    let len = s.len();
    if start >= len {
        return "";
    }
    &s[start..end.min(len)]
}

fn trim_ascii_bytes(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map_or(start, |i| i + 1);
    &bytes[start..end]
}

/// Verify `line[..min(len,78)]` is ASCII (decision f) and return that prefix
/// as a `&str`. Byte-level slicing means this never panics on a multibyte
/// character the way the original `&str`-based column slicing did (#1776).
fn ascii_checked_prefix<'a>(
    line_bytes: &'a [u8],
    line_no: usize,
    record: &str,
) -> Result<&'a str, PdbFieldError> {
    let check_len = line_bytes.len().min(78);
    let checked = &line_bytes[..check_len];
    if let Some(bad_idx) = checked.iter().position(|b| !b.is_ascii()) {
        return Err(PdbFieldError::new(
            line_no,
            record,
            "<line>",
            (bad_idx + 1, bad_idx + 1),
            line_bytes,
            PdbFieldErrorKind::NonAscii,
        ));
    }
    Ok(std::str::from_utf8(checked).expect("ascii-checked bytes are valid UTF-8"))
}

/// Bundles the three pieces every field-error needs beyond the field's own
/// column range and text -- keeps the parse helpers below under clippy's
/// too-many-arguments threshold without losing any of the diagnostic detail
/// `PdbFieldError` reports.
#[derive(Clone, Copy)]
struct FieldCtx<'a> {
    line_no: usize,
    record: &'a str,
    raw_line: &'a [u8],
}

impl<'a> FieldCtx<'a> {
    fn err(
        &self,
        field: &'static str,
        columns: (usize, usize),
        kind: PdbFieldErrorKind,
    ) -> PdbFieldError {
        PdbFieldError::new(
            self.line_no,
            self.record,
            field,
            columns,
            self.raw_line,
            kind,
        )
    }
}

fn required_i32(
    s: &str,
    start: usize,
    end: usize,
    ctx: FieldCtx,
    field: &'static str,
) -> Result<i32, PdbFieldError> {
    let text = partial_str(s, start, end).trim();
    text.parse::<i32>()
        .map_err(|_| ctx.err(field, (start + 1, end), PdbFieldErrorKind::Unparseable))
}

fn required_f32(
    s: &str,
    start: usize,
    end: usize,
    ctx: FieldCtx,
    field: &'static str,
) -> Result<f32, PdbFieldError> {
    let text = partial_str(s, start, end).trim();
    let value: f32 = text
        .parse()
        .map_err(|_| ctx.err(field, (start + 1, end), PdbFieldErrorKind::Unparseable))?;
    if !value.is_finite() {
        return Err(ctx.err(field, (start + 1, end), PdbFieldErrorKind::NonFinite));
    }
    Ok(value)
}

/// Parses the SAME field text as `required_f32` independently as `f64` --
/// never `value as f64` from the f32 result. Used only for the `x64/y64/z64`
/// companion fields (debt #1885); `AtomRecord`'s own `x/y/z` stay `f32`.
fn required_f64(
    s: &str,
    start: usize,
    end: usize,
    ctx: FieldCtx,
    field: &'static str,
) -> Result<f64, PdbFieldError> {
    let text = partial_str(s, start, end).trim();
    let value: f64 = text
        .parse()
        .map_err(|_| ctx.err(field, (start + 1, end), PdbFieldErrorKind::Unparseable))?;
    if !value.is_finite() {
        return Err(ctx.err(field, (start + 1, end), PdbFieldErrorKind::NonFinite));
    }
    Ok(value)
}

/// Blank-or-value semantics for a trailing float field (occupancy,
/// B-factor): blank ONLY if the partial slice actually available is empty
/// after trimming -- a short line does not, by itself, make the field blank
/// (decision c). A non-blank, unparseable, or non-finite value is a hard
/// error (OBS-105 option a): the caller no longer gets to silently keep a
/// substituted default for a value that was actually present but wrong.
fn optional_f32_blank_default(
    s: &str,
    start: usize,
    end: usize,
    default: f32,
    ctx: FieldCtx,
    field: &'static str,
) -> Result<f32, PdbFieldError> {
    let text = partial_str(s, start, end).trim();
    if text.is_empty() {
        return Ok(default);
    }
    let value: f32 = text
        .parse()
        .map_err(|_| ctx.err(field, (start + 1, end), PdbFieldErrorKind::Unparseable))?;
    if !value.is_finite() {
        return Err(ctx.err(field, (start + 1, end), PdbFieldErrorKind::NonFinite));
    }
    Ok(value)
}

// ---------------------------------------------------------------------
// Residue-reappearance tracking (decision d).
// ---------------------------------------------------------------------

struct ChainTracker {
    last_key: Option<(i32, char)>,
    seen: HashSet<(i32, char)>,
}

fn check_residue_reappearance(
    trackers: &mut HashMap<(usize, String), ChainTracker>,
    model: usize,
    rec: &PdbAtomRecord,
    raw_line: &[u8],
) -> Result<(), PdbFieldError> {
    let key = (rec.res_seq, rec.i_code);
    let tracker = trackers
        .entry((model, rec.chain_id.clone()))
        .or_insert_with(|| ChainTracker {
            last_key: None,
            seen: HashSet::new(),
        });

    if Some(key) == tracker.last_key {
        return Ok(());
    }
    if tracker.seen.contains(&key) {
        return Err(PdbFieldError::new(
            rec.line,
            if rec.is_hetatm { "HETATM" } else { "ATOM" },
            "res_seq+i_code",
            (23, 27),
            raw_line,
            PdbFieldErrorKind::ResidueReappears,
        ));
    }
    if let Some(prev) = tracker.last_key.take() {
        tracker.seen.insert(prev);
    }
    tracker.last_key = Some(key);
    Ok(())
}

// ---------------------------------------------------------------------
// Record parsing.
// ---------------------------------------------------------------------

fn parse_atom_record(
    line_bytes: &[u8],
    line_no: usize,
    model: usize,
    record: &str,
) -> Result<PdbAtomRecord, PdbFieldError> {
    if line_bytes.len() < 54 {
        return Err(PdbFieldError::new(
            line_no,
            record,
            "<line>",
            (1, 54),
            line_bytes,
            PdbFieldErrorKind::LineTooShort,
        ));
    }

    let s = ascii_checked_prefix(line_bytes, line_no, record)?;
    let ctx = FieldCtx {
        line_no,
        record,
        raw_line: line_bytes,
    };

    let serial = required_i32(s, 6, 11, ctx, "serial")?;
    let atom_name = partial_str(s, 12, 16).trim().to_string();
    let alt_loc = s.as_bytes().get(16).copied().unwrap_or(b' ') as char;
    let res_name = partial_str(s, 17, 20).trim().to_string();
    let chain_id = partial_str(s, 21, 22).trim().to_string();
    let res_seq = required_i32(s, 22, 26, ctx, "res_seq")?;
    let i_code = s.as_bytes().get(26).copied().unwrap_or(b' ') as char;

    let x = required_f32(s, 30, 38, ctx, "x")?;
    let y = required_f32(s, 38, 46, ctx, "y")?;
    let z = required_f32(s, 46, 54, ctx, "z")?;
    let x64 = required_f64(s, 30, 38, ctx, "x")?;
    let y64 = required_f64(s, 38, 46, ctx, "y")?;
    let z64 = required_f64(s, 46, 54, ctx, "z")?;

    let occupancy = optional_f32_blank_default(s, 54, 60, 1.0, ctx, "occupancy")?;
    let temp_factor = optional_f32_blank_default(s, 60, 66, 0.0, ctx, "temp_factor")?;

    // Columns 77-78 (0-idx 76..78): partial-slice semantics per decision c --
    // a line that ends mid-way through this field still yields whatever
    // characters are actually there, and blank is decided on THAT slice, not
    // on the line's overall length.
    let element_field = partial_str(s, 76, 78).trim();
    let element = if element_field.is_empty() {
        infer_element(atom_name.trim()).to_string()
    } else {
        element_field.to_string()
    };

    Ok(PdbAtomRecord {
        line: line_no,
        model,
        serial,
        atom_name,
        alt_loc,
        res_name,
        chain_id,
        res_seq,
        i_code,
        x,
        y,
        z,
        x64,
        y64,
        z64,
        occupancy,
        temp_factor,
        element,
        charge: None,
        radius: None,
        is_hetatm: record == "HETATM",
    })
}

fn parse_model_number(line_bytes: &[u8], line_no: usize) -> Result<usize, PdbFieldError> {
    let s = ascii_checked_prefix(line_bytes, line_no, "MODEL")?;
    // Accept both the spec column layout and non-standard single-space
    // separation ("MODEL 1"): take the first whitespace-delimited token
    // after the "MODEL" keyword itself, wherever it falls.
    let rest = partial_str(s, 5, s.len()).trim_start();
    let token = rest.split_whitespace().next();
    match token.and_then(|t| t.parse::<usize>().ok()) {
        Some(n) => Ok(n),
        None => Err(PdbFieldError::new(
            line_no,
            "MODEL",
            "model_serial",
            (11, 14),
            line_bytes,
            PdbFieldErrorKind::InvalidModel,
        )),
    }
}

/// Parse every ATOM/HETATM record from `reader`, in order, tracking MODEL
/// boundaries and residue-reappearance (decision d) as it goes. The first
/// malformed record anywhere in the file aborts the whole parse with a
/// [`PdbFieldError`] naming exactly where and what -- there is no partial
/// result and no silent drop.
pub fn parse_pdb_records<R: BufRead>(mut reader: R) -> Result<Vec<PdbAtomRecord>, PdbFieldError> {
    let mut records = Vec::new();
    let mut current_model: usize = 1;
    let mut trackers: HashMap<(usize, String), ChainTracker> = HashMap::new();
    let mut line_no: usize = 0;
    let mut buf: Vec<u8> = Vec::new();

    loop {
        buf.clear();
        let n = reader
            .read_until(b'\n', &mut buf)
            .map_err(|e| PdbFieldError::io(line_no + 1, &e))?;
        if n == 0 {
            break;
        }
        line_no += 1;

        while matches!(buf.last(), Some(b'\n') | Some(b'\r')) {
            buf.pop();
        }
        let line_bytes: &[u8] = &buf;

        let record_field = slice_field(line_bytes, 0, 6);
        let record_trimmed = trim_ascii_bytes(record_field);
        let record_str = String::from_utf8_lossy(record_trimmed);

        match record_str.as_ref() {
            "ATOM" | "HETATM" => {
                let rec =
                    parse_atom_record(line_bytes, line_no, current_model, record_str.as_ref())?;
                check_residue_reappearance(&mut trackers, current_model, &rec, line_bytes)?;
                records.push(rec);
            }
            "MODEL" => {
                current_model = parse_model_number(line_bytes, line_no)?;
            }
            "ENDMDL" | "TER" | "ANISOU" => {
                // Explicitly ignored, matching pdb.rs's existing behaviour.
            }
            _ => {
                // Any other record (REMARK, HEADER, CRYST1, AUTHOR, ...):
                // proxide does not parse it, so its bytes are never
                // interpreted -- non-UTF-8/non-ASCII content there is fine
                // (decision f).
            }
        }
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn records(text: &str) -> Result<Vec<PdbAtomRecord>, PdbFieldError> {
        parse_pdb_records(text.as_bytes())
    }

    /// A known-good 80-column ATOM line to mutate field-by-field in tests,
    /// so every test's column math is derived from the same real line
    /// instead of hand-aligned (and easily miscounted) literals.
    const BASE_LINE: &str =
        "ATOM      1  N   MET A   1      20.154  29.699   5.276  1.00 49.05           N  ";

    /// Overwrite `line[start..end]` (0-based, half-open, matching this
    /// module's own column convention) with `value`, right-padded/truncated
    /// to fit -- mirrors fixed-column PDB field layout so the rest of the
    /// line's columns stay put.
    fn set_field(line: &mut Vec<u8>, start: usize, end: usize, value: &str) {
        let width = end - start;
        let mut field = value.as_bytes().to_vec();
        field.truncate(width);
        while field.len() < width {
            field.push(b' ');
        }
        line[start..end].copy_from_slice(&field);
    }

    fn mutated(start: usize, end: usize, value: &str) -> Vec<u8> {
        let mut line = BASE_LINE.as_bytes().to_vec();
        set_field(&mut line, start, end, value);
        line
    }

    #[test]
    fn valid_atom_line_parses() {
        let recs = records(BASE_LINE).unwrap();
        assert_eq!(recs.len(), 1);
        let r = &recs[0];
        assert_eq!(r.serial, 1);
        assert_eq!(r.atom_name, "N");
        assert_eq!(r.res_name, "MET");
        assert_eq!(r.chain_id, "A");
        assert_eq!(r.res_seq, 1);
        assert_eq!(r.element, "N");
        assert!((r.x - 20.154).abs() < 1e-3);
        assert!((r.x64 - 20.154).abs() < 1e-9);
        assert_eq!(r.model, 1);
    }

    #[test]
    fn short_line_is_line_too_short() {
        let err = records("ATOM").unwrap_err();
        assert_eq!(err.kind, PdbFieldErrorKind::LineTooShort);
    }

    #[test]
    fn bare_atom_line_is_line_too_short() {
        let err = records("ATOM\n").unwrap_err();
        assert_eq!(err.kind, PdbFieldErrorKind::LineTooShort);
    }

    #[test]
    fn bad_serial_is_unparseable_named_field() {
        let line = mutated(6, 11, "XXXXX");
        let err = parse_pdb_records(&line[..]).unwrap_err();
        assert_eq!(err.kind, PdbFieldErrorKind::Unparseable);
        assert_eq!(err.field, "serial");
    }

    #[test]
    fn bad_x_is_unparseable_named_field() {
        let line = mutated(30, 38, "xx.xxx");
        let err = parse_pdb_records(&line[..]).unwrap_err();
        assert_eq!(err.kind, PdbFieldErrorKind::Unparseable);
        assert_eq!(err.field, "x");
    }

    #[test]
    fn nan_x_is_nonfinite() {
        let line = mutated(30, 38, "nan");
        let err = parse_pdb_records(&line[..]).unwrap_err();
        assert_eq!(err.kind, PdbFieldErrorKind::NonFinite);
        assert_eq!(err.field, "x");
    }

    #[test]
    fn nan_b_factor_is_nonfinite() {
        let line = mutated(60, 66, "nan");
        let err = parse_pdb_records(&line[..]).unwrap_err();
        assert_eq!(err.kind, PdbFieldErrorKind::NonFinite);
        assert_eq!(err.field, "temp_factor");
    }

    #[test]
    fn malformed_occupancy_is_unparseable() {
        // OBS-105 option a, named test 1/2 (fixer prompt Step 4).
        let line = mutated(54, 60, "x.xx");
        let err = parse_pdb_records(&line[..]).unwrap_err();
        assert_eq!(err.kind, PdbFieldErrorKind::Unparseable);
        assert_eq!(err.field, "occupancy");
    }

    #[test]
    fn blank_occupancy_takes_documented_default() {
        // OBS-105 option a, named test 2/2.
        let line = mutated(54, 60, "");
        let recs = parse_pdb_records(&line[..]).unwrap();
        assert_eq!(recs[0].occupancy, 1.0);
    }

    #[test]
    fn non_ascii_atom_name_errors() {
        let mut line = BASE_LINE.as_bytes().to_vec();
        // Overwrite the atom-name field (cols 13-16, 0-idx 12..16) with an
        // invalid UTF-8 byte followed by '(' -- previously this byte range
        // was sliced as `&str` and would panic; now it's a structured error.
        line[12] = 0xC3;
        line[13] = 0x28;
        let err = parse_pdb_records(&line[..]).unwrap_err();
        assert_eq!(err.kind, PdbFieldErrorKind::NonAscii);
    }

    #[test]
    fn latin1_remark_line_is_accepted() {
        // A REMARK line proxide never parses may contain arbitrary non-UTF-8
        // bytes (e.g. Latin-1 e-acute, 0xE9) without erroring -- only
        // ATOM/HETATM/MODEL content is ASCII-checked (decision f).
        let mut content = b"REMARK 1 R\xe9sum\xe9 of structure\n".to_vec();
        content.extend_from_slice(BASE_LINE.as_bytes());
        content.push(b'\n');
        let recs = parse_pdb_records(&content[..]).unwrap();
        assert_eq!(recs.len(), 1);
    }

    #[test]
    fn hex_serial_is_unparseable() {
        let line = mutated(6, 11, "A0000");
        let err = parse_pdb_records(&line[..]).unwrap_err();
        assert_eq!(err.kind, PdbFieldErrorKind::Unparseable);
        assert_eq!(err.field, "serial");
    }

    #[test]
    fn hex_res_seq_is_unparseable() {
        let line = mutated(22, 26, "A000");
        let err = parse_pdb_records(&line[..]).unwrap_err();
        assert_eq!(err.kind, PdbFieldErrorKind::Unparseable);
        assert_eq!(err.field, "res_seq");
    }

    #[test]
    fn wraparound_duplicate_residue_errors() {
        let content = "\
ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00           N
ATOM      2  N   ALA A   2      1.000   0.000   0.000  1.00  0.00           N
ATOM      3  N   ALA A   1      2.000   0.000   0.000  1.00  0.00           N
";
        let err = records(content).unwrap_err();
        assert_eq!(err.kind, PdbFieldErrorKind::ResidueReappears);
    }

    #[test]
    fn contiguous_same_residue_is_fine() {
        let content = "\
ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      1.000   0.000   0.000  1.00  0.00           C
ATOM      3  N   ALA A   2      2.000   0.000   0.000  1.00  0.00           N
";
        let recs = records(content).unwrap();
        assert_eq!(recs.len(), 3);
    }

    #[test]
    fn model_with_single_space_is_ok() {
        let content = format!("MODEL 1\n{BASE_LINE}\nENDMDL\n");
        let recs = records(&content).unwrap();
        assert_eq!(recs[0].model, 1);
    }

    #[test]
    fn model_non_numeric_is_invalid_model() {
        let content = format!("MODEL x\n{BASE_LINE}\n");
        let err = records(&content).unwrap_err();
        assert_eq!(err.kind, PdbFieldErrorKind::InvalidModel);
    }

    #[test]
    fn length_77_element_line_is_read_not_ignored() {
        // Columns 1-77 present (element column has only its first char);
        // decision c: read from the partial slice, don't treat as blank
        // just because the line as a whole is short. BASE_LINE's element
        // column (0-idx 76..78) is "N " -- truncate to 77 bytes so only the
        // 'N' at index 76 remains.
        let mut line = BASE_LINE.as_bytes().to_vec();
        line.truncate(77);
        assert_eq!(line.len(), 77);
        let recs = parse_pdb_records(&line[..]).unwrap();
        assert_eq!(recs[0].element, "N");
    }

    #[test]
    fn truncation_marker_present_for_long_raw_line() {
        let mut line = mutated(30, 38, "xx.xxx");
        line.extend_from_slice(&b"X".repeat(50));
        let err = parse_pdb_records(&line[..]).unwrap_err();
        assert!(err.raw.contains('\u{2026}'), "raw: {}", err.raw);
        assert!(err.raw.contains("bytes)"), "raw: {}", err.raw);
    }

    #[test]
    fn display_includes_literal_error_code() {
        let err = records("ATOM").unwrap_err();
        let text = err.to_string();
        assert!(text.contains(PDB_MALFORMED_RECORD_CODE));
        assert_eq!(PDB_MALFORMED_RECORD_CODE, "PROX-IO-MALFORMED-RECORD");
    }
}
