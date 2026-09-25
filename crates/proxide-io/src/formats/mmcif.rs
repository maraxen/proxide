//! mmCIF/PDBx file format parser
//!
//! High-performance parser for macromolecular Crystallographic Information File (mmCIF) format.
//! mmCIF is the primary deposition format for the Protein Data Bank.
//!
//! Format reference: https://mmcif.wwpdb.org/
//!
//! Sprint 25 (task 260922_autonomous-loop, track b, debt #1920, decisions
//! d-g/h/l/m) replaced the original line-at-a-time state machine -- which
//! tokenized ONE line at a time (so a legally wrapped row or a `;`
//! multi-line text field was silently skipped via
//! `if values.len() != column_names.len() { continue; }`), opened quotes
//! mid-token, silently dropped the rest of a line on an unterminated quote,
//! matched `_atom_site` as a PREFIX (so `_atom_sites.` / `_atom_site_anisotrop.`
//! were misread as the same category), collapsed CIF `.`/`?` into the same
//! empty string even for required fields, and zero-filled/defaulted every
//! numeric field on a parse failure -- with a fail-loud, line-anchored
//! tokenizer and row assembler. Every malformed row is now a hard, structured
//! [`CifFieldError`] naming the row's start line, the mmCIF column, the
//! `AtomRecord` field, and what precisely was wrong; `parse_mmcif_from_reader`
//! aborts the whole parse on the first one rather than silently skipping it.

use crate::formats::field_parse::{
    self, truncate_raw_str, TokenFieldErrorKind, IO_MALFORMED_RECORD_CODE,
};
use proxide_core::chem::masses::infer_element;
use proxide_core::structure::{AtomRecord, RawAtomData};
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A structured, non-silent mmCIF parse failure: the row's start line, the
/// mmCIF column name involved (dynamic -- it may be a `label_*` or `auth_*`
/// variant depending on decision f's per-column source resolution, or a
/// placeholder like `"<row>"` for a tokenizer/row-assembly failure that
/// is not about any one field), the `AtomRecord` field it maps to, the raw
/// value (see [`truncate_raw_str`]), and the [`TokenFieldErrorKind`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CifFieldError {
    /// 1-based line number the offending row started on (decision d). For a
    /// tokenizer-level failure (unterminated quote, a too-long row) this is
    /// the line the problem was actually detected on, which is documented
    /// per call site below -- decision d distinguishes "the row's start
    /// line" (a boundary/EOF with a row pending) from "THAT line" (a line
    /// that pushes a row over its expected token count).
    pub row_start_line: usize,
    /// The mmCIF column name (e.g. `"Cartn_x"`, `"auth_atom_id"`), or a
    /// `"<...>"` placeholder for a non-field (tokenizer/structural) error.
    pub column: String,
    /// The `AtomRecord` field this maps to (e.g. `"x"`, `"atom_name"`).
    pub field: &'static str,
    /// The offending raw value, truncated (see [`truncate_raw_str`]).
    pub raw: String,
    pub kind: TokenFieldErrorKind,
}

impl CifFieldError {
    fn new(
        row_start_line: usize,
        column: impl Into<String>,
        field: &'static str,
        raw: &str,
        kind: TokenFieldErrorKind,
    ) -> Self {
        Self {
            row_start_line,
            column: column.into(),
            field,
            raw: truncate_raw_str(raw),
            kind,
        }
    }
}

impl fmt::Display for CifFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] row starting at line {}: column '{}' (field '{}'): {} -- raw: {:?}",
            IO_MALFORMED_RECORD_CODE,
            self.row_start_line,
            self.column,
            self.field,
            self.kind,
            self.raw
        )
    }
}

impl std::error::Error for CifFieldError {}

/// A single mmCIF data value, distinguishing "genuinely present text" from
/// the two CIF placeholder tokens `.` (Inapplicable) and `?` (Unknown) --
/// decision e. A QUOTED `.` or `?` (single/double quote, or a `;` text
/// field) is always [`CifValue::Present`]: quoting is how CIF says "this is
/// literal text, not the placeholder".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CifValue<'a> {
    Present(&'a str),
    Inapplicable,
    Unknown,
}

fn classify_value(text: &str, quoted: bool) -> CifValue<'_> {
    if !quoted {
        if text == "." {
            return CifValue::Inapplicable;
        }
        if text == "?" {
            return CifValue::Unknown;
        }
    }
    CifValue::Present(text)
}

fn cif_value<'a>(
    tokens: &'a [(String, bool)],
    column_map: &HashMap<String, usize>,
    column: &str,
) -> Option<CifValue<'a>> {
    let idx = *column_map.get(column)?;
    let (text, quoted) = tokens.get(idx)?;
    Some(classify_value(text.as_str(), *quoted))
}

// ---------------------------------------------------------------------
// Tokenizer (decision d): quotes open ONLY at token start; a quote closes
// on the same delimiter followed by whitespace/EOL (never mid-token); an
// unquoted token runs to the next whitespace, so `O5'` (no leading quote)
// is one literal token including the apostrophe.
// ---------------------------------------------------------------------

/// Tokenize a single physical line (or a `;`-text-field's closing-line
/// remainder) into `(text, quoted)` pairs. `Err(())` if a quote is opened
/// but never closed before the line ends -- the caller attaches the line
/// number and raw context to build an [`UnterminatedQuote`] error.
///
/// [`UnterminatedQuote`]: TokenFieldErrorKind::UnterminatedQuote
fn tokenize_line(line: &str) -> Result<Vec<(String, bool)>, ()> {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut tokens = Vec::new();

    while i < n {
        while i < n && (chars[i] == ' ' || chars[i] == '\t') {
            i += 1;
        }
        if i >= n {
            break;
        }
        let c = chars[i];
        if c == '\'' || c == '"' {
            let quote = c;
            let start = i + 1;
            let mut j = start;
            let mut end = None;
            while j < n {
                if chars[j] == quote {
                    let next_is_ws_or_eol =
                        j + 1 >= n || chars[j + 1] == ' ' || chars[j + 1] == '\t';
                    if next_is_ws_or_eol {
                        end = Some(j);
                        break;
                    }
                }
                j += 1;
            }
            let Some(end) = end else {
                return Err(());
            };
            let text: String = chars[start..end].iter().collect();
            tokens.push((text, true));
            i = end + 1;
        } else {
            let start = i;
            while i < n && chars[i] != ' ' && chars[i] != '\t' {
                i += 1;
            }
            let text: String = chars[start..i].iter().collect();
            tokens.push((text, false));
        }
    }

    Ok(tokens)
}

// ---------------------------------------------------------------------
// Structural / row-level error builders
// ---------------------------------------------------------------------

/// decision d: an incomplete row hitting a boundary/EOF is reported at the
/// row's START line; a row that exceeds the expected column count is
/// reported at the line where the overflow was detected. Both call sites
/// pass the correct `line_no` for their case -- this helper does not decide
/// which one to use.
fn row_length_error(line_no: usize, actual_tokens: usize, expected: usize) -> CifFieldError {
    CifFieldError::new(
        line_no,
        "<row>",
        "<tokens>",
        &format!("{actual_tokens} tokens (expected {expected})"),
        TokenFieldErrorKind::RowLength,
    )
}

fn unterminated_quote_error(line_no: usize, context: &str) -> CifFieldError {
    CifFieldError::new(
        line_no,
        "<quote>",
        "<token>",
        context,
        TokenFieldErrorKind::UnterminatedQuote,
    )
}

fn non_loop_error(line_no: usize, tag: &str) -> CifFieldError {
    CifFieldError::new(
        line_no,
        tag.to_string(),
        "<atom_site>",
        tag,
        TokenFieldErrorKind::NonLoopUnsupported,
    )
}

fn multiple_data_blocks_error(line_no: usize, data_block_index: usize) -> CifFieldError {
    CifFieldError::new(
        line_no,
        "_atom_site",
        "<data_block>",
        &format!("data block #{data_block_index}"),
        TokenFieldErrorKind::MultipleDataBlocks,
    )
}

// ---------------------------------------------------------------------
// Per-column label/auth source resolution (decision f)
// ---------------------------------------------------------------------

enum SeqMode {
    /// `label_seq_id` present, no `auth_seq_id` column to fall back to.
    LabelOnly,
    /// No `label_seq_id` column; `auth_seq_id` used directly (the
    /// "auth-only header" case).
    AuthOnly,
    /// Both present: `label_seq_id` is the source; a PER-ROW `.` falls back
    /// to `auth_seq_id`, a `?` is a hard [`TokenFieldErrorKind::Unknown`].
    LabelWithAuthFallback,
}

struct ColumnSources {
    atom_id: String,
    comp_id: String,
    asym_id: String,
    seq_mode: SeqMode,
}

fn resolve_id_col(
    column_map: &HashMap<String, usize>,
    base: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let label = format!("label_{base}");
    if column_map.contains_key(&label) {
        return Ok(label);
    }
    let auth = format!("auth_{base}");
    if column_map.contains_key(&auth) {
        return Ok(auth);
    }
    Err(format!(
        "mmCIF _atom_site loop is missing a required column: need label_{base} or auth_{base}"
    )
    .into())
}

fn resolve_column_sources(
    column_map: &HashMap<String, usize>,
) -> Result<ColumnSources, Box<dyn std::error::Error>> {
    let atom_id = resolve_id_col(column_map, "atom_id")?;
    let comp_id = resolve_id_col(column_map, "comp_id")?;
    let asym_id = resolve_id_col(column_map, "asym_id")?;
    let has_label_seq = column_map.contains_key("label_seq_id");
    let has_auth_seq = column_map.contains_key("auth_seq_id");
    let seq_mode = match (has_label_seq, has_auth_seq) {
        (true, true) => SeqMode::LabelWithAuthFallback,
        (true, false) => SeqMode::LabelOnly,
        (false, true) => SeqMode::AuthOnly,
        (false, false) => return Err(
            "mmCIF _atom_site loop is missing a required column: need label_seq_id or auth_seq_id"
                .into(),
        ),
    };
    Ok(ColumnSources {
        atom_id,
        comp_id,
        asym_id,
        seq_mode,
    })
}

// ---------------------------------------------------------------------
// Field-level readers (decisions e-g, l)
// ---------------------------------------------------------------------

fn required_i32(
    tokens: &[(String, bool)],
    column_map: &HashMap<String, usize>,
    column: &str,
    field: &'static str,
    row_start_line: usize,
) -> Result<i32, CifFieldError> {
    match cif_value(tokens, column_map, column) {
        None => Err(CifFieldError::new(
            row_start_line,
            column,
            field,
            "<column missing>",
            TokenFieldErrorKind::Unparseable,
        )),
        Some(CifValue::Inapplicable) => Err(CifFieldError::new(
            row_start_line,
            column,
            field,
            ".",
            TokenFieldErrorKind::Inapplicable,
        )),
        Some(CifValue::Unknown) => Err(CifFieldError::new(
            row_start_line,
            column,
            field,
            "?",
            TokenFieldErrorKind::Unknown,
        )),
        Some(CifValue::Present(s)) => field_parse::parse_decimal_i32(s).ok_or_else(|| {
            CifFieldError::new(
                row_start_line,
                column,
                field,
                s,
                TokenFieldErrorKind::Unparseable,
            )
        }),
    }
}

fn required_f32(
    tokens: &[(String, bool)],
    column_map: &HashMap<String, usize>,
    column: &str,
    field: &'static str,
    row_start_line: usize,
) -> Result<f32, CifFieldError> {
    match cif_value(tokens, column_map, column) {
        None => Err(CifFieldError::new(
            row_start_line,
            column,
            field,
            "<column missing>",
            TokenFieldErrorKind::Unparseable,
        )),
        Some(CifValue::Inapplicable) => Err(CifFieldError::new(
            row_start_line,
            column,
            field,
            ".",
            TokenFieldErrorKind::Inapplicable,
        )),
        Some(CifValue::Unknown) => Err(CifFieldError::new(
            row_start_line,
            column,
            field,
            "?",
            TokenFieldErrorKind::Unknown,
        )),
        Some(CifValue::Present(s)) => field_parse::parse_finite_f32(s)
            .map_err(|kind| CifFieldError::new(row_start_line, column, field, s, kind)),
    }
}

fn required_str_from(
    tokens: &[(String, bool)],
    column_map: &HashMap<String, usize>,
    column: &str,
    field: &'static str,
    row_start_line: usize,
) -> Result<String, CifFieldError> {
    match cif_value(tokens, column_map, column) {
        None => Err(CifFieldError::new(
            row_start_line,
            column,
            field,
            "<column missing>",
            TokenFieldErrorKind::Unparseable,
        )),
        Some(CifValue::Inapplicable) => Err(CifFieldError::new(
            row_start_line,
            column,
            field,
            ".",
            TokenFieldErrorKind::Inapplicable,
        )),
        Some(CifValue::Unknown) => Err(CifFieldError::new(
            row_start_line,
            column,
            field,
            "?",
            TokenFieldErrorKind::Unknown,
        )),
        Some(CifValue::Present(s)) => Ok(s.to_string()),
    }
}

/// decision f: `label_seq_id` is the source; ONLY an Inapplicable (`.`)
/// falls back per row to `auth_seq_id` (when that column exists) -- an
/// Unknown (`?`) is always a hard error, never a silent auth borrow.
fn read_seq_id(
    tokens: &[(String, bool)],
    column_map: &HashMap<String, usize>,
    seq_mode: &SeqMode,
    row_start_line: usize,
) -> Result<i32, CifFieldError> {
    match seq_mode {
        SeqMode::LabelOnly => required_i32(
            tokens,
            column_map,
            "label_seq_id",
            "res_seq",
            row_start_line,
        ),
        SeqMode::AuthOnly => {
            required_i32(tokens, column_map, "auth_seq_id", "res_seq", row_start_line)
        }
        SeqMode::LabelWithAuthFallback => {
            match cif_value(tokens, column_map, "label_seq_id") {
                None => required_i32(
                    tokens,
                    column_map,
                    "label_seq_id",
                    "res_seq",
                    row_start_line,
                ),
                Some(CifValue::Unknown) => Err(CifFieldError::new(
                    row_start_line,
                    "label_seq_id",
                    "res_seq",
                    "?",
                    TokenFieldErrorKind::Unknown,
                )),
                Some(CifValue::Inapplicable) => {
                    // Per-row fallback to auth_seq_id (the RCSB HETATM/HOH case).
                    required_i32(tokens, column_map, "auth_seq_id", "res_seq", row_start_line)
                }
                Some(CifValue::Present(s)) => field_parse::parse_decimal_i32(s).ok_or_else(|| {
                    CifFieldError::new(
                        row_start_line,
                        "label_seq_id",
                        "res_seq",
                        s,
                        TokenFieldErrorKind::Unparseable,
                    )
                }),
            }
        }
    }
}

/// decision g: present but unparseable, `.`, or `?` -> InvalidModel. Absent
/// column -> keep the previous model number (documented-default: mmCIF rows
/// without a `pdbx_PDB_model_num` column carry no model information at all,
/// so there is nothing to update).
fn read_model_num(
    tokens: &[(String, bool)],
    column_map: &HashMap<String, usize>,
    row_start_line: usize,
    current_model: i32,
) -> Result<i32, CifFieldError> {
    match cif_value(tokens, column_map, "pdbx_PDB_model_num") {
        None => Ok(current_model), // documented-default: column absent -> keep previous model number
        Some(CifValue::Inapplicable) => Err(CifFieldError::new(
            row_start_line,
            "pdbx_PDB_model_num",
            "model_num",
            ".",
            TokenFieldErrorKind::InvalidModel,
        )),
        Some(CifValue::Unknown) => Err(CifFieldError::new(
            row_start_line,
            "pdbx_PDB_model_num",
            "model_num",
            "?",
            TokenFieldErrorKind::InvalidModel,
        )),
        Some(CifValue::Present(s)) => {
            let v = field_parse::parse_decimal_i32(s).ok_or_else(|| {
                CifFieldError::new(
                    row_start_line,
                    "pdbx_PDB_model_num",
                    "model_num",
                    s,
                    TokenFieldErrorKind::InvalidModel,
                )
            })?;
            if v < 0 {
                return Err(CifFieldError::new(
                    row_start_line,
                    "pdbx_PDB_model_num",
                    "model_num",
                    s,
                    TokenFieldErrorKind::InvalidModel,
                ));
            }
            Ok(v)
        }
    }
}

/// decision l: `group_PDB` present must be `ATOM` or `HETATM`, else
/// `BadGroup`; absent -> `ATOM` (today's behaviour). Returns `is_hetatm`.
fn read_group_pdb(
    tokens: &[(String, bool)],
    column_map: &HashMap<String, usize>,
    row_start_line: usize,
) -> Result<bool, CifFieldError> {
    match cif_value(tokens, column_map, "group_PDB") {
        None => Ok(false), // documented-default: group_PDB column absent -> ATOM
        Some(CifValue::Present("ATOM")) => Ok(false),
        Some(CifValue::Present("HETATM")) => Ok(true),
        Some(CifValue::Present(s)) => Err(CifFieldError::new(
            row_start_line,
            "group_PDB",
            "group_pdb",
            s,
            TokenFieldErrorKind::BadGroup,
        )),
        Some(CifValue::Inapplicable) => Err(CifFieldError::new(
            row_start_line,
            "group_PDB",
            "group_pdb",
            ".",
            TokenFieldErrorKind::BadGroup,
        )),
        Some(CifValue::Unknown) => Err(CifFieldError::new(
            row_start_line,
            "group_PDB",
            "group_pdb",
            "?",
            TokenFieldErrorKind::BadGroup,
        )),
    }
}

/// decision l: occupancy/B_iso_or_equiv absent, `.`, or `?` -> the
/// documented default; present-but-unparseable -> Unparseable; nan/inf ->
/// NonFinite.
fn read_occ_or_b(
    tokens: &[(String, bool)],
    column_map: &HashMap<String, usize>,
    column: &str,
    field: &'static str,
    row_start_line: usize,
    default: f32,
) -> Result<f32, CifFieldError> {
    match cif_value(tokens, column_map, column) {
        None => Ok(default), // documented-default: column absent -> occupancy 1.0 / B 0.0 (decision l)
        Some(CifValue::Inapplicable) | Some(CifValue::Unknown) => Ok(default), // documented-default: '.'/'?' -> occupancy 1.0 / B 0.0 (decision l)
        Some(CifValue::Present(s)) => field_parse::parse_finite_f32(s)
            .map_err(|kind| CifFieldError::new(row_start_line, column, field, s, kind)),
    }
}

/// decision e: `label_alt_id`/`pdbx_PDB_ins_code` are optional -- both `.`
/// and `?` (and an absent column) mean "not specified" (blank). decision g:
/// a present value longer than one character is `Unparseable` (previously
/// silently truncated by `chars().next()`).
fn read_optional_char(
    tokens: &[(String, bool)],
    column_map: &HashMap<String, usize>,
    column: &str,
    field: &'static str,
    row_start_line: usize,
) -> Result<char, CifFieldError> {
    match cif_value(tokens, column_map, column) {
        None => Ok(' '), // documented-default: column absent -> blank
        Some(CifValue::Inapplicable) | Some(CifValue::Unknown) => Ok(' '), // documented-default: '.'/'?' both mean "not specified" (decision e)
        Some(CifValue::Present("")) => Ok(' '), // documented-default: quoted empty string -> blank
        Some(CifValue::Present(s)) => {
            let mut chars = s.chars();
            let c = chars.next().expect("non-empty checked above");
            if chars.next().is_none() {
                Ok(c)
            } else {
                Err(CifFieldError::new(
                    row_start_line,
                    column,
                    field,
                    s,
                    TokenFieldErrorKind::Unparseable,
                ))
            }
        }
    }
}

/// decision e: `type_symbol` is optional; absent, `.`, or `?` -> empty
/// string (the caller falls back to name-based `infer_element`, same as
/// today).
fn read_optional_str(
    tokens: &[(String, bool)],
    column_map: &HashMap<String, usize>,
    column: &str,
) -> String {
    match cif_value(tokens, column_map, column) {
        None | Some(CifValue::Inapplicable) | Some(CifValue::Unknown) => String::new(), // documented-default: absent/'.'/'?' -> "" (infer_element fallback)
        Some(CifValue::Present(s)) => s.to_string(),
    }
}

fn extract_row(
    tokens: &[(String, bool)],
    column_map: &HashMap<String, usize>,
    sources: &ColumnSources,
    row_start_line: usize,
    current_model: i32,
) -> Result<(AtomRecord, i32), CifFieldError> {
    let new_model = read_model_num(tokens, column_map, row_start_line, current_model)?;
    let is_hetatm = read_group_pdb(tokens, column_map, row_start_line)?;
    let serial = required_i32(tokens, column_map, "id", "serial", row_start_line)?;
    let atom_name = required_str_from(
        tokens,
        column_map,
        &sources.atom_id,
        "atom_name",
        row_start_line,
    )?;
    let res_name = required_str_from(
        tokens,
        column_map,
        &sources.comp_id,
        "res_name",
        row_start_line,
    )?;
    let chain_id = required_str_from(
        tokens,
        column_map,
        &sources.asym_id,
        "chain_id",
        row_start_line,
    )?;
    let res_seq = read_seq_id(tokens, column_map, &sources.seq_mode, row_start_line)?;
    let alt_loc = read_optional_char(
        tokens,
        column_map,
        "label_alt_id",
        "alt_loc",
        row_start_line,
    )?;
    let i_code = read_optional_char(
        tokens,
        column_map,
        "pdbx_PDB_ins_code",
        "i_code",
        row_start_line,
    )?;
    let x = required_f32(tokens, column_map, "Cartn_x", "x", row_start_line)?;
    let y = required_f32(tokens, column_map, "Cartn_y", "y", row_start_line)?;
    let z = required_f32(tokens, column_map, "Cartn_z", "z", row_start_line)?;
    let occupancy = read_occ_or_b(
        tokens,
        column_map,
        "occupancy",
        "occupancy",
        row_start_line,
        1.0,
    )?;
    let temp_factor = read_occ_or_b(
        tokens,
        column_map,
        "B_iso_or_equiv",
        "temp_factor",
        row_start_line,
        0.0,
    )?;

    // `_atom_site.type_symbol` is mmCIF's element column, but unlike PDB's
    // fixed columns 77-78 it is genuinely optional per the mmCIF dictionary
    // (and absent entirely, or '.'/'?', in some non-RCSB-authored files).
    // Fall back to the same two-letter-aware `infer_element` used by
    // `pdb.rs`/`pqr.rs` (backlog #5052) rather than leaving `element` empty.
    let type_symbol = read_optional_str(tokens, column_map, "type_symbol");
    let element = if type_symbol.is_empty() {
        infer_element(&atom_name).to_string()
    } else {
        type_symbol
    };

    Ok((
        AtomRecord {
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
            occupancy,
            temp_factor,
            element,
            charge: None,
            radius: None,
            is_hetatm,
        },
        new_model,
    ))
}

// ---------------------------------------------------------------------
// Line-anchored row assembly + document-level state machine (decision d)
// ---------------------------------------------------------------------

struct CifParseState {
    column_names: Vec<String>,
    column_map: HashMap<String, usize>,
    /// True while plain (non-tag, non-reserved) lines should be tokenized as
    /// `_atom_site` row data -- i.e. the most recently opened `loop_`'s
    /// columns are `_atom_site.*` ones.
    current_loop_is_atom_site: bool,
    column_sources: Option<ColumnSources>,
    /// Which `data_` block index first contained an `_atom_site` loop
    /// (decision m: a SECOND block containing one is `MultipleDataBlocks`).
    atom_site_data_block: Option<usize>,
    data_block_index: usize,
    current_model: i32,
    /// `Some((row_start_line, tokens))` while a row is only partially
    /// assembled (fewer tokens than `column_names.len()`).
    pending_row: Option<(usize, Vec<(String, bool)>)>,
    raw_data: RawAtomData,
    model_ids: Vec<usize>,
}

impl CifParseState {
    fn new() -> Self {
        Self {
            column_names: Vec::new(),
            column_map: HashMap::new(),
            current_loop_is_atom_site: false,
            column_sources: None,
            atom_site_data_block: None,
            data_block_index: 0,
            current_model: 1,
            pending_row: None,
            raw_data: RawAtomData::new(),
            model_ids: Vec::new(),
        }
    }

    fn check_no_pending_row(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some((start_line, tokens)) = &self.pending_row {
            return Err(Box::new(row_length_error(
                *start_line,
                tokens.len(),
                self.column_names.len(),
            )));
        }
        Ok(())
    }

    fn handle_token(&mut self, text: String, quoted: bool, line_no: usize) {
        if !self.current_loop_is_atom_site {
            return;
        }
        match &mut self.pending_row {
            Some((_, tokens)) => tokens.push((text, quoted)),
            None => self.pending_row = Some((line_no, vec![(text, quoted)])),
        }
    }

    /// Called once all tokens sourced from physical line `line_no` have
    /// been pushed via [`Self::handle_token`]. Completes the row if it now
    /// has exactly `column_names.len()` tokens, errors `RowLength` if it has
    /// more (decision d: "a completed row leaves tokens on the same line"),
    /// and otherwise leaves it pending for the next line.
    fn finish_line(&mut self, line_no: usize) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.column_names.len();
        let len = match &self.pending_row {
            None => return Ok(()),
            Some((_, tokens)) => tokens.len(),
        };
        if len < n {
            return Ok(());
        }
        if len > n {
            return Err(Box::new(row_length_error(line_no, len, n)));
        }
        let (start_line, tokens) = self.pending_row.take().expect("checked Some above");
        if self.column_sources.is_none() {
            self.column_sources = Some(resolve_column_sources(&self.column_map)?);
        }
        let sources = self.column_sources.as_ref().expect("just set");
        let (atom, new_model) = extract_row(
            &tokens,
            &self.column_map,
            sources,
            start_line,
            self.current_model,
        )?;
        self.current_model = new_model;
        self.model_ids.push(new_model as usize);
        self.raw_data.add_atom(atom);
        Ok(())
    }

    fn handle_reserved(
        &mut self,
        lower: &str,
        _line_no: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.check_no_pending_row()?;
        if lower.starts_with("data_") {
            self.data_block_index += 1;
        }
        self.column_names.clear();
        self.column_map.clear();
        self.current_loop_is_atom_site = false;
        Ok(())
    }

    fn handle_tag_line(
        &mut self,
        trimmed: &str,
        line_no: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.check_no_pending_row()?;
        let mut parts = trimmed.splitn(2, char::is_whitespace);
        let tag = parts.next().expect("trimmed line is non-empty");
        let rest_val = parts.next().map(str::trim).unwrap_or(""); // documented-default: a bare tag with nothing after it -> no inline value (the normal loop-header form)
        let dot = tag.find('.');
        let category = match dot {
            Some(d) => &tag[..d],
            None => tag,
        };
        if category == "_atom_site" {
            if !rest_val.is_empty() {
                return Err(Box::new(non_loop_error(line_no, tag)));
            }
            match self.atom_site_data_block {
                Some(prev_block) if prev_block != self.data_block_index => {
                    return Err(Box::new(multiple_data_blocks_error(
                        line_no,
                        self.data_block_index,
                    )));
                }
                Some(_) => {}
                None => self.atom_site_data_block = Some(self.data_block_index),
            }
            let col_name = dot.map(|d| tag[d + 1..].to_string()).unwrap_or_default(); // documented-default: a bare "_atom_site" tag with no ".column" suffix has no column name
            if !self.current_loop_is_atom_site {
                self.current_loop_is_atom_site = true;
                self.column_sources = None;
            }
            self.column_map
                .insert(col_name.clone(), self.column_names.len());
            self.column_names.push(col_name);
        } else {
            self.current_loop_is_atom_site = false;
        }
        Ok(())
    }
}

/// Parse mmCIF file and return raw atom data with model IDs
pub fn parse_mmcif_file<P: AsRef<Path>>(
    path: P,
) -> Result<(RawAtomData, Vec<usize>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    parse_mmcif_from_reader(reader)
}

/// Internal parser implementation taking any reader
pub fn parse_mmcif_from_reader<R: BufRead>(
    reader: R,
) -> Result<(RawAtomData, Vec<usize>), Box<dyn std::error::Error>> {
    let mut state = CifParseState::new();
    let mut in_text_field = false;
    let mut text_field_buffer = String::new();
    let mut text_field_start_line = 0usize;
    let mut last_line_no = 0usize;

    for (idx, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line_no = idx + 1;
        last_line_no = line_no;

        // ';' text fields take precedence over every other line classifier
        // (decision d) -- checked first, both to open and to close one.
        if in_text_field {
            if let Some(remainder) = line.strip_prefix(';') {
                let text = std::mem::take(&mut text_field_buffer);
                in_text_field = false;
                state.handle_token(text, true, text_field_start_line);
                if !remainder.trim().is_empty() {
                    match tokenize_line(remainder) {
                        Ok(toks) => {
                            for (t, q) in toks {
                                state.handle_token(t, q, line_no);
                            }
                        }
                        Err(()) => {
                            return Err(Box::new(unterminated_quote_error(line_no, remainder)))
                        }
                    }
                }
                state.finish_line(line_no)?;
            } else {
                if !text_field_buffer.is_empty() {
                    text_field_buffer.push('\n');
                }
                text_field_buffer.push_str(&line);
            }
            continue;
        }

        if let Some(remainder) = line.strip_prefix(';') {
            text_field_buffer = remainder.to_string();
            text_field_start_line = line_no;
            in_text_field = true;
            continue;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("loop_")
            || lower.starts_with("data_")
            || lower.starts_with("save_")
            || lower.starts_with("global_")
            || lower.starts_with("stop_")
        {
            state.handle_reserved(&lower, line_no)?;
            continue;
        }

        if trimmed.starts_with('_') {
            state.handle_tag_line(trimmed, line_no)?;
            continue;
        }

        if state.current_loop_is_atom_site {
            match tokenize_line(&line) {
                Ok(toks) => {
                    for (t, q) in toks {
                        state.handle_token(t, q, line_no);
                    }
                    state.finish_line(line_no)?;
                }
                Err(()) => return Err(Box::new(unterminated_quote_error(line_no, &line))),
            }
        }
    }

    if in_text_field {
        // EOF while a ';' text field was still open is the same class of
        // malformation as an unclosed '/" string -- never silently drop the
        // partial content.
        return Err(Box::new(unterminated_quote_error(
            last_line_no,
            &text_field_buffer,
        )));
    }

    if let Some((start_line, tokens)) = &state.pending_row {
        return Err(Box::new(row_length_error(
            *start_line,
            tokens.len(),
            state.column_names.len(),
        )));
    }

    if state.raw_data.num_atoms == 0 {
        return Err("No atoms found in mmCIF file".into());
    }

    Ok((state.raw_data, state.model_ids))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_str(text: &str) -> Result<(RawAtomData, Vec<usize>), Box<dyn std::error::Error>> {
        parse_mmcif_from_reader(text.as_bytes())
    }

    fn expect_ok(text: &str) -> RawAtomData {
        parse_str(text)
            .unwrap_or_else(|e| panic!("expected Ok, got Err: {e}"))
            .0
    }

    fn expect_cif_err(text: &str) -> CifFieldError {
        match parse_str(text) {
            Ok(_) => panic!("expected an error, got Ok"),
            Err(e) => e
                .downcast_ref::<CifFieldError>()
                .cloned()
                .unwrap_or_else(|| panic!("error was not a CifFieldError: {e}")),
        }
    }

    fn atom_site_header(cols: &[&str]) -> String {
        let mut s = String::from("data_TEST\nloop_\n");
        for c in cols {
            s.push_str("_atom_site.");
            s.push_str(c);
            s.push('\n');
        }
        s
    }

    fn line_of(text: &str, needle: &str) -> usize {
        for (i, line) in text.lines().enumerate() {
            if line.contains(needle) {
                return i + 1;
            }
        }
        panic!("needle {needle:?} not found in:\n{text}");
    }

    /// group_PDB, id, label_atom_id, label_alt_id, label_comp_id,
    /// label_asym_id, label_seq_id, pdbx_PDB_ins_code, Cartn_x, Cartn_y,
    /// Cartn_z, occupancy, B_iso_or_equiv, pdbx_PDB_model_num (14 columns).
    const FULL_COLS: &[&str] = &[
        "group_PDB",
        "id",
        "label_atom_id",
        "label_alt_id",
        "label_comp_id",
        "label_asym_id",
        "label_seq_id",
        "pdbx_PDB_ins_code",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "B_iso_or_equiv",
        "pdbx_PDB_model_num",
    ];
    const VALID_ROW: &str = "ATOM 1 N . ALA A 1 ? 0.000 0.000 0.000 1.00 10.00 1";

    // -------------------------------------------------------------
    // Old tests preserved: still exercise the PUBLIC parse_mmcif_from_reader
    // API and their assertions hold unchanged under the new implementation
    // (verified by hand: no '.'/'?' collision with a real numeric value in
    // either fixture). `parse_cif_values`/`extract_atom_record` no longer
    // exist in their old shape (replaced by the tokenizer + extract_row
    // above), so the old unit tests that called them directly
    // (`test_parse_cif_values_simple`, `test_parse_cif_values_quoted`,
    // `test_extract_atom_record`) are removed; `test_extract_atom_record_
    // missing_type_symbol_falls_back_to_name` is REQUIRED to still pass by
    // the sprint plan and is re-expressed below against the public API,
    // preserving its exact intent (see that test for the deviation note).
    // -------------------------------------------------------------

    #[test]
    fn test_mmcif_multiple_loops_and_missing() {
        let cif_content = "
# Dummy section
_entity.id 1
_entity.type polymer

loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
ATOM 1 N N . ALA A 1 ? 0.000 0.000 0.000 1.00 0.00
ATOM 2 C CA . ALA A 1 . 1.000 1.000 1.000 1.00 0.00

loop_
_next_category.field value
";
        let (raw_data, _) = parse_mmcif_from_reader(cif_content.as_bytes()).unwrap();
        assert_eq!(raw_data.num_atoms, 2);
        assert_eq!(raw_data.atom_names[0], "N");
        assert_eq!(raw_data.atom_names[1], "CA");
        assert_eq!(raw_data.insertion_codes[0], ' '); // '?' -> optional -> blank
        assert_eq!(raw_data.insertion_codes[1], ' '); // '.' -> optional -> blank
    }

    #[test]
    fn test_mmcif_permuted_columns() {
        let cif_content = "
loop_
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
0.0 0.0 0.0 ATOM 1 N ALA A 1
1.0 2.0 3.0 ATOM 2 CA ALA A 1
";
        let (raw_data, _) = parse_mmcif_from_reader(cif_content.as_bytes()).unwrap();
        assert_eq!(raw_data.num_atoms, 2);
        assert_eq!(raw_data.atom_names[1], "CA");
        assert_eq!(raw_data.coords[3], 1.0);
        assert_eq!(raw_data.coords[4], 2.0);
        assert_eq!(raw_data.coords[5], 3.0);
    }

    #[test]
    fn test_extract_atom_record_missing_type_symbol_falls_back_to_name() {
        // Re-expressed against the public API (decision h): the old
        // `extract_atom_record(values: &[&str], column_map: &HashMap<..>)`
        // this test called directly no longer exists in that shape -- row
        // extraction now works from `Vec<(String, bool)>` tokens plus a
        // resolved `ColumnSources`, not raw `&str` slices, since values now
        // carry a quoted flag (decision d). The BEHAVIOUR under test is
        // unchanged: `_atom_site.type_symbol` absent entirely -> fall back
        // to the same two-letter-aware `infer_element` used by
        // `pdb.rs`/`pqr.rs`, not an empty string and not a naive
        // first-character slice (which would wrongly report "C" for "CL").
        let text = format!(
            "{}ATOM 1 CL . CL A 500 ? 12.000 3.000 4.000 1.00 0.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let raw = expect_ok(&text);
        assert_eq!(raw.atom_names[0], "CL");
        assert_eq!(raw.elements[0], "Cl");
    }

    #[test]
    fn valid_row_parses_with_expected_defaults() {
        let text = format!("{}{}\n", atom_site_header(FULL_COLS), VALID_ROW);
        let raw = expect_ok(&text);
        assert_eq!(raw.num_atoms, 1);
        assert_eq!(raw.alt_locs[0], ' '); // '.' alt_id -> blank
        assert_eq!(raw.insertion_codes[0], ' '); // '?' ins_code -> blank
        assert_eq!(raw.chain_ids[0], "A");
        assert_eq!(raw.res_ids[0], 1);
        assert!(!raw.is_hetatm[0]);
    }

    // -------------------------------------------------------------
    // Parse Ok
    // -------------------------------------------------------------

    #[test]
    fn row_wrapped_over_two_lines_ok() {
        let text = format!(
            "{}ATOM 1 N . ALA A 1 ?\n0.000 0.000 0.000 1.00 10.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let raw = expect_ok(&text);
        assert_eq!(raw.num_atoms, 1);
        assert_eq!(raw.atom_names[0], "N");
    }

    #[test]
    fn text_field_in_row_ok() {
        let text = format!(
            "{}ATOM 1\n;\nN\n;\n. ALA A 1 ? 0.000 0.000 0.000 1.00 10.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let raw = expect_ok(&text);
        assert_eq!(raw.num_atoms, 1);
        assert_eq!(raw.atom_names[0], "N");
    }

    #[test]
    fn text_field_containing_line_starting_with_underscore_ok() {
        let text = format!(
            "{}ATOM 1 N .\n;\n_weird\nALA\n;\nA 1 ? 0.000 0.000 0.000 1.00 10.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let raw = expect_ok(&text);
        assert_eq!(raw.num_atoms, 1);
        assert_eq!(raw.res_names[0], "_weird\nALA");
    }

    #[test]
    fn unquoted_o5_prime_ok() {
        let text = format!(
            "{}ATOM 1 O5' . ALA A 1 ? 0.000 0.000 0.000 1.00 10.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let raw = expect_ok(&text);
        assert_eq!(raw.atom_names[0], "O5'");
    }

    #[test]
    fn double_quoted_o5_prime_ok() {
        let text = format!(
            "{}ATOM 1 \"O5'\" . ALA A 1 ? 0.000 0.000 0.000 1.00 10.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let raw = expect_ok(&text);
        assert_eq!(raw.atom_names[0], "O5'");
    }

    #[test]
    fn single_quoted_o5_prime_ok() {
        let text = format!(
            "{}ATOM 1 'O5'' . ALA A 1 ? 0.000 0.000 0.000 1.00 10.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let raw = expect_ok(&text);
        assert_eq!(raw.atom_names[0], "O5'");
    }

    #[test]
    fn auth_only_header_ok() {
        let cols = [
            "group_PDB",
            "id",
            "auth_atom_id",
            "auth_comp_id",
            "auth_asym_id",
            "auth_seq_id",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
        ];
        let text = format!(
            "{}ATOM 1 N ALA A 1 0.000 0.000 0.000\n",
            atom_site_header(&cols)
        );
        let raw = expect_ok(&text);
        assert_eq!(raw.num_atoms, 1);
        assert_eq!(raw.atom_names[0], "N");
        assert_eq!(raw.chain_ids[0], "A");
        assert_eq!(raw.res_ids[0], 1);
    }

    #[test]
    fn dot_label_seq_id_with_auth_seq_id_ok() {
        let mut cols = FULL_COLS.to_vec();
        cols.push("auth_seq_id");
        let text = format!(
            "{}ATOM 1 N . ALA A . ? 0.000 0.000 0.000 1.00 10.00 1 55\n",
            atom_site_header(&cols)
        );
        let raw = expect_ok(&text);
        assert_eq!(raw.res_ids[0], 55);
    }

    #[test]
    fn question_ins_code_ok_blank() {
        // VALID_ROW's ins_code token is already '?'.
        let text = format!("{}{}\n", atom_site_header(FULL_COLS), VALID_ROW);
        let raw = expect_ok(&text);
        assert_eq!(raw.insertion_codes[0], ' ');
    }

    #[test]
    fn dot_alt_id_ok_blank() {
        // VALID_ROW's alt_id token is already '.'.
        let text = format!("{}{}\n", atom_site_header(FULL_COLS), VALID_ROW);
        let raw = expect_ok(&text);
        assert_eq!(raw.alt_locs[0], ' ');
    }

    // -------------------------------------------------------------
    // Parse Ok with a checked value
    // -------------------------------------------------------------

    #[test]
    fn quoted_question_mark_in_label_atom_id_is_present_text() {
        let text = format!(
            "{}ATOM 1 '?' . ALA A 1 ? 0.000 0.000 0.000 1.00 10.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let raw = expect_ok(&text);
        assert_eq!(raw.atom_names[0], "?");
    }

    #[test]
    fn dot_occupancy_gives_1_0() {
        let text = format!(
            "{}ATOM 1 N . ALA A 1 ? 0.000 0.000 0.000 . 10.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let raw = expect_ok(&text);
        assert_eq!(raw.occupancy[0], 1.0);
    }

    // -------------------------------------------------------------
    // Tokenizer and row errors
    // -------------------------------------------------------------

    #[test]
    fn unterminated_quote_errors() {
        let text = format!(
            "{}ATOM 1 'N . ALA A 1 ? 0.000 0.000 0.000 1.00 10.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::UnterminatedQuote);
    }

    #[test]
    fn one_extra_token_on_a_line_gives_row_length_naming_that_line() {
        let text = format!(
            "{}ATOM 1 N .\nALA A 1 ? 0.000 0.000 0.000 1.00 10.00 1 EXTRA\n",
            atom_site_header(FULL_COLS)
        );
        let expected_line = line_of(&text, "EXTRA");
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::RowLength);
        assert_eq!(err.row_start_line, expected_line);
    }

    #[test]
    fn row_cut_by_eof_gives_row_length_naming_row_start() {
        let text = format!("{}ATOM 1 N .\n", atom_site_header(FULL_COLS));
        let expected_line = line_of(&text, "ATOM 1 N .");
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::RowLength);
        assert_eq!(err.row_start_line, expected_line);
    }

    #[test]
    fn loop_mid_row_gives_row_length() {
        let text = format!("{}ATOM 1 N .\nloop_\n", atom_site_header(FULL_COLS));
        let expected_line = line_of(&text, "ATOM 1 N .");
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::RowLength);
        assert_eq!(err.row_start_line, expected_line);
    }

    #[test]
    fn atom_sites_category_is_a_boundary() {
        let text = format!(
            "{}ATOM 1 N .\n_atom_sites.entry_id 1CRN\n",
            atom_site_header(FULL_COLS)
        );
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::RowLength);
    }

    // -------------------------------------------------------------
    // Field errors
    // -------------------------------------------------------------

    #[test]
    fn question_cartn_x_is_unknown() {
        let text = format!(
            "{}ATOM 1 N . ALA A 1 ? ? 0.000 0.000 1.00 10.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::Unknown);
        assert_eq!(err.field, "x");
        assert_eq!(err.column, "Cartn_x");
    }

    #[test]
    fn dot_label_atom_id_is_inapplicable_label_present_auth_ignored() {
        let mut cols = FULL_COLS.to_vec();
        cols.push("auth_atom_id");
        let text = format!(
            "{}ATOM 1 . . ALA A 1 ? 0.000 0.000 0.000 1.00 10.00 1 REALNAME\n",
            atom_site_header(&cols)
        );
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::Inapplicable);
        assert_eq!(err.column, "label_atom_id");
        assert_eq!(err.field, "atom_name");
    }

    #[test]
    fn question_label_seq_id_is_unknown() {
        let mut cols = FULL_COLS.to_vec();
        cols.push("auth_seq_id");
        let text = format!(
            "{}ATOM 1 N . ALA A ? ? 0.000 0.000 0.000 1.00 10.00 1 9\n",
            atom_site_header(&cols)
        );
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::Unknown);
        assert_eq!(err.column, "label_seq_id");
        assert_eq!(err.field, "res_seq");
    }

    #[test]
    fn ab_alt_id_is_unparseable() {
        let text = format!(
            "{}ATOM 1 N AB ALA A 1 ? 0.000 0.000 0.000 1.00 10.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::Unparseable);
        assert_eq!(err.column, "label_alt_id");
        assert_eq!(err.field, "alt_loc");
    }

    #[test]
    fn garbage_id_is_unparseable() {
        let text = format!(
            "{}ATOM garbage N . ALA A 1 ? 0.000 0.000 0.000 1.00 10.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::Unparseable);
        assert_eq!(err.column, "id");
        assert_eq!(err.field, "serial");
    }

    #[test]
    fn x_model_num_is_invalid_model() {
        let text = format!(
            "{}ATOM 1 N . ALA A 1 ? 0.000 0.000 0.000 1.00 10.00 x\n",
            atom_site_header(FULL_COLS)
        );
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::InvalidModel);
        assert_eq!(err.field, "model_num");
    }

    #[test]
    fn dot_model_num_is_invalid_model() {
        let text = format!(
            "{}ATOM 1 N . ALA A 1 ? 0.000 0.000 0.000 1.00 10.00 .\n",
            atom_site_header(FULL_COLS)
        );
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::InvalidModel);
        assert_eq!(err.field, "model_num");
    }

    #[test]
    fn garbage_occupancy_is_unparseable() {
        let text = format!(
            "{}ATOM 1 N . ALA A 1 ? 0.000 0.000 0.000 garbage 10.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::Unparseable);
        assert_eq!(err.field, "occupancy");
    }

    #[test]
    fn nan_b_is_nonfinite() {
        let text = format!(
            "{}ATOM 1 N . ALA A 1 ? 0.000 0.000 0.000 1.00 nan 1\n",
            atom_site_header(FULL_COLS)
        );
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::NonFinite);
        assert_eq!(err.field, "temp_factor");
    }

    #[test]
    fn group_pdb_foo_is_bad_group() {
        let text = format!(
            "{}FOO 1 N . ALA A 1 ? 0.000 0.000 0.000 1.00 10.00 1\n",
            atom_site_header(FULL_COLS)
        );
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::BadGroup);
        assert_eq!(err.column, "group_PDB");
    }

    // -------------------------------------------------------------
    // Structural errors
    // -------------------------------------------------------------

    #[test]
    fn non_loop_atom_site_is_non_loop_unsupported() {
        let text = "data_TEST\n_atom_site.id 5\n";
        let err = expect_cif_err(text);
        assert_eq!(err.kind, TokenFieldErrorKind::NonLoopUnsupported);
    }

    #[test]
    fn two_data_blocks_give_multiple_data_blocks() {
        let cols = [
            "group_PDB",
            "id",
            "label_atom_id",
            "label_comp_id",
            "label_asym_id",
            "label_seq_id",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
        ];
        let block = |name: &str| {
            format!(
                "data_{name}\nloop_\n{}ATOM 1 N ALA A 1 0.000 0.000 0.000\n",
                atom_site_header(&cols)
                    .lines()
                    .skip(2) // drop the helper's own "data_TEST\nloop_\n" prefix
                    .map(|l| format!("{l}\n"))
                    .collect::<String>()
            )
        };
        let text = format!("{}{}", block("ONE"), block("TWO"));
        let err = expect_cif_err(&text);
        assert_eq!(err.kind, TokenFieldErrorKind::MultipleDataBlocks);
    }
}
