//! Shared, format-neutral token-parsing helpers (sprint 25, task
//! 260922_autonomous-loop, track a, decision h, debt #1919/#1920).
//!
//! `pdb_fields.rs` (sprint 24), `pqr.rs` (sprint 25 track a), and `mmcif.rs`
//! (sprint 25 track b) each need the same two primitives -- "does this text
//! parse as a decimal integer" and "does this text parse as a finite float,
//! distinguishing an unparseable value from a parseable-but-NaN/infinite one"
//! -- and a shared vocabulary of what kind of malformation was found, so a
//! caller can `downcast_ref` a `Box<dyn Error>` to one of `PdbFieldError`,
//! `PqrFieldError`, or `CifFieldError` and match on a `TokenFieldErrorKind`
//! that means the same thing in all three.
//!
//! `PdbFieldError`/`PdbFieldErrorKind` (in `pdb_fields.rs`) are NOT changed by
//! this module -- they predate it (sprint 24) and their variant set has
//! already shipped. `pdb_fields.rs`'s `required_i32`/`required_f32`/
//! `required_f64` are refactored to call the helpers here internally, with
//! unchanged behaviour (same `str::parse` calls, same finite check), and to
//! map the shared [`TokenFieldErrorKind`] back to the pre-existing
//! `PdbFieldErrorKind`.

use std::fmt;

/// Literal error code shared by every `TokenFieldErrorKind`-based error type
/// (`PqrFieldError`, `CifFieldError`) introduced in sprint 25. `pdb_fields.rs`
/// keeps its own pre-existing `PDB_MALFORMED_RECORD_CODE` constant (same
/// string, not re-exported from here, to avoid changing that module's public
/// surface -- decision h).
pub const IO_MALFORMED_RECORD_CODE: &str = "PROX-IO-MALFORMED-RECORD";

/// What kind of malformation a token-based field parse found. Shared across
/// `pqr.rs`'s [`PqrFieldError`](crate::formats::pqr::PqrFieldError) and
/// `mmcif.rs`'s `CifFieldError` (track b). `#[non_exhaustive]`: PQR only
/// produces a subset of these variants; mmCIF's row/tokenizer/CifValue rules
/// need the rest.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenFieldErrorKind {
    /// A field's text does not parse as the expected numeric type.
    Unparseable,
    /// A field parsed successfully but is NaN or +/-infinity.
    NonFinite,
    /// A model-number field (PDB `MODEL`, mmCIF `pdbx_PDB_model_num`) is
    /// present but unparseable, or (mmCIF) is `.`/`?`.
    InvalidModel,
    /// A record/row has the wrong number of whitespace-delimited tokens, or a
    /// token merged two logical columns together (PQR decision a).
    TokenCount,
    /// A mmCIF loop row was not properly terminated at a line boundary
    /// (track b, decision d).
    RowLength,
    /// A mmCIF quoted value was opened but never closed (track b).
    UnterminatedQuote,
    /// A mmCIF value was the literal `.` (Inapplicable) where a required
    /// field forbids it (track b, decision e).
    Inapplicable,
    /// A mmCIF value was the literal `?` (Unknown) where a required field
    /// forbids it (track b, decision e).
    Unknown,
    /// A mmCIF `_atom_site.<column> value` non-loop form was encountered
    /// (track b, decision m).
    NonLoopUnsupported,
    /// More than one `data_` block contains an `_atom_site` loop (track b,
    /// decision m).
    MultipleDataBlocks,
    /// A mmCIF `group_PDB` value was present but was neither `ATOM` nor
    /// `HETATM` (track b, decision l).
    BadGroup,
    /// An I/O error while reading a line (not a content problem).
    Io,
}

impl fmt::Display for TokenFieldErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            TokenFieldErrorKind::Unparseable => "unparseable value",
            TokenFieldErrorKind::NonFinite => "non-finite value (NaN/infinity)",
            TokenFieldErrorKind::InvalidModel => "invalid model number",
            TokenFieldErrorKind::TokenCount => "wrong or merged token count",
            TokenFieldErrorKind::RowLength => "row not terminated at a line boundary",
            TokenFieldErrorKind::UnterminatedQuote => "unterminated quoted value",
            TokenFieldErrorKind::Inapplicable => {
                "value is '.' (inapplicable) but field is required"
            }
            TokenFieldErrorKind::Unknown => "value is '?' (unknown) but field is required",
            TokenFieldErrorKind::NonLoopUnsupported => "non-loop _atom_site form is unsupported",
            TokenFieldErrorKind::MultipleDataBlocks => {
                "more than one data_ block contains an _atom_site loop"
            }
            TokenFieldErrorKind::BadGroup => "group_PDB is neither ATOM nor HETATM",
            TokenFieldErrorKind::Io => "I/O error",
        };
        f.write_str(s)
    }
}

/// Truncate a `&str` to at most 80 characters, appending a `…(+N bytes)`
/// marker naming exactly how many additional raw bytes were not shown --
/// ledger B9: a truncated error must say that it truncated, not silently
/// present a partial line as if it were the whole one. Shared by
/// `PqrFieldError`/`CifFieldError`; `pdb_fields.rs` keeps its own byte-slice
/// variant (`truncate_raw`) since it works from `&[u8]`, not `&str`.
pub fn truncate_raw_str(s: &str) -> String {
    if s.chars().count() <= 80 {
        return s.to_string();
    }
    let shown: String = s.chars().take(80).collect();
    let shown_bytes = shown.len();
    let remaining = s.len().saturating_sub(shown_bytes);
    format!("{shown}\u{2026}(+{remaining} bytes)")
}

/// Parse `text` as a decimal `i32`. `None` on any parse failure (leading `+`,
/// hex, empty, trailing junk, out of range, ...).
pub fn parse_decimal_i32(text: &str) -> Option<i32> {
    text.parse::<i32>().ok()
}

/// Parse `text` as an `f32`, distinguishing "does not parse at all"
/// ([`TokenFieldErrorKind::Unparseable`]) from "parses, but is NaN or
/// infinite" ([`TokenFieldErrorKind::NonFinite`]).
pub fn parse_finite_f32(text: &str) -> Result<f32, TokenFieldErrorKind> {
    let value: f32 = text.parse().map_err(|_| TokenFieldErrorKind::Unparseable)?;
    if !value.is_finite() {
        return Err(TokenFieldErrorKind::NonFinite);
    }
    Ok(value)
}

/// Same as [`parse_finite_f32`] but for `f64` -- parsed independently from
/// the same field text, never cast from the `f32` result (a decimal literal
/// can round to a different bit pattern in each width if cast instead of
/// parsed directly).
pub fn parse_finite_f64(text: &str) -> Result<f64, TokenFieldErrorKind> {
    let value: f64 = text.parse().map_err(|_| TokenFieldErrorKind::Unparseable)?;
    if !value.is_finite() {
        return Err(TokenFieldErrorKind::NonFinite);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_i32_rejects_hex_and_junk() {
        assert_eq!(parse_decimal_i32("42"), Some(42));
        assert_eq!(parse_decimal_i32("-7"), Some(-7));
        assert_eq!(parse_decimal_i32("A000"), None);
        assert_eq!(parse_decimal_i32(""), None);
        assert_eq!(parse_decimal_i32("4.0"), None);
    }

    #[test]
    fn finite_f32_distinguishes_unparseable_from_nonfinite() {
        assert_eq!(parse_finite_f32("1.5"), Ok(1.5f32));
        assert_eq!(
            parse_finite_f32("xx.xxx"),
            Err(TokenFieldErrorKind::Unparseable)
        );
        assert_eq!(parse_finite_f32("nan"), Err(TokenFieldErrorKind::NonFinite));
        assert_eq!(parse_finite_f32("inf"), Err(TokenFieldErrorKind::NonFinite));
        assert_eq!(
            parse_finite_f32("-inf"),
            Err(TokenFieldErrorKind::NonFinite)
        );
    }

    #[test]
    fn finite_f64_distinguishes_unparseable_from_nonfinite() {
        assert_eq!(parse_finite_f64("1.5"), Ok(1.5f64));
        assert_eq!(
            parse_finite_f64("xx.xxx"),
            Err(TokenFieldErrorKind::Unparseable)
        );
        assert_eq!(parse_finite_f64("nan"), Err(TokenFieldErrorKind::NonFinite));
    }

    #[test]
    fn truncate_raw_str_marks_truncation() {
        let long = "x".repeat(200);
        let out = truncate_raw_str(&long);
        assert!(out.contains('\u{2026}'));
        assert!(out.contains("bytes)"));
        assert!(!truncate_raw_str("short").contains('\u{2026}'));
    }
}
