//! Ground-truth loader for the mbondi2 / OBC2 parameter tables.
//!
//! The numbers live in `data/mbondi2.xml`, not in this file. That separation is
//! deliberate and is the point of the module: a physical constant written as a
//! Rust literal is invisible to review as *data*, and this repository has
//! already shipped constants that changed silently inside an unrelated refactor
//! (CLAUDE.md, ledger A2/A3). Keeping them in a provenance-carrying data file
//! makes "which element got which number, and who says so" a diffable question.
//!
//! # Why absence is the interesting case
//!
//! mbondi2 is a protein/nucleic-acid radius set. It defines radii for C, N, O,
//! F, Si, P, S, Cl and (by rule) H -- and for nothing else. Selenium, sodium,
//! copper, iron, zinc and the rest are **not in the set at all**. The reference
//! implementation substitutes a documented catch-all for them.
//!
//! Substituting that catch-all is an inference the caller did not license, so
//! every assignment here is tagged with the [`ParameterSource`] that produced
//! it. A caller can then distinguish a tabulated measurement from a shrug --
//! which is exactly the distinction that first-character dispatch destroyed
//! when it handed `"SE"` sulfur's radius and `"NA"` nitrogen's.

use std::collections::HashMap;
use std::sync::OnceLock;

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

/// The embedded ground-truth table. Compiled in so there is no runtime file
/// dependency to go missing, while the reviewable artifact stays a data file.
const MBONDI2_XML: &str = include_str!("../../data/mbondi2.xml");

/// Where a single atom's parameter came from.
///
/// This is the machine-readable provenance channel. It is deliberately
/// `#[repr(u8)]` so it can cross the Python boundary as a plain `uint8` array
/// alongside the values, with no boxing and no structured dtype (JAX refuses
/// structured dtypes, and a per-atom Python object would mean 10^4-10^6 boxed
/// objects on the hot path).
///
/// Discriminants are part of the wire format: **append new variants, never
/// renumber existing ones**. `source_code_meanings()` is the schema a consumer
/// reads to interpret them, so an unrecognised code surfaces as unknown rather
/// than being silently dropped.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterSource {
    /// The element is in the mbondi2 table and this is its tabulated value.
    Tabulated = 0,
    /// Hydrogen bonded to nitrogen -- the licensed mbondi2 bond-dependent rule.
    HydrogenBondedToNitrogen = 1,
    /// Hydrogen bonded to anything else -- the licensed rule's other branch.
    HydrogenDefault = 2,
    /// The element is **absent from mbondi2**; the documented catch-all was
    /// substituted. This is not a measurement. Callers that require real
    /// parameters must treat this as missing data, not as a value.
    Fallback = 3,
    /// A hydrogen with no bonds at all, so the bond-dependent rule could not be
    /// evaluated. The unbonded branch's value was used. Almost always signals a
    /// topology that was never built, rather than genuine chemistry.
    HydrogenUnbonded = 4,
}

impl ParameterSource {
    /// True when the value is a real tabulated/derived mbondi2 parameter.
    ///
    /// False means the number is a stand-in and the true value is unknown.
    pub fn is_licensed(self) -> bool {
        !matches!(
            self,
            ParameterSource::Fallback | ParameterSource::HydrogenUnbonded
        )
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Schema for the `u8` provenance codes, so a consumer can render them without
/// hard-coding this crate's discriminants. Exposed through the Python bindings.
pub fn source_code_meanings() -> &'static [(u8, &'static str)] {
    &[
        (0, "tabulated"),
        (1, "hydrogen_bonded_to_nitrogen"),
        (2, "hydrogen_default"),
        (3, "fallback_element_absent_from_mbondi2"),
        (4, "hydrogen_unbonded"),
    ]
}

/// Where the numbers came from, carried alongside them.
#[derive(Debug, Clone, Default)]
pub struct Provenance {
    pub scheme: String,
    pub citation: String,
    pub reference_project: String,
    pub reference_path: String,
    pub reference_ref: String,
    pub captured: String,
}

/// The parsed mbondi2 / OBC2 tables.
#[derive(Debug, Clone)]
pub struct Mbondi2Table {
    radii: HashMap<String, f32>,
    screen: HashMap<String, f32>,
    h_bonded_to_nitrogen: f32,
    h_otherwise: f32,
    fallback_radius: f32,
    fallback_screen: f32,
    provenance: Provenance,
}

impl Mbondi2Table {
    /// Radius for an element symbol, or `None` if mbondi2 does not define one.
    ///
    /// `None` is the honest answer for Se/Na/Cu/Fe/Zn/... -- it is not an error,
    /// and it must not be papered over by the caller without recording that it
    /// did so. Hydrogen returns `None` here because its radius is bond
    /// dependent; use [`Self::hydrogen_radius`].
    pub fn radius(&self, element: &str) -> Option<f32> {
        self.radii.get(element).copied()
    }

    /// Screening factor for an element symbol, or `None` if undefined.
    pub fn screen(&self, element: &str) -> Option<f32> {
        self.screen.get(element).copied()
    }

    /// The mbondi2 bond-dependent hydrogen rule.
    pub fn hydrogen_radius(&self, bonded_to_nitrogen: bool) -> f32 {
        if bonded_to_nitrogen {
            self.h_bonded_to_nitrogen
        } else {
            self.h_otherwise
        }
    }

    /// The documented catch-all radius. Only ever paired with
    /// [`ParameterSource::Fallback`] -- never returned as if tabulated.
    pub fn fallback_radius(&self) -> f32 {
        self.fallback_radius
    }

    /// The documented catch-all screening factor.
    pub fn fallback_screen(&self) -> f32 {
        self.fallback_screen
    }

    /// Element symbols mbondi2 actually defines a radius for.
    pub fn defined_radius_elements(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.radii.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }

    /// Element symbols with a defined screening factor.
    pub fn defined_screen_elements(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.screen.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }

    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

/// Parse failure in the embedded table.
#[derive(Debug, thiserror::Error)]
pub enum GbParameterError {
    #[error("malformed mbondi2 parameter table: {0}")]
    Xml(String),
    #[error("mbondi2 parameter table is missing required section <{0}>")]
    MissingSection(&'static str),
    #[error("mbondi2 parameter table entry <{tag}> has a bad or missing '{attr}' attribute")]
    BadAttribute {
        tag: &'static str,
        attr: &'static str,
    },
}

fn attr(e: &BytesStart<'_>, key: &str) -> Option<String> {
    e.attributes()
        .flatten()
        .find(|a| a.key.as_ref() == key.as_bytes())
        .and_then(|a| String::from_utf8(a.value.into_owned()).ok())
}

fn attr_f32(e: &BytesStart<'_>, key: &str) -> Option<f32> {
    attr(e, key)?.trim().parse::<f32>().ok()
}

/// Which sub-table an `<Element>` entry belongs to. Positional, so the parser
/// tracks the enclosing section rather than guessing from attributes.
#[derive(PartialEq)]
enum Section {
    None,
    Radii,
    Screen,
}

/// Parse the embedded table. Kept separate from [`table`] so tests can exercise
/// failure modes on hand-written input.
pub fn parse_table(xml: &str) -> Result<Mbondi2Table, GbParameterError> {
    let mut reader = Reader::from_str(xml);
    reader.trim_text(true);

    let mut radii: HashMap<String, f32> = HashMap::new();
    let mut screen: HashMap<String, f32> = HashMap::new();
    let mut h_bonded_to_nitrogen: Option<f32> = None;
    let mut h_otherwise: Option<f32> = None;
    let mut fallback_radius: Option<f32> = None;
    let mut fallback_screen: Option<f32> = None;
    let mut provenance = Provenance::default();

    let mut section = Section::None;
    let mut in_citation = false;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => match e.name().as_ref() {
                b"GBParameters" => {
                    provenance.scheme = attr(e, "scheme").unwrap_or_default();
                }
                b"Radii" => section = Section::Radii,
                b"Screen" => section = Section::Screen,
                b"Citation" => in_citation = true,
                _ => {}
            },
            Ok(Event::End(ref e)) => match e.name().as_ref() {
                b"Radii" | b"Screen" => section = Section::None,
                b"Citation" => in_citation = false,
                _ => {}
            },
            Ok(Event::Text(ref t)) if in_citation => {
                if let Ok(s) = t.unescape() {
                    provenance.citation = s.split_whitespace().collect::<Vec<_>>().join(" ");
                }
            }
            Ok(Event::Empty(ref e)) => match e.name().as_ref() {
                b"Element" => {
                    let symbol = attr(e, "symbol").ok_or(GbParameterError::BadAttribute {
                        tag: "Element",
                        attr: "symbol",
                    })?;
                    match section {
                        Section::Radii => {
                            let r =
                                attr_f32(e, "radius").ok_or(GbParameterError::BadAttribute {
                                    tag: "Element",
                                    attr: "radius",
                                })?;
                            radii.insert(symbol, r);
                        }
                        Section::Screen => {
                            let s =
                                attr_f32(e, "screen").ok_or(GbParameterError::BadAttribute {
                                    tag: "Element",
                                    attr: "screen",
                                })?;
                            screen.insert(symbol, s);
                        }
                        Section::None => {}
                    }
                }
                b"BondedTo" => {
                    h_bonded_to_nitrogen =
                        Some(attr_f32(e, "radius").ok_or(GbParameterError::BadAttribute {
                            tag: "BondedTo",
                            attr: "radius",
                        })?);
                }
                b"Otherwise" => {
                    h_otherwise =
                        Some(attr_f32(e, "radius").ok_or(GbParameterError::BadAttribute {
                            tag: "Otherwise",
                            attr: "radius",
                        })?);
                }
                b"Fallback" => {
                    fallback_radius =
                        Some(attr_f32(e, "radius").ok_or(GbParameterError::BadAttribute {
                            tag: "Fallback",
                            attr: "radius",
                        })?);
                    fallback_screen =
                        Some(attr_f32(e, "screen").ok_or(GbParameterError::BadAttribute {
                            tag: "Fallback",
                            attr: "screen",
                        })?);
                }
                b"ReferenceImplementation" => {
                    provenance.reference_project = attr(e, "project").unwrap_or_default();
                    provenance.reference_path = attr(e, "path").unwrap_or_default();
                    provenance.reference_ref = attr(e, "ref").unwrap_or_default();
                    provenance.captured = attr(e, "captured").unwrap_or_default();
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(e) => return Err(GbParameterError::Xml(e.to_string())),
            _ => {}
        }
        buf.clear();
    }

    if radii.is_empty() {
        return Err(GbParameterError::MissingSection("Radii"));
    }
    if screen.is_empty() {
        return Err(GbParameterError::MissingSection("Screen"));
    }

    Ok(Mbondi2Table {
        radii,
        screen,
        h_bonded_to_nitrogen: h_bonded_to_nitrogen
            .ok_or(GbParameterError::MissingSection("HydrogenRule"))?,
        h_otherwise: h_otherwise.ok_or(GbParameterError::MissingSection("HydrogenRule"))?,
        fallback_radius: fallback_radius.ok_or(GbParameterError::MissingSection("Fallback"))?,
        fallback_screen: fallback_screen.ok_or(GbParameterError::MissingSection("Fallback"))?,
        provenance,
    })
}

/// The process-wide mbondi2 table, parsed once from the embedded data file.
///
/// # Panics
/// If the embedded table is malformed. That is a build-integrity failure, not a
/// runtime condition: the data is `include_str!`d, so a healthy binary cannot
/// reach this. `embedded_table_parses` covers it in CI.
pub fn table() -> &'static Mbondi2Table {
    static TABLE: OnceLock<Mbondi2Table> = OnceLock::new();
    TABLE.get_or_init(|| {
        parse_table(MBONDI2_XML).unwrap_or_else(|e| {
            panic!(
                "embedded mbondi2 parameter table (crates/proxide-physics/data/mbondi2.xml) \
                 failed to parse: {e}. This is a build-integrity failure -- the file is \
                 compiled in, so it cannot be missing or edited at runtime."
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_table_parses() {
        let t = table();
        assert_eq!(t.provenance().scheme, "mbondi2");
        assert!(!t.provenance().citation.is_empty());
        assert_eq!(t.provenance().reference_project, "OpenMM");
        assert!(!t.provenance().reference_ref.is_empty());
    }

    /// Pins the exact element coverage of mbondi2.
    ///
    /// This is a ratchet in BOTH directions (ledger B5): adding an element that
    /// mbondi2 does not define -- "completing" the table with a Bondi or UFF
    /// value -- fails here just as loudly as deleting one. Silent growth is the
    /// failure mode this whole module exists to prevent.
    #[test]
    fn radius_coverage_is_exactly_mbondi2() {
        assert_eq!(
            table().defined_radius_elements(),
            vec!["C", "Cl", "F", "N", "O", "P", "S", "Si"]
        );
    }

    #[test]
    fn screen_coverage_is_exactly_mbondi2() {
        assert_eq!(
            table().defined_screen_elements(),
            vec!["C", "F", "H", "N", "O", "P", "S"]
        );
    }

    #[test]
    fn values_match_reference_implementation() {
        let t = table();
        // Transcribed from OpenMM `_mbondi2_radii`, pinned ref in the data file.
        assert_eq!(t.radius("N"), Some(1.55));
        assert_eq!(t.radius("O"), Some(1.50));
        assert_eq!(t.radius("F"), Some(1.50));
        assert_eq!(t.radius("Si"), Some(2.10));
        assert_eq!(t.radius("P"), Some(1.85));
        assert_eq!(t.radius("S"), Some(1.80));
        assert_eq!(t.radius("Cl"), Some(1.70));
        assert_eq!(t.radius("C"), Some(1.70));
        assert_eq!(t.hydrogen_radius(true), 1.30);
        assert_eq!(t.hydrogen_radius(false), 1.20);
        assert_eq!(t.fallback_radius(), 1.50);
        assert_eq!(t.fallback_screen(), 0.80);
    }

    /// The elements that motivated this work. mbondi2 genuinely does not define
    /// them, so `None` is the correct and only honest answer.
    #[test]
    fn elements_outside_mbondi2_report_absent_not_a_number() {
        let t = table();
        for element in ["Se", "Na", "Cu", "Fe", "Zn", "Mn", "Br", "I", "K", "Mg"] {
            assert_eq!(
                t.radius(element),
                None,
                "{element} is not part of mbondi2; it must report absent rather than \
                 borrow a neighbouring element's tabulated value"
            );
        }
    }

    #[test]
    fn fallback_is_never_reported_as_licensed() {
        assert!(!ParameterSource::Fallback.is_licensed());
        assert!(!ParameterSource::HydrogenUnbonded.is_licensed());
        assert!(ParameterSource::Tabulated.is_licensed());
        assert!(ParameterSource::HydrogenBondedToNitrogen.is_licensed());
    }

    /// Discriminants are a wire format shared with Python. Renumbering silently
    /// re-labels every previously written provenance array.
    #[test]
    fn source_discriminants_are_stable() {
        assert_eq!(ParameterSource::Tabulated.as_u8(), 0);
        assert_eq!(ParameterSource::HydrogenBondedToNitrogen.as_u8(), 1);
        assert_eq!(ParameterSource::HydrogenDefault.as_u8(), 2);
        assert_eq!(ParameterSource::Fallback.as_u8(), 3);
        assert_eq!(ParameterSource::HydrogenUnbonded.as_u8(), 4);
        assert_eq!(source_code_meanings().len(), 5);
    }

    #[test]
    fn malformed_table_is_an_error_not_a_default() {
        assert!(parse_table("<GBParameters/>").is_err());
        assert!(parse_table("not xml at all <<<").is_err());
    }
}
