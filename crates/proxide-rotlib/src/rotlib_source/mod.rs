//! Abstraction layer for rotamer library sources (chi angles + statistics).
//!
//! Allows pluggable sources of backbone-dependent rotamer data: Dunbrack, MASTER, etc.
//! Each source provides residue codes, phi/psi bins with rotamer populations and chi angles.

pub mod dunbrack;

pub use dunbrack::DunbrackSource;

/// One rotamer entry within a (phi, psi) bin: chi angles, per-rotamer probability, observation count.
#[derive(Debug, Clone)]
pub struct RotamerEntry {
    /// Chi dihedral angles in degrees (chi1, chi2, ...).
    pub chi_values: Vec<f32>,
    /// Standard deviations of chi angles in degrees.
    pub chi_sigmas: Vec<f32>,
    /// Per-rotamer probability (0.0 to 1.0); typically normalized within the bin.
    pub probability: f64,
    /// Dunbrack observation count (raw number of observations).
    pub count: u32,
}

/// One (phi, psi) bin's full contents for a single residue: coordinates, rotamer populations.
#[derive(Debug, Clone)]
pub struct BinData {
    /// Phi dihedral angle (degrees, center of bin).
    pub phi: f64,
    /// Psi dihedral angle (degrees, center of bin).
    pub psi: f64,
    /// Bin frequency: sum of rotamer probabilities or observation fraction.
    pub freq: f64,
    /// Rotamers in this bin, typically sorted by probability (descending).
    pub rotamers: Vec<RotamerEntry>,
}

/// Source of chi-angle rotamer data. Implemented by various rotamer libraries.
///
/// Allows `convert_rotlib` to accept different sources of backbone-dependent rotamer
/// data (Dunbrack BBDEP2010, MASTER, etc.) and produce a consistent protobuf output.
pub trait RotlibSource: Send + Sync {
    /// Return all residue codes present in this source (e.g., "ALA", "ARG", ..., "PRO").
    /// Should be sorted for deterministic output.
    fn residue_codes(&self) -> Vec<String>;

    /// Return all (phi, psi) bins and their rotamers for a given residue code.
    /// Returns empty vec if code is not present.
    fn bins(&self, code: &str) -> Vec<BinData>;

    /// Return the index (0-based) of the default bin for a given residue code.
    /// Typically the bin with the highest frequency or observation count.
    fn default_bin_index(&self, code: &str) -> usize;

    /// Return the data license string (e.g., "ODC-BY-1.0").
    fn data_license(&self) -> &str;

    /// Return the attribution notice (e.g., "Contains data from Dunbrack BBDEP2010").
    fn attribution(&self) -> &str;

    /// Return a short source tag used in file naming and provenance (e.g., "dunbrack2010_simpleopt1").
    fn source_tag(&self) -> &str;
}
