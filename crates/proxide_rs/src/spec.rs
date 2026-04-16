//! Output specification for structure formatting
//!
//! Defines the OutputSpec struct that controls how structures are formatted
//! and what optional processing steps should be applied.

/// Coordinate format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordFormat {
    /// Atom37 format: (N_res, 37, 3) - standard AlphaFold representation
    Atom37,
    /// Atom14 format: (N_res, 14, 3) - reduced representation
    Atom14,
    /// Full format: all atoms with padding
    Full,
    /// Backbone only: (N_res, 4, 3) - N, CA, C, O
    BackboneOnly,
}

/// Output format target for backbone ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormatTarget {
    /// General format: 0:N, 1:CA, 2:C, 3:CB, 4:O (standard Atom37)
    #[default]
    General,
    /// MPNN format: 0:N, 1:CA, 2:C, 3:O, 4:CB (PrxteinMPNN compatible)
    Mpnn,
}

impl OutputFormatTarget {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "general" => Ok(Self::General),
            "mpnn" => Ok(Self::Mpnn),
            _ => Err(format!(
                "Invalid output_format_target: {}. Must be 'general' or 'mpnn'",
                s
            )),
        }
    }
}

/// Error handling mode for missing atoms/residues
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorMode {
    /// Log warnings and continue
    Warn,
    /// Skip problematic atoms/residues silently
    Skip,
    /// Fail entire structure on error
    Fail,
}

/// How to handle missing residue templates during parameterization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingResidueMode {
    /// Skip residue and log warning
    SkipWarn,
    /// Fail with error
    #[default]
    Fail,
    /// Try GAFF fallback (future - not implemented)
    GaffFallback,
    /// Match closest residue by shared atom names
    ClosestMatch,
}

/// Source for hydrogen atom placement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HydrogenSource {
    /// Use force field templates first, fallback to fragment library
    #[default]
    ForceFieldFirst,
    /// Use fragment library only (geometric placement via Kabsch)
    FragmentLibrary,
    /// Use force field templates only (fail if not defined)
    ForceFieldOnly,
}

/// Output specification for structure formatting
#[derive(Debug, Clone)]
pub struct OutputSpec {
    // Format
    pub coord_format: CoordFormat,
    pub output_format_target: String,

    // Filtering
    pub models: Option<Vec<usize>>,
    pub chains: Option<Vec<String>>,
    pub remove_hetatm: bool,
    pub include_hetatm: bool,
    pub remove_solvent: bool,
    pub residue_range: Option<(i32, i32)>,

    // Processing
    pub add_hydrogens: bool,
    pub hydrogen_source: HydrogenSource,
    pub relax_hydrogens: bool,
    pub relax_max_iterations: Option<usize>,
    pub infer_bonds: bool,

    // Geometry Features
    pub compute_rbf: bool,
    pub rbf_num_neighbors: usize,

    // Physics Features
    pub compute_electrostatics: bool,
    pub electrostatics_noise: Option<f32>,
    pub compute_vdw: bool,
    pub parameterize_md: bool,
    pub force_field: Option<String>,
    pub auto_terminal_caps: bool,
    pub missing_residue_mode: MissingResidueMode,

    // Environment
    pub ph: Option<f32>,

    // Optional fields
    pub include_b_factors: bool,
    pub include_occupancy: bool,

    // Error handling
    pub error_mode: ErrorMode,

    // Performance
    pub enable_caching: bool,
}

impl Default for OutputSpec {
    fn default() -> Self {
        Self {
            coord_format: CoordFormat::Atom37,
            output_format_target: "general".to_string(),
            models: None,
            chains: None,
            remove_hetatm: false,
            include_hetatm: false,
            remove_solvent: true,
            residue_range: None,
            add_hydrogens: false,
            hydrogen_source: HydrogenSource::ForceFieldFirst,
            relax_hydrogens: false,
            relax_max_iterations: None,
            infer_bonds: false,
            compute_rbf: false,
            rbf_num_neighbors: 30,
            compute_electrostatics: false,
            electrostatics_noise: None,
            compute_vdw: false,
            parameterize_md: false,
            force_field: None,
            auto_terminal_caps: true,
            missing_residue_mode: MissingResidueMode::Fail,
            ph: Some(7.0),
            include_b_factors: false,
            include_occupancy: false,
            error_mode: ErrorMode::Warn,
            enable_caching: false,
        }
    }
}
