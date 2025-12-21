//! Output specification for structure formatting
//!
//! Defines the OutputSpec struct that controls how structures are formatted
//! and what optional processing steps should be applied.

use pyo3::prelude::*;

/// Coordinate format options
#[pyclass(eq, eq_int)]
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

    pub fn to_str(&self) -> &'static str {
        match self {
            Self::General => "general",
            Self::Mpnn => "mpnn",
        }
    }
}

/// Error handling mode for missing atoms/residues
#[pyclass(eq, eq_int)]
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
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingResidueMode {
    /// Skip residue and log warning (default)
    #[default]
    SkipWarn,
    /// Fail with error
    Fail,
    /// Try GAFF fallback (future - not implemented)
    GaffFallback,
    /// Match closest residue by shared atom names
    ClosestMatch,
}

/// Source for hydrogen atom placement
#[pyclass(eq, eq_int)]
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
#[pyclass]
#[derive(Debug, Clone)]
pub struct OutputSpec {
    // Format
    #[pyo3(get, set)]
    pub coord_format: CoordFormat,
    #[pyo3(get, set)]
    pub output_format_target: String,

    // Filtering
    #[pyo3(get, set)]
    pub models: Option<Vec<usize>>,
    #[pyo3(get, set)]
    pub chains: Option<Vec<String>>,
    #[pyo3(get, set)]
    pub remove_hetatm: bool,
    #[pyo3(get, set)]
    pub include_hetatm: bool,
    #[pyo3(get, set)]
    pub remove_solvent: bool,
    #[pyo3(get, set)]
    pub residue_range: Option<(i32, i32)>,

    // Processing
    #[pyo3(get, set)]
    pub add_hydrogens: bool,
    #[pyo3(get, set)]
    pub hydrogen_source: HydrogenSource,
    #[pyo3(get, set)]
    pub relax_hydrogens: bool,
    #[pyo3(get, set)]
    pub relax_max_iterations: Option<usize>,
    #[pyo3(get, set)]
    pub infer_bonds: bool,

    // Geometry Features
    #[pyo3(get, set)]
    pub compute_rbf: bool,
    #[pyo3(get, set)]
    pub rbf_num_neighbors: usize,

    // Physics Features
    #[pyo3(get, set)]
    pub compute_electrostatics: bool,
    #[pyo3(get, set)]
    pub electrostatics_noise: Option<f32>,
    #[pyo3(get, set)]
    pub compute_vdw: bool,
    #[pyo3(get, set)]
    pub parameterize_md: bool,
    #[pyo3(get, set)]
    pub force_field: Option<String>,
    #[pyo3(get, set)]
    pub auto_terminal_caps: bool,
    #[pyo3(get, set)]
    pub missing_residue_mode: MissingResidueMode,

    // Optional fields
    #[pyo3(get, set)]
    pub include_b_factors: bool,
    #[pyo3(get, set)]
    pub include_occupancy: bool,

    // Error handling
    #[pyo3(get, set)]
    pub error_mode: ErrorMode,

    // Performance
    #[pyo3(get, set)]
    pub enable_caching: bool,
}

#[pymethods]
impl OutputSpec {
    #[new]
    #[pyo3(signature = (
        coord_format=CoordFormat::Atom37,
        output_format_target="general",
        models=None,
        chains=None,
        remove_hetatm=false,
        include_hetatm=false,
        remove_solvent=true,
        residue_range=None,
        add_hydrogens=false,
        hydrogen_source=HydrogenSource::ForceFieldFirst,
        relax_hydrogens=false,
        relax_max_iterations=None,
        infer_bonds=false,
        compute_rbf=false,
        rbf_num_neighbors=30,
        compute_electrostatics=false,
        compute_vdw=false,
        parameterize_md=false,
        force_field=None,
        auto_terminal_caps=true,
        missing_residue_mode=MissingResidueMode::SkipWarn,
        include_b_factors=false,
        include_occupancy=false,
        error_mode=ErrorMode::Warn,
        enable_caching=false,
    ))]
    pub fn new(
        coord_format: CoordFormat,
        output_format_target: &str,
        models: Option<Vec<usize>>,
        chains: Option<Vec<String>>,
        remove_hetatm: bool,
        include_hetatm: bool,
        remove_solvent: bool,
        residue_range: Option<(i32, i32)>,
        add_hydrogens: bool,
        hydrogen_source: HydrogenSource,
        relax_hydrogens: bool,
        relax_max_iterations: Option<usize>,
        infer_bonds: bool,
        compute_rbf: bool,
        rbf_num_neighbors: usize,
        compute_electrostatics: bool,
        compute_vdw: bool,
        parameterize_md: bool,
        force_field: Option<String>,
        auto_terminal_caps: bool,
        missing_residue_mode: MissingResidueMode,
        include_b_factors: bool,
        include_occupancy: bool,
        error_mode: ErrorMode,
        enable_caching: bool,
    ) -> Self {
        OutputSpec {
            coord_format,
            output_format_target: output_format_target.to_string(),
            models,
            chains,
            remove_hetatm,
            include_hetatm,
            remove_solvent,
            residue_range,
            add_hydrogens,
            hydrogen_source,
            relax_hydrogens,
            relax_max_iterations,
            infer_bonds,
            compute_rbf,
            rbf_num_neighbors,
            compute_electrostatics,
            electrostatics_noise: None,
            compute_vdw,
            parameterize_md,
            force_field,
            auto_terminal_caps,
            missing_residue_mode,
            include_b_factors,
            include_occupancy,
            error_mode,
            enable_caching,
        }
    }
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
            missing_residue_mode: MissingResidueMode::SkipWarn,
            include_b_factors: false,
            include_occupancy: false,
            error_mode: ErrorMode::Warn,
            enable_caching: false,
        }
    }
}
