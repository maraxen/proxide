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

// TODO: Review allow attributes at a later point
#[allow(clippy::too_many_arguments)]
#[pymethods]
impl OutputSpec {
    #[new]
    #[pyo3(signature = (**kwargs))]
    pub fn new(kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let mut spec = OutputSpec::default();
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.iter() {
                let key_str: &str = key.extract()?;
                match key_str {
                    "coord_format" => spec.coord_format = value.extract()?,
                    "output_format_target" => spec.output_format_target = value.extract()?,
                    "models" => spec.models = value.extract()?,
                    "chains" => spec.chains = value.extract()?,
                    "remove_hetatm" => spec.remove_hetatm = value.extract()?,
                    "include_hetatm" => spec.include_hetatm = value.extract()?,
                    "remove_solvent" => spec.remove_solvent = value.extract()?,
                    "residue_range" => spec.residue_range = value.extract()?,
                    "add_hydrogens" => spec.add_hydrogens = value.extract()?,
                    "hydrogen_source" => spec.hydrogen_source = value.extract()?,
                    "relax_hydrogens" => spec.relax_hydrogens = value.extract()?,
                    "relax_max_iterations" => spec.relax_max_iterations = value.extract()?,
                    "infer_bonds" => spec.infer_bonds = value.extract()?,
                    "compute_rbf" => spec.compute_rbf = value.extract()?,
                    "rbf_num_neighbors" => spec.rbf_num_neighbors = value.extract()?,
                    "compute_electrostatics" => spec.compute_electrostatics = value.extract()?,
                    "compute_vdw" => spec.compute_vdw = value.extract()?,
                    "parameterize_md" => spec.parameterize_md = value.extract()?,
                    "force_field" => spec.force_field = value.extract()?,
                    "auto_terminal_caps" => spec.auto_terminal_caps = value.extract()?,
                    "missing_residue_mode" => spec.missing_residue_mode = value.extract()?,
                    "include_b_factors" => spec.include_b_factors = value.extract()?,
                    "include_occupancy" => spec.include_occupancy = value.extract()?,
                    "error_mode" => spec.error_mode = value.extract()?,
                    "enable_caching" => spec.enable_caching = value.extract()?,
                    _ => {
                        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                            "Unexpected keyword argument: {}",
                            key_str
                        )))
                    }
                }
            }
        }
        Ok(spec)
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
