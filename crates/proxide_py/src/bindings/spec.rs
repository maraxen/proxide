use proxide_rs::spec::{CoordFormat, ErrorMode, HydrogenSource, MissingResidueMode, OutputSpec};
use proxide_units::UnitSystem;
use pyo3::prelude::*;

#[pyclass(name = "CoordFormat", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyCoordFormat {
    Atom37,
    Atom14,
    Full,
    BackboneOnly,
}

impl From<CoordFormat> for PyCoordFormat {
    fn from(f: CoordFormat) -> Self {
        match f {
            CoordFormat::Atom37 => Self::Atom37,
            CoordFormat::Atom14 => Self::Atom14,
            CoordFormat::Full => Self::Full,
            CoordFormat::BackboneOnly => Self::BackboneOnly,
        }
    }
}

impl From<PyCoordFormat> for CoordFormat {
    fn from(f: PyCoordFormat) -> Self {
        match f {
            PyCoordFormat::Atom37 => Self::Atom37,
            PyCoordFormat::Atom14 => Self::Atom14,
            PyCoordFormat::Full => Self::Full,
            PyCoordFormat::BackboneOnly => Self::BackboneOnly,
        }
    }
}

#[pyclass(name = "ErrorMode", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyErrorMode {
    Warn,
    Skip,
    Fail,
}

impl From<ErrorMode> for PyErrorMode {
    fn from(m: ErrorMode) -> Self {
        match m {
            ErrorMode::Warn => Self::Warn,
            ErrorMode::Skip => Self::Skip,
            ErrorMode::Fail => Self::Fail,
        }
    }
}

impl From<PyErrorMode> for ErrorMode {
    fn from(m: PyErrorMode) -> Self {
        match m {
            PyErrorMode::Warn => Self::Warn,
            PyErrorMode::Skip => Self::Skip,
            PyErrorMode::Fail => Self::Fail,
        }
    }
}

#[pyclass(name = "MissingResidueMode", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyMissingResidueMode {
    SkipWarn,
    Fail,
    GaffFallback,
    ClosestMatch,
}

impl From<MissingResidueMode> for PyMissingResidueMode {
    fn from(m: MissingResidueMode) -> Self {
        match m {
            MissingResidueMode::SkipWarn => Self::SkipWarn,
            MissingResidueMode::Fail => Self::Fail,
            MissingResidueMode::GaffFallback => Self::GaffFallback,
            MissingResidueMode::ClosestMatch => Self::ClosestMatch,
        }
    }
}

impl From<PyMissingResidueMode> for MissingResidueMode {
    fn from(m: PyMissingResidueMode) -> Self {
        match m {
            PyMissingResidueMode::SkipWarn => Self::SkipWarn,
            PyMissingResidueMode::Fail => Self::Fail,
            PyMissingResidueMode::GaffFallback => Self::GaffFallback,
            PyMissingResidueMode::ClosestMatch => Self::ClosestMatch,
        }
    }
}

#[pyclass(name = "HydrogenSource", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyHydrogenSource {
    ForceFieldFirst,
    FragmentLibrary,
    ForceFieldOnly,
}

impl From<HydrogenSource> for PyHydrogenSource {
    fn from(s: HydrogenSource) -> Self {
        match s {
            HydrogenSource::ForceFieldFirst => Self::ForceFieldFirst,
            HydrogenSource::FragmentLibrary => Self::FragmentLibrary,
            HydrogenSource::ForceFieldOnly => Self::ForceFieldOnly,
        }
    }
}

impl From<PyHydrogenSource> for HydrogenSource {
    fn from(s: PyHydrogenSource) -> Self {
        match s {
            PyHydrogenSource::ForceFieldFirst => Self::ForceFieldFirst,
            PyHydrogenSource::FragmentLibrary => Self::FragmentLibrary,
            PyHydrogenSource::ForceFieldOnly => Self::ForceFieldOnly,
        }
    }
}

#[pyclass(name = "UnitSystem", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyUnitSystem {
    /// AMBER units: Angstroms (length), kcal/mol (energy). Default.
    Amber,
    /// GROMACS units: nm (length), kJ/mol (energy). Pass-through (no conversion).
    Gromacs,
    /// OpenMM units: nm (length), kJ/mol (energy). Same as Gromacs internally.
    OpenMM,
}

impl From<UnitSystem> for PyUnitSystem {
    fn from(u: UnitSystem) -> Self {
        match u {
            UnitSystem::Amber => Self::Amber,
            UnitSystem::Gromacs => Self::Gromacs,
            UnitSystem::OpenMM => Self::OpenMM,
        }
    }
}

impl From<PyUnitSystem> for UnitSystem {
    fn from(u: PyUnitSystem) -> Self {
        match u {
            PyUnitSystem::Amber => Self::Amber,
            PyUnitSystem::Gromacs => Self::Gromacs,
            PyUnitSystem::OpenMM => Self::OpenMM,
        }
    }
}

#[pyclass(name = "OutputSpec")]
#[derive(Debug, Clone)]
pub struct PyOutputSpec {
    pub inner: OutputSpec,
}

#[pymethods]
impl PyOutputSpec {
    #[new]
    #[pyo3(signature = (
        coord_format = PyCoordFormat::Atom37,
        output_format_target = "general".to_string(),
        models = None,
        chains = None,
        remove_hetatm = false,
        include_hetatm = false,
        remove_solvent = true,
        residue_range = None,
        add_hydrogens = false,
        hydrogen_source = PyHydrogenSource::ForceFieldFirst,
        relax_hydrogens = false,
        relax_max_iterations = None,
        infer_bonds = false,
        compute_rbf = false,
        rbf_num_neighbors = 30,
        compute_electrostatics = false,
        electrostatics_noise = None,
        compute_vdw = false,
        parameterize_md = false,
        force_field = None,
        auto_terminal_caps = true,
        missing_residue_mode = PyMissingResidueMode::Fail,
        ph = Some(7.0),
        include_b_factors = false,
        include_occupancy = false,
        error_mode = PyErrorMode::Warn,
        enable_caching = false,
        unit_system = PyUnitSystem::Amber
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        coord_format: PyCoordFormat,
        output_format_target: String,
        models: Option<Vec<usize>>,
        chains: Option<Vec<String>>,
        remove_hetatm: bool,
        include_hetatm: bool,
        remove_solvent: bool,
        residue_range: Option<(i32, i32)>,
        add_hydrogens: bool,
        hydrogen_source: PyHydrogenSource,
        relax_hydrogens: bool,
        relax_max_iterations: Option<usize>,
        infer_bonds: bool,
        compute_rbf: bool,
        rbf_num_neighbors: usize,
        compute_electrostatics: bool,
        electrostatics_noise: Option<f32>,
        compute_vdw: bool,
        parameterize_md: bool,
        force_field: Option<String>,
        auto_terminal_caps: bool,
        missing_residue_mode: PyMissingResidueMode,
        ph: Option<f32>,
        include_b_factors: bool,
        include_occupancy: bool,
        error_mode: PyErrorMode,
        enable_caching: bool,
        unit_system: PyUnitSystem,
    ) -> Self {
        Self {
            inner: OutputSpec {
                coord_format: coord_format.into(),
                output_format_target,
                models,
                chains,
                remove_hetatm,
                include_hetatm,
                remove_solvent,
                residue_range,
                add_hydrogens,
                hydrogen_source: hydrogen_source.into(),
                relax_hydrogens,
                relax_max_iterations,
                infer_bonds,
                compute_rbf,
                rbf_num_neighbors,
                compute_electrostatics,
                electrostatics_noise,
                compute_vdw,
                parameterize_md,
                force_field,
                auto_terminal_caps,
                missing_residue_mode: missing_residue_mode.into(),
                ph,
                include_b_factors,
                include_occupancy,
                error_mode: error_mode.into(),
                enable_caching,
                unit_system: unit_system.into(),
            },
        }
    }

    // --- Getters and Setters ---
    #[getter]
    fn get_coord_format(&self) -> PyCoordFormat {
        self.inner.coord_format.into()
    }
    #[setter]
    fn set_coord_format(&mut self, value: PyCoordFormat) {
        self.inner.coord_format = value.into();
    }

    #[getter]
    fn get_enable_caching(&self) -> bool {
        self.inner.enable_caching
    }
    #[setter]
    fn set_enable_caching(&mut self, val: bool) {
        self.inner.enable_caching = val;
    }

    #[getter]
    fn get_output_format_target(&self) -> String {
        self.inner.output_format_target.clone()
    }
    #[setter]
    fn set_output_format_target(&mut self, val: String) {
        self.inner.output_format_target = val;
    }

    #[getter]
    fn get_remove_solvent(&self) -> bool {
        self.inner.remove_solvent
    }
    #[setter]
    fn set_remove_solvent(&mut self, val: bool) {
        self.inner.remove_solvent = val;
    }

    #[getter]
    fn get_include_hetatm(&self) -> bool {
        self.inner.include_hetatm
    }
    #[setter]
    fn set_include_hetatm(&mut self, val: bool) {
        self.inner.include_hetatm = val;
    }

    #[getter]
    fn get_compute_rbf(&self) -> bool {
        self.inner.compute_rbf
    }
    #[setter]
    fn set_compute_rbf(&mut self, val: bool) {
        self.inner.compute_rbf = val;
    }

    #[getter]
    fn get_rbf_num_neighbors(&self) -> usize {
        self.inner.rbf_num_neighbors
    }
    #[setter]
    fn set_rbf_num_neighbors(&mut self, val: usize) {
        self.inner.rbf_num_neighbors = val;
    }

    #[getter]
    fn get_compute_electrostatics(&self) -> bool {
        self.inner.compute_electrostatics
    }
    #[setter]
    fn set_compute_electrostatics(&mut self, val: bool) {
        self.inner.compute_electrostatics = val;
    }

    #[getter]
    fn get_compute_vdw(&self) -> bool {
        self.inner.compute_vdw
    }
    #[setter]
    fn set_compute_vdw(&mut self, val: bool) {
        self.inner.compute_vdw = val;
    }

    #[getter]
    fn get_parameterize_md(&self) -> bool {
        self.inner.parameterize_md
    }
    #[setter]
    fn set_parameterize_md(&mut self, val: bool) {
        self.inner.parameterize_md = val;
    }

    #[getter]
    fn get_ph(&self) -> Option<f32> {
        self.inner.ph
    }
    #[setter]
    fn set_ph(&mut self, val: Option<f32>) {
        self.inner.ph = val;
    }

    #[getter]
    fn get_infer_bonds(&self) -> bool {
        self.inner.infer_bonds
    }
    #[setter]
    fn set_infer_bonds(&mut self, val: bool) {
        self.inner.infer_bonds = val;
    }

    #[getter]
    fn get_add_hydrogens(&self) -> bool {
        self.inner.add_hydrogens
    }
    #[setter]
    fn set_add_hydrogens(&mut self, val: bool) {
        self.inner.add_hydrogens = val;
    }

    #[getter]
    fn get_relax_hydrogens(&self) -> bool {
        self.inner.relax_hydrogens
    }
    #[setter]
    fn set_relax_hydrogens(&mut self, val: bool) {
        self.inner.relax_hydrogens = val;
    }

    #[getter]
    fn get_auto_terminal_caps(&self) -> bool {
        self.inner.auto_terminal_caps
    }
    #[setter]
    fn set_auto_terminal_caps(&mut self, val: bool) {
        self.inner.auto_terminal_caps = val;
    }

    #[getter]
    fn get_unit_system(&self) -> PyUnitSystem {
        self.inner.unit_system.into()
    }
    #[setter]
    fn set_unit_system(&mut self, value: PyUnitSystem) {
        self.inner.unit_system = value.into();
    }
}
