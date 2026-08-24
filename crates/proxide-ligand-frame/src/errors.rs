use thiserror::Error;

/// Typed error surface for `canonicalize_ligand_topology` and
/// `extract_ligand_frame_coordinates` (spec §4, closes Finding 8/10).
#[derive(Debug, Error, Clone, PartialEq)]
pub enum LigandFrameError {
    #[error(
        "input graph has {component_count} disconnected components; ligand-frame v1 requires \
         a single connected molecular graph (split cofactors/counter-ions into separate calls)"
    )]
    DisconnectedGraph { component_count: usize },

    #[error("unsupported element: {element}")]
    UnsupportedElement { element: String },

    #[error("invalid valence at atom index {atom_index}")]
    InvalidValence { atom_index: usize },

    #[error("SSSR/aromaticity input inconsistent with bonds_in: {reason}")]
    SssrInputInvalid { reason: String },

    #[error("invalid bond order {order}: expected 1 (single), 2 (double), or 3 (triple)")]
    InvalidBondOrder { order: u8 },

    #[error("Espaloma charge inference failed: {reason}")]
    ChargeInferenceFailure { reason: String },

    #[error("reference geometry failed validation: {reason}")]
    InvalidReferenceGeometry { reason: String },

    #[error(
        "topology/positions atom count mismatch: topology expects {expected_atoms} atoms, got {got_atoms}"
    )]
    TopologyPositionMismatch {
        expected_atoms: usize,
        got_atoms: usize,
    },
}
