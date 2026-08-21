//! Ring perception for GAFF2 atom typing.
//!
//! Phase 1: accepts caller-supplied SSSR from RDKit (via `MolGraph.rings`).
//! Phase 2 (gated separately): native ring detection and SSSR computation.
//!
//! Top parity risk: RDKit's GetRingInfo().AtomRings() returns *symmetrized* SSSR,
//! not textbook SSSR. Fused and cage systems may disagree. Validation against
//! phase-1 oracle (RDKit) is a separate, independently-gated change.

pub fn todo_ring_detection() {
    todo!("Phase 2: native ring perception — NOT in phase 1")
}
