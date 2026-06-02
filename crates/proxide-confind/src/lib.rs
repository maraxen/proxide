//! Contact-degree algorithm for protein residue pairs (proxide-confind).
//!
//! Implements the ConFind contact-degree metric — a Rust/rayon port of
//! Mosaist's `mstcondeg`. Given a protein backbone and a rotamer library,
//! ConFind estimates how much rotamer freedom each residue loses due to
//! sidechain clashes with its neighbours.
//!
//! # Algorithm phases
//!
//! - **Phase A — cache:** for each residue, enumerate its rotamers, prune
//!   those clashing with backbone atoms, and store crowdedness
//!   (fraction pruned).
//! - **Phase B — contacts:** for each residue pair within the cutoff,
//!   enumerate cross-rotamer clashes in parallel (rayon) to compute
//!   collision probabilities.
//! - **Phase C — freedom:** aggregate collision probabilities into a scalar
//!   rotamer-freedom value per residue.
//!
//! # Typical usage
//!
//! ```no_run
//! use proxide_confind::{ConFind, ResidueIndex};
//! use proxide_rotlib::RotamerLibrary;
//! use std::{path::Path, sync::Arc};
//!
//! let rotlib = Arc::new(RotamerLibrary::load(Path::new("rotlib.bin")).unwrap());
//! let backbone = Arc::new(proxide_confind::load_pdb_f64(Path::new("protein.pdb")).unwrap());
//! let cf = ConFind::new(rotlib, backbone, false);
//! let residues = vec![ResidueIndex(0), ResidueIndex(1)];
//! let contacts = cf.contacts(&residues, 0.0).unwrap();
//! ```
//!
//! # References
//!
//! - Mosaist protein design suite: <https://grigoryanlab.org/mosaist/>
//! - Grigoryan G, Keating AE. "Structural specificity in coiled-coil
//!   interactions." *Curr Opin Struct Biol.* 2008;18(4):477-483.

pub mod cache;
pub mod confind;
pub mod contact_list;
pub mod coords;
pub mod error;
pub mod freedom;
pub mod grid;
pub mod parallel;
pub mod params;

pub use confind::ConFind;
pub use contact_list::{ContactList, CONTACT_THRESHOLD};
pub use coords::{extract_f64_backbone, load_pdb_f64, ProteinBackbone, ResidueBackbone, ResidueIndex};
pub use error::ConFindError;
pub use freedom::compute_freedom;
pub use params::{aa_propensity, AA_NAMES, CLASH_DIST, CONT_DIST, DCUT, HI_COLL_PROB, LO_COLL_PROB};
