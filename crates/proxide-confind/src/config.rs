//! Typed configuration for the ConFind pipeline.
//!
//! All behaviour-selecting parameters are carried by one [`ConFindConfig`]. Options are
//! orthogonal: each affects a documented set of outputs and every combination of valid
//! values is legal. The only restrictions are the validation rules below; nothing is
//! silently corrected or replaced by a fallback.
//!
//! | Option | Affects |
//! |---|---|
//! | [`CbClashPolicy`] | surviving rotamer sets, hence every output |
//! | [`CrowdednessMode`] | `crowdedness` only |
//! | [`FreedomMode`] | `freedom` only |
//! | [`ResidueScope`] | `freedom` only (through the neighbour set) |
//! | [`Cutoffs`] | `d_clash`: every output; `d_cont`: contacts and freedom; `d_cut`: contacts and freedom (exactness-preserving when valid); `tau_*`: freedom |
//! | `threads` | no numeric output. `None` is the orx-parallel default; `Some(n)` limits every parallel phase |
//!
//! # Validation
//!
//! - **V1** every cutoff is finite and `> 0` ([`ConfigError::InvalidCutoff`]).
//! - **V2** `tau_lo < tau_hi` when the freedom mode is [`FreedomMode::Rms`]
//!   ([`ConfigError::TauOrder`]); [`FreedomMode::SingleThreshold`] only needs V1 for `tau_hi`.
//! - **V3** a [`ResidueScope::Subset`] is non-empty ([`ConfigError::EmptyScope`]).
//! - **V4** string parsing of a variant name rejects unknown names
//!   ([`ConfigError::UnknownVariant`]).
//! - **V5** at construction, `d_cut >= 2 * max_a R_max(a) + d_cont` for the loaded rotamer
//!   library ([`ConfigError::DcutTooSmall`]).
//! - **V6** at construction, every scope index is below the residue count
//!   ([`ConfigError::ScopeOutOfRange`]).
//!
//! [`ConFindConfig::validate`] performs V1–V3; construction performs V1, V2, V3, V6, V5 in
//! that order.

use crate::coords::ResidueIndex;
use crate::params::{CLASH_DIST, CONT_DIST, DCUT, HI_COLL_PROB_CUT, LO_COLL_PROB_CUT};
use std::fmt;
use std::num::NonZeroUsize;
use std::str::FromStr;

/// Configuration validation error.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ConfigError {
    /// V1: a cutoff is non-finite or not strictly positive.
    #[error("cutoff {name} must be finite and > 0 (got {value})")]
    InvalidCutoff {
        /// Name of the offending cutoff field.
        name: &'static str,
        /// The offending value.
        value: f64,
    },
    /// V2: `tau_lo >= tau_hi` under [`FreedomMode::Rms`].
    #[error("tau_lo ({tau_lo}) must be < tau_hi ({tau_hi}) when freedom_mode is Rms")]
    TauOrder {
        /// Configured lower threshold.
        tau_lo: f64,
        /// Configured upper threshold.
        tau_hi: f64,
    },
    /// V5: `d_cut` is smaller than the library-derived minimum.
    #[error("d_cut {d_cut} Å < required {required} Å (= 2*max R_max + d_cont) for the loaded rotamer library")]
    DcutTooSmall {
        /// Configured neighbour cutoff.
        d_cut: f64,
        /// Required minimum `2 * max_a R_max(a) + d_cont`.
        required: f64,
    },
    /// V3: an empty scope subset.
    #[error("scope subset is empty")]
    EmptyScope,
    /// V6: a scope index outside the structure.
    #[error("scope contains residue index {index} but structure has {n_residues} residues")]
    ScopeOutOfRange {
        /// Offending residue index.
        index: u32,
        /// Number of residues in the structure.
        n_residues: usize,
    },
    /// V4: unknown variant name when parsing.
    #[error("unknown {option} value '{value}'; expected one of: {expected}")]
    UnknownVariant {
        /// Option name.
        option: &'static str,
        /// The rejected string.
        value: String,
        /// The allowed values.
        expected: &'static str,
    },
}

/// Which rotamers' Cβ atom is included in the backbone-clash test.
///
/// Variants: `AlaOnly`, `AllResidues`. Default: `AllResidues`.
/// Defines the clash atom set `S_clash`; affects the surviving rotamer sets and therefore
/// every output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CbClashPolicy {
    /// Cβ is tested against backbone for alanine only.
    AlaOnly,
    /// Cβ is tested against backbone for every amino acid.
    #[default]
    AllResidues,
}

/// How crowdedness `C(i)` aggregates pruned rotamers.
///
/// Variants: `ProbabilityWeighted`, `PrunedCountFraction`. Default: `ProbabilityWeighted`.
/// Affects `ConFind::crowdedness` only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CrowdednessMode {
    /// `C = 1 - W_surv / W_lib` (prior-probability weighted).
    #[default]
    ProbabilityWeighted,
    /// `C = 1 - N_surv / N_lib` (rotamer count).
    PrunedCountFraction,
}

/// How freedom `F(i)` combines threshold counts.
///
/// Variants: `Rms`, `SingleThreshold`. Default: `Rms`.
/// Affects `ConFind::freedom` only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FreedomMode {
    /// `sqrt((N_lo^2 + N_hi^2) / 2) / N_lib`.
    #[default]
    Rms,
    /// `N_lo / N_lib`.
    SingleThreshold,
}

/// Residues eligible as neighbours in the collision accumulation of freedom.
///
/// Variants: `All`, `Subset(..)`. Default: `All`. Affects `ConFind::freedom` only
/// (through the neighbour set; `freedom(i)` for `i` outside the scope is an
/// `OutOfScope` error).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ResidueScope {
    /// Every residue of the structure.
    #[default]
    All,
    /// An explicit set of residues; duplicates are permitted and ignored.
    Subset(Vec<ResidueIndex>),
}

/// Distance and threshold cutoffs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cutoffs {
    /// Backbone clash distance in Å (inclusive). Default 2.0.
    pub d_clash: f64,
    /// Sidechain contact distance in Å (inclusive). Default 3.0.
    pub d_cont: f64,
    /// Cα–Cα neighbour cutoff in Å (inclusive). Default 25.0.
    pub d_cut: f64,
    /// Lower collision-mass threshold (dimensionless, strict). Default 0.5.
    pub tau_lo: f64,
    /// Upper collision-mass threshold (dimensionless, strict). Default 2.0.
    pub tau_hi: f64,
}

impl Default for Cutoffs {
    fn default() -> Self {
        Self {
            d_clash: CLASH_DIST,
            d_cont: CONT_DIST,
            d_cut: DCUT,
            tau_lo: LO_COLL_PROB_CUT,
            tau_hi: HI_COLL_PROB_CUT,
        }
    }
}

/// Complete ConFind configuration.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ConFindConfig {
    /// Distance and threshold cutoffs.
    pub cutoffs: Cutoffs,
    /// Freedom analysis scope.
    pub scope: ResidueScope,
    /// Backbone-clash atom policy.
    pub cb_clash_policy: CbClashPolicy,
    /// Crowdedness aggregation mode.
    pub crowdedness_mode: CrowdednessMode,
    /// Freedom combination mode.
    pub freedom_mode: FreedomMode,
    /// Worker limit for every parallel phase (Phase A cache, Phase B contacts, and the
    /// CLI freedom map).
    ///
    /// `None` leaves the orx-parallel default. `Some(n)` applies `.num_threads(n)` on each
    /// `into_par()` in this crate. Spec §3.9 does not define this field; it changes no
    /// numeric result. Zero is not representable.
    pub threads: Option<NonZeroUsize>,
}

impl ConFindConfig {
    /// Replace the cutoffs.
    pub fn with_cutoffs(mut self, c: Cutoffs) -> Self {
        self.cutoffs = c;
        self
    }

    /// Replace the analysis scope.
    pub fn with_scope(mut self, s: ResidueScope) -> Self {
        self.scope = s;
        self
    }

    /// Replace the Cβ clash policy.
    pub fn with_cb_clash_policy(mut self, p: CbClashPolicy) -> Self {
        self.cb_clash_policy = p;
        self
    }

    /// Replace the crowdedness mode.
    pub fn with_crowdedness_mode(mut self, m: CrowdednessMode) -> Self {
        self.crowdedness_mode = m;
        self
    }

    /// Replace the freedom mode.
    pub fn with_freedom_mode(mut self, m: FreedomMode) -> Self {
        self.freedom_mode = m;
        self
    }

    /// Replace the worker limit. `None` restores the orx-parallel default.
    pub fn with_threads(mut self, threads: Option<NonZeroUsize>) -> Self {
        self.threads = threads;
        self
    }

    /// Context-free validation: rules V1, V2 and V3 (V4 applies to string parsing; V5 and V6
    /// need the library and structure and are checked at construction).
    pub fn validate(&self) -> Result<(), ConfigError> {
        let c = &self.cutoffs;
        for (name, value) in [
            ("d_clash", c.d_clash),
            ("d_cont", c.d_cont),
            ("d_cut", c.d_cut),
            ("tau_lo", c.tau_lo),
            ("tau_hi", c.tau_hi),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ConfigError::InvalidCutoff { name, value });
            }
        }
        if self.freedom_mode == FreedomMode::Rms && c.tau_lo >= c.tau_hi {
            return Err(ConfigError::TauOrder {
                tau_lo: c.tau_lo,
                tau_hi: c.tau_hi,
            });
        }
        if let ResidueScope::Subset(v) = &self.scope {
            if v.is_empty() {
                return Err(ConfigError::EmptyScope);
            }
        }
        Ok(())
    }
}

macro_rules! kebab_enum {
    ($ty:ident, $option:literal, $expected:literal, { $($variant:ident => $name:literal),+ $(,)? }) => {
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(match self { $( $ty::$variant => $name ),+ })
            }
        }

        impl FromStr for $ty {
            type Err = ConfigError;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $( $name => Ok($ty::$variant), )+
                    _ => Err(ConfigError::UnknownVariant {
                        option: $option,
                        value: s.to_string(),
                        expected: $expected,
                    }),
                }
            }
        }
    };
}

kebab_enum!(CbClashPolicy, "cbclash", "ala-only, all-residues", {
    AlaOnly => "ala-only",
    AllResidues => "all-residues",
});
kebab_enum!(
    CrowdednessMode,
    "crowdmode",
    "probability-weighted, pruned-count-fraction",
    {
        ProbabilityWeighted => "probability-weighted",
        PrunedCountFraction => "pruned-count-fraction",
    }
);
kebab_enum!(FreedomMode, "freedommode", "rms, single-threshold", {
    Rms => "rms",
    SingleThreshold => "single-threshold",
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_fromstr_roundtrip() {
        for p in [CbClashPolicy::AlaOnly, CbClashPolicy::AllResidues] {
            assert_eq!(p.to_string().parse::<CbClashPolicy>().unwrap(), p);
        }
        for m in [
            CrowdednessMode::ProbabilityWeighted,
            CrowdednessMode::PrunedCountFraction,
        ] {
            assert_eq!(m.to_string().parse::<CrowdednessMode>().unwrap(), m);
        }
        for m in [FreedomMode::Rms, FreedomMode::SingleThreshold] {
            assert_eq!(m.to_string().parse::<FreedomMode>().unwrap(), m);
        }
    }

    #[test]
    fn unknown_variant_names_option() {
        let e = "bogus".parse::<FreedomMode>().unwrap_err();
        assert!(matches!(
            e,
            ConfigError::UnknownVariant {
                option: "freedommode",
                ..
            }
        ));
    }
}
