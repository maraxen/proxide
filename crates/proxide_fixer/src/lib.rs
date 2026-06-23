pub mod models;
pub mod templates;
pub mod finder;
pub mod builder;
pub mod repack;
pub mod pipeline;
pub mod solvate;
pub mod loop_model;
pub mod mutate;

#[cfg(feature = "amber_ff")]
pub mod amber_ff;

#[cfg(feature = "protonation")]
pub mod propka_wrap;

#[cfg(feature = "protonation")]
pub mod protonation_state;

#[cfg(any(feature = "protonation", feature = "capping", feature = "stereo", feature = "disulfide"))]
pub mod sanitizers;

pub trait Sanitizer {
    #[cfg(feature = "protonation")]
    fn protonate(&mut self) -> Result<(), String>;

    #[cfg(feature = "capping")]
    fn cap(&mut self) -> Result<(), String>;

    #[cfg(feature = "stereo")]
    fn fix_stereo(&mut self) -> Result<(), String>;

    // TODO(#977 C2): this returns () like the other sanitizer methods, so the
    // detected CYS-CYS pairs are not surfaced through the trait. Concrete impls
    // (use DisulfideSanitizer::run directly) get Vec<Disulfide>; add a trait-level
    // accessor for the pairs when a real Sanitizer implementor is built.
    #[cfg(feature = "disulfide")]
    fn detect_disulfides(&mut self) -> Result<(), String>;
}
