pub mod models;
pub mod templates;
pub mod finder;
pub mod builder;
pub mod protonate;

#[cfg(any(feature = "protonation", feature = "capping", feature = "stereo", feature = "disulfide"))]
pub mod sanitizers;

pub trait Sanitizer {
    #[cfg(feature = "protonation")]
    fn protonate(&mut self) -> Result<(), String>;

    #[cfg(feature = "capping")]
    fn cap(&mut self) -> Result<(), String>;

    #[cfg(feature = "stereo")]
    fn fix_stereo(&mut self) -> Result<(), String>;

    #[cfg(feature = "disulfide")]
    fn detect_disulfides(&mut self) -> Result<(), String>;
}
