//! Coordinate formatters for protein structures
//!
//! This module provides formatters to convert ProcessedStructure into various
//! standard coordinate representations (Atom37, Atom14, Full, Custom, etc.)

pub mod atom14;
pub mod atom37;
pub mod backbone;
pub mod cache;
pub mod custom;
pub mod full;

pub use atom14::Atom14Formatter;
pub use atom37::Atom37Formatter;
pub use backbone::BackboneFormatter;
pub use cache::{get_cached, insert_cached, CacheKey, CachedStructure};
pub use full::FullFormatter;
// Note: CustomFormatter and cache_size/clear_cache are available but not re-exported
// Use formatters::custom::CustomFormatter or formatters::cache::* directly if needed
