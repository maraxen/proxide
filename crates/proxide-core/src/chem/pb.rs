//! Generated protobuf types for the bundled CCD chem_comp_type schema
//! (`proto/ccd_chem_comp_types.proto`).
pub mod proxide {
    pub mod core {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/proxide.core.v1.rs"));
        }
    }
}

/// Re-export convenience alias for v1 types.
pub use proxide::core::v1 as core_v1;
