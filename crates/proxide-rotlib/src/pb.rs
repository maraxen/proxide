//! Generated protobuf types for the rotamer library schema.
pub mod proxide {
    pub mod rotlib {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/proxide.rotlib.v1.rs"));
        }
    }
}

/// Re-export convenience alias for v1 types.
pub use proxide::rotlib::v1 as rotlib_v1;
