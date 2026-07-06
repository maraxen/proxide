fn main() {
    // Use the vendored protoc binary rather than requiring one on PATH --
    // CI wheel-build runners (Windows/macOS/manylinux Docker) have no protoc
    // preinstalled, and this crate is a dependency of every published wheel.
    if std::env::var_os("PROTOC").is_none() {
        std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path().unwrap());
    }
    prost_build::compile_protos(&["proto/ccd_chem_comp_types.proto"], &["proto/"])
        .expect("failed to compile protobuf files");
}
