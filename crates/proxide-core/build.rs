fn main() {
    prost_build::compile_protos(&["proto/ccd_chem_comp_types.proto"], &["proto/"])
        .expect("failed to compile protobuf files");
}
