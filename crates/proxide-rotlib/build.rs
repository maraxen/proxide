fn main() {
    prost_build::compile_protos(
        &["proto/rotlib.proto", "proto/residue_geometry.proto"],
        &["proto/"],
    )
    .expect("failed to compile protobuf files");
}
