fn main() {
    prost_build::compile_protos(
        &["proto/rotlib.proto"],
        &["proto/"],
    )
    .expect("failed to compile rotlib.proto");
}
