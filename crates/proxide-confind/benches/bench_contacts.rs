use criterion::{criterion_group, criterion_main, Criterion};

/// Benchmark for ConFind contact discovery (Phases B+C).
///
/// This benchmark is marked as ignored for CI because it requires:
/// 1. A real rotamer library file
/// 2. A real protein structure (PDB file)
/// 3. Substantial time to run (full contact discovery can be slow)
///
/// To run locally:
///   cargo bench --bench bench_contacts -- --nocapture
///
/// The benchmark would measure:
/// - Time to compute all contacts for a structure
/// - Time to compute contact degrees
/// - Parallel efficiency of rayon work distribution
/// - Grid query performance at scale
/// - Memory usage during parallel phases
///
/// For now, we provide a simple no-op benchmark that ensures compilation.

fn bench_contacts_noop(c: &mut Criterion) {
    c.bench_function("contacts_noop", |b| b.iter(|| {
        let _ = std::hint::black_box(42u64);
    }));
}

criterion_group!(benches, bench_contacts_noop);
criterion_main!(benches);
