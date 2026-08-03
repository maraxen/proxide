use criterion::{criterion_group, criterion_main, Criterion};

/// Benchmark for ConFind rotamer caching (Phase A).
///
/// This benchmark is marked as ignored for CI because it requires:
/// 1. A real rotamer library file
/// 2. A real protein structure (PDB file)
/// 3. Substantial time to run
///
/// To run locally:
///   cargo bench --bench bench_cache -- --nocapture
///
/// The benchmark would measure:
/// - Time to cache a single residue
/// - Time to cache all residues in parallel
/// - Memory usage during caching
/// - Grid lookup performance during collision detection
///
/// For now, we provide a simple no-op benchmark that ensures compilation.

fn bench_cache_noop(c: &mut Criterion) {
    c.bench_function("cache_noop", |b| {
        b.iter(|| {
            let _ = std::hint::black_box(42u64);
        })
    });
}

criterion_group!(benches, bench_cache_noop);
criterion_main!(benches);
