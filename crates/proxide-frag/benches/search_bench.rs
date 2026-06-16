//! Criterion benchmark: search() vs search_prefiltered() at various DB sizes and epsilon values.
//!
//! Builds random 5-mer databases (10K, 100K, 1M fragments) and measures RMSD search
//! performance with and without the norm-bound pre-filter, at small (ε=1.0) and
//! generous (ε=5.0) cutoff thresholds.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use proxide_frag::{Fragment, FragmentDb, FragmentDbBuilder, Raw, SourceLabel};
use rand::{Rng, SeedableRng};

/// Build a random FragmentDb<5> with N fragments using a fixed RNG seed.
fn build_random_db(n: usize, seed: u64) -> FragmentDb<5> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut builder = FragmentDbBuilder::<5>::new();

    for i in 0..n {
        let mut coords: [[[f32; 3]; 4]; 5] = [[[0.0; 3]; 4]; 5];
        for r in 0..5 {
            for at in 0..4 {
                coords[r][at][0] = rng.gen_range(-20.0_f32..20.0);
                coords[r][at][1] = rng.gen_range(-20.0_f32..20.0);
                coords[r][at][2] = rng.gen_range(-20.0_f32..20.0);
            }
        }
        let label = SourceLabel::new(&format!("DB{:07}", i), 'A', 1, ' ', 5, ' ');
        let frag = Fragment::<5, Raw>::new(coords);
        let _ = builder.add_fragment(frag, label);
    }

    builder.build()
}

/// Generate a centered query fragment.
fn random_query(seed: u64) -> Fragment<5, proxide_frag::Centered> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut coords: [[[f32; 3]; 4]; 5] = [[[0.0; 3]; 4]; 5];
    for r in 0..5 {
        for at in 0..4 {
            coords[r][at][0] = rng.gen_range(-20.0_f32..20.0);
            coords[r][at][1] = rng.gen_range(-20.0_f32..20.0);
            coords[r][at][2] = rng.gen_range(-20.0_f32..20.0);
        }
    }
    let query_raw = Fragment::<5, Raw>::new(coords);
    let (query, _) = query_raw.center().expect("Failed to center query");
    query
}

fn search_benchmarks(c: &mut Criterion) {
    let sizes = vec![10_000, 100_000, 1_000_000];
    let epsilons = vec![1.0_f32, 5.0_f32];

    for size in sizes {
        for epsilon in &epsilons {
            // Build database once (outside timed loop).
            let db = build_random_db(size, 42);
            let query = random_query(99);

            let bench_id = BenchmarkId::new("search_regular", format!("N{}_eps{}", size, epsilon));
            c.bench_with_input(bench_id, &size, |b, _| {
                b.iter(|| {
                    let _ = black_box(&db).search(black_box(&query), black_box(*epsilon));
                })
            });

            let bench_id_prefilt = BenchmarkId::new("search_prefiltered", format!("N{}_eps{}", size, epsilon));
            c.bench_with_input(bench_id_prefilt, &size, |b, _| {
                b.iter(|| {
                    let _ = black_box(&db).search_prefiltered(black_box(&query), black_box(*epsilon));
                })
            });
        }
    }
}

criterion_group!(benches, search_benchmarks);
criterion_main!(benches);
