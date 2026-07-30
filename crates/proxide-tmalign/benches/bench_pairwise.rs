//! Criterion benchmark: `pairwise_tm_scores` (row-parallel) vs. a naive
//! fully-serial nested-loop reference, at increasing batch sizes.
//!
//! Structures are synthetic non-collinear coordinate sets (helix-shaped,
//! varying length/phase per index) rather than real PDBs — this benchmark
//! exercises the batch-parallelism dispatch overhead and scaling, not
//! TM-align's numerical accuracy (already covered by the parity tests), so
//! it doesn't depend on `~/repos/USalign` or any committed fixture PDBs.

use criterion::{black_box, BenchmarkId, Criterion};
use nalgebra::Vector3;
use proxide_core::processing::residues::ResidueId;
use proxide_tmalign::parallel::pairwise_tm_scores;
use proxide_tmalign::pipeline::tmalign_pair_serial;
use proxide_tmalign::CaTrace;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Synthetic non-collinear structure of `n` residues, phase-shifted by
/// `seed` so distinct batch entries aren't literally identical.
fn synthetic_trace(n: usize, seed: u64) -> CaTrace {
    let mut rng = StdRng::seed_from_u64(seed);
    let phase: f32 = rng.gen_range(0.0..360.0);
    let coords: Vec<Vector3<f32>> = (0..n)
        .map(|i| {
            let angle = (i as f32 * 100.0 + phase).to_radians();
            let z = i as f32 * 1.5;
            Vector3::new(2.3 * angle.cos(), 2.3 * angle.sin(), z)
        })
        .collect();
    CaTrace {
        seq: vec![b'A'; n],
        res_ids: (0..n as i32)
            .map(|i| ResidueId { chain_id: "A".to_string(), res_id: i + 1, insertion_code: ' ' })
            .collect(),
        coords,
    }
}

fn build_batch(n_structures: usize, residues_per_structure: usize) -> Vec<CaTrace> {
    (0..n_structures)
        .map(|i| synthetic_trace(residues_per_structure, i as u64))
        .collect()
}

/// Naive fully-serial reference: no `orx-parallel` dispatch at all, just a
/// direct nested loop calling `tmalign_pair_serial` — the baseline
/// `pairwise_tm_scores`'s row-parallelism is measured against.
fn pairwise_tm_scores_naive_serial(traces: &[CaTrace]) -> ndarray::Array2<f32> {
    let n = traces.len();
    let mut mat = ndarray::Array2::<f32>::zeros((n, n));
    for i in 0..n {
        mat[[i, i]] = 1.0;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let r = tmalign_pair_serial(&traces[i].coords, &traces[j].coords)
                .expect("synthetic traces are never empty");
            mat[[i, j]] = r.tm_score_norm1;
            mat[[j, i]] = r.tm_score_norm2;
        }
    }
    mat
}

fn pairwise_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("pairwise_tm_scores");
    // Small residue count per structure keeps total wall-clock reasonable
    // while still exercising the full seed/DP_iter/refinement pipeline per
    // pair; batch sizes span below and above the ≥10-structure boundary
    // called out in the phase spec's verification section.
    for &n_structures in &[5usize, 10, 20] {
        let batch = build_batch(n_structures, 20);

        group.bench_with_input(
            BenchmarkId::new("row_parallel", n_structures),
            &n_structures,
            |b, _| b.iter(|| pairwise_tm_scores(black_box(&batch)).expect("non-empty batch")),
        );

        group.bench_with_input(
            BenchmarkId::new("naive_serial", n_structures),
            &n_structures,
            |b, _| b.iter(|| black_box(pairwise_tm_scores_naive_serial(black_box(&batch)))),
        );
    }
    group.finish();
}

criterion::criterion_group!(benches, pairwise_benchmarks);
criterion::criterion_main!(benches);
