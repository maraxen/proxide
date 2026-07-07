use criterion::{black_box, criterion_group, criterion_main, Criterion};
use proxide_jaccard::jaccard_distance;

fn bench_jaccard_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("jaccard_distance");
    for &n in &[200usize, 2_000, 20_000] {
        let a: Vec<i64> = (0..n as i64).map(|x| x * 3).collect();
        let b: Vec<i64> = (0..n as i64).map(|x| x * 5).collect();
        group.bench_function(format!("sketch_len_{n}"), |bencher| {
            bencher.iter(|| jaccard_distance(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_jaccard_kernel);
criterion_main!(benches);
