//! Placeholder bench — replaced by the profiling fixer (task 260602_proxcon_deferred4, item 4).
use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder(c: &mut Criterion) {
    c.bench_function("placeholder", |b| b.iter(|| 1 + 1));
}

criterion_group!(benches, placeholder);
criterion_main!(benches);
