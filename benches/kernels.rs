use branded_index::kernels::{add, gather, rng, spmv, stencil, union_find};
use branded_index::{scope, with_len};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

/// Two sizes per kernel: one that fits in L1/L2 (bounds checks compete with compute) and one
/// that spills to DRAM (memory bandwidth tends to hide them).
const SIZES: [(&str, usize); 2] = [("small", 1 << 10), ("large", 1 << 20)];

fn bench_add(c: &mut Criterion) {
    for (label, n) in SIZES {
        let mut r = rng(1);
        let a: Vec<f32> = (0..n).map(|_| (r() % 1000) as f32).collect();
        let b: Vec<f32> = (0..n).map(|_| (r() % 1000) as f32).collect();
        let mut out = vec![0.0f32; n];

        let mut g = c.benchmark_group(format!("add/{label}"));
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function("checked", |bn| bn.iter(|| add::checked(black_box(&a), black_box(&b), &mut out)));
        g.bench_function("hoisted", |bn| bn.iter(|| add::hoisted(black_box(&a), black_box(&b), &mut out)));
        g.bench_function("idiomatic", |bn| bn.iter(|| add::idiomatic(black_box(&a), black_box(&b), &mut out)));
        g.bench_function("unchecked", |bn| bn.iter(|| add::unchecked(black_box(&a), black_box(&b), &mut out)));
        g.bench_function("branded", |bn| bn.iter(|| add::branded(black_box(&a), black_box(&b), &mut out)));
        g.finish();
    }
}

fn bench_gather(c: &mut Criterion) {
    let count = 1 << 16;
    for (label, n) in SIZES {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut r = rng(2);
        let idx: Vec<u32> = (0..count).map(|_| (r() % n as u64) as u32).collect();
        let mut out = vec![0.0f64; count];

        let mut g = c.benchmark_group(format!("gather/{label}"));
        g.throughput(Throughput::Elements(count as u64));
        g.bench_function("checked", |bn| bn.iter(|| gather::checked(black_box(&data), black_box(&idx), &mut out)));
        g.bench_function("unchecked", |bn| {
            // SAFETY: every index was generated `< n == data.len()`.
            bn.iter(|| unsafe { gather::unchecked(black_box(&data), black_box(&idx), &mut out) })
        });
        scope(&data, |d| {
            let v = gather::validate(d.len(), &idx).unwrap();
            g.bench_function("branded", |bn| bn.iter(|| gather::branded(black_box(d), black_box(&v), &mut out)));
        });
        g.bench_function("validate (one-off)", |bn| {
            bn.iter(|| with_len(n, |len| gather::validate(len, black_box(&idx)).map(|v| v.len())))
        });
        g.finish();
    }
}

fn bench_spmv(c: &mut Criterion) {
    let per_row = 16;
    for (label, n) in [("small", 1 << 10), ("large", 1 << 16)] {
        let a = spmv::Csr::random(n, n, per_row, 3);
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let mut y = vec![0.0f64; n];

        let mut g = c.benchmark_group(format!("spmv/{label}"));
        g.throughput(Throughput::Elements(a.vals.len() as u64));
        g.bench_function("checked", |bn| bn.iter(|| spmv::checked(black_box(&a), black_box(&x), &mut y)));
        g.bench_function("idiomatic", |bn| bn.iter(|| spmv::idiomatic(black_box(&a), black_box(&x), &mut y)));
        g.bench_function("unchecked", |bn| {
            // SAFETY: `Csr::random` builds a well-formed matrix and `x.len() == ncols`.
            bn.iter(|| unsafe { spmv::unchecked(black_box(&a), black_box(&x), &mut y) })
        });
        with_len(a.vals.len(), |z| {
            with_len(a.ncols, |cl| {
                let m = spmv::validate(z, cl, &a).unwrap();
                let (cols, vals, xb) = (z.brand(&m.cols).unwrap(), z.brand(&a.vals).unwrap(), cl.brand(&x).unwrap());
                g.bench_function("branded", |bn| {
                    bn.iter(|| spmv::branded(black_box(m.rows), black_box(cols), black_box(vals), black_box(xb), &mut y))
                });
            })
        });
        g.bench_function("validate (one-off)", |bn| {
            bn.iter(|| {
                with_len(a.vals.len(), |z| with_len(a.ncols, |cl| spmv::validate(z, cl, black_box(&a)).is_some()))
            })
        });
        g.finish();
    }
}

fn bench_union_find(c: &mut Criterion) {
    for (label, n) in [("small", 1 << 12), ("large", 1 << 20)] {
        let ops = union_find::random_ops(n, n, 4);
        let init: Vec<u32> = (0..n as u32).collect();
        let mut p = init.clone();

        // Each iteration resets the parent array, then runs every union. The reset is the same
        // `memcpy` in all variants.
        let mut g = c.benchmark_group(format!("union_find/{label}"));
        g.throughput(Throughput::Elements(ops.len() as u64));
        g.bench_function("checked", |bn| {
            bn.iter(|| {
                p.copy_from_slice(&init);
                union_find::checked(&mut p, black_box(&ops))
            })
        });
        g.bench_function("unchecked", |bn| {
            bn.iter(|| {
                p.copy_from_slice(&init);
                // SAFETY: identity parents and ops generated `< n`.
                unsafe { union_find::unchecked(&mut p, black_box(&ops)) }
            })
        });
        with_len(n, |len| {
            let binit = union_find::identity(len).unwrap();
            let bops = union_find::validate_ops(len, &ops).unwrap();
            let mut storage = binit.clone();
            g.bench_function("branded", |bn| {
                bn.iter(|| {
                    storage.copy_from_slice(&binit);
                    union_find::branded(len.brand_mut(&mut storage).unwrap(), black_box(&bops))
                })
            });
        });
        g.bench_function("validate (one-off)", |bn| {
            bn.iter(|| with_len(n, |len| union_find::validate_ops(len, black_box(&ops)).map(|v| v.len())))
        });
        g.finish();
    }
}

fn bench_stencil(c: &mut Criterion) {
    for (label, side) in [("small", 64), ("large", 2048)] {
        let n = side * side;
        let mut r = rng(5);
        let a: Vec<f64> = (0..n).map(|_| (r() % 1000) as f64).collect();
        let mut out = vec![0.0f64; n];

        let mut g = c.benchmark_group(format!("stencil/{label}"));
        g.throughput(Throughput::Elements(((side - 2) * (side - 2)) as u64));
        g.bench_function("checked", |bn| bn.iter(|| stencil::checked(black_box(&a), &mut out, black_box(side))));
        g.bench_function("idiomatic", |bn| bn.iter(|| stencil::idiomatic(black_box(&a), &mut out, black_box(side))));
        g.bench_function("unchecked", |bn| {
            // SAFETY: equal lengths, `n` a multiple of `side`.
            bn.iter(|| unsafe { stencil::unchecked(black_box(&a), &mut out, black_box(side)) })
        });
        with_len(n, |len| {
            let grid = len.grid(side).unwrap();
            let ab = len.brand(&a).unwrap();
            let mut ob = len.brand_mut(&mut out).unwrap();
            g.bench_function("branded", |bn| {
                bn.iter(|| stencil::branded(black_box(grid), black_box(ab), ob.reborrow()))
            });
        });
        g.finish();
    }
}

fn bench_spmv_reassoc(c: &mut Criterion) {
    // (label, rows = cols, nonzeros per row)
    for (label, n, per_row) in [("small-16", 1 << 10, 16), ("large-16", 1 << 16, 16), ("wide-256", 1 << 12, 256)] {
        let a = spmv::Csr::random(n, n, per_row, 3);
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let mut y = vec![0.0f64; n];

        let mut g = c.benchmark_group(format!("spmv_reassoc/{label}"));
        g.throughput(Throughput::Elements(a.vals.len() as u64));
        g.bench_function("unchecked", |bn| {
            // SAFETY: `Csr::random` builds a well-formed matrix and `x.len() == ncols`.
            bn.iter(|| unsafe { spmv::unchecked(black_box(&a), black_box(&x), &mut y) })
        });
        with_len(a.vals.len(), |z| {
            with_len(a.ncols, |cl| {
                let m = spmv::validate(z, cl, &a).unwrap();
                let (cols, vals, xb) = (z.brand(&m.cols).unwrap(), z.brand(&a.vals).unwrap(), cl.brand(&x).unwrap());
                g.bench_function("branded", |bn| {
                    bn.iter(|| spmv::branded(black_box(m.rows), black_box(cols), black_box(vals), black_box(xb), &mut y))
                });
                g.bench_function("reassoc K=2", |bn| {
                    bn.iter(|| spmv::branded_reassoc::<2>(black_box(m.rows), black_box(cols), black_box(vals), black_box(xb), &mut y))
                });
                g.bench_function("reassoc K=4", |bn| {
                    bn.iter(|| spmv::branded_reassoc::<4>(black_box(m.rows), black_box(cols), black_box(vals), black_box(xb), &mut y))
                });
                g.bench_function("reassoc K=8", |bn| {
                    bn.iter(|| spmv::branded_reassoc::<8>(black_box(m.rows), black_box(cols), black_box(vals), black_box(xb), &mut y))
                });
            })
        });
        g.finish();
    }
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(40)
}

criterion_group! {
    name = benches;
    config = config();
    targets = bench_add, bench_gather, bench_spmv, bench_union_find, bench_stencil, bench_spmv_reassoc
}
criterion_main!(benches);
