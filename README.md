# branded-index

Branded indices for Rust. A bounds check is paid once, when an index is validated, instead of on
every access. User code stays 100% safe. The handful of `get_unchecked` calls live inside the
crate, each with a `SAFETY` argument.

```rust
branded_index::scope(&data, |d| {
    // one check per index, once
    let idx: Vec<_> = raw.iter().map(|&i| d.len().check_as(i).unwrap()).collect();
    for _ in 0..iterations {
        let sum: f64 = idx.iter().map(|&i| d[i]).sum(); // zero bounds checks
    }
});
```

## How it works

`with_len(n, |len| ...)` passes the closure a `Len<'id>`, where `'id` is a fresh, invariant
lifetime. Because the closure must work for *every* `'id`, the brand cannot be named, unified
with another brand, or smuggled out of the closure.

| Type | Invariant | Created by |
|---|---|---|
| `Len<'id>` | the length `n` | `with_len`, `scope`, `scope_mut` |
| `Index<'id, I = usize>` | `< n`; `I` can be `u8`..`u64` (`Index<'id, u32>` is 4 bytes) | `Len::check`, `check_as`, `indices`, `indices_as` |
| `Range<'id>` | `start <= end <= n` | `Len::range`, `Offsets::ranges` |
| `Offsets<'id, '_>` | non-decreasing, all `<= n` (CSR `row_ptr`, ragged arrays) | `Len::offsets` |
| `Slice<'id, '_, T>` / `SliceMut` | length exactly `n` | `Len::brand`, `brand_mut` |
| `Grid<'id>` | row-major, `width * height == n` | `Len::grid` |
| `Interior<'id, R>` | at least `R` cells from every edge; `at::<DR, DC>()` reaches any neighbour with `\|DR\|, \|DC\| <= R`, bound checked at compile time | `Grid::interior_rows::<R>()` |

Indexing a `Slice<'id>` with an `Index<'id>` or `Range<'id>` compiles to a plain load. Mixing
brands, or letting an index escape its scope, is a compile error. The `compile_fail` doctests in
`src/lib.rs` check both.

## Verification

- `cargo test`: unit tests, plus kernel tests asserting all variants produce identical output.
- `cargo +nightly miri test --lib`: clean.
- Disassembly: the `branded` kernels contain no panic calls. The `checked` ones contain 1 to 6.

## Benchmarks

`cargo bench`. AMD Threadripper PRO 9975WX, rustc 1.100, `codegen-units = 1`. Median times.
"Small" fits in L1/L2; "large" spills to DRAM.

| Kernel | checked | best safe w/o crate | `unchecked` (unsafe) | **branded** | validate (once) |
|---|---|---|---|---|---|
| add, small (1K) | 63.3 ns | 61.7 ns (zip) | 61.9 ns | 62.1 ns | n/a |
| add, large (1M) | 102 µs | 102 µs | 102 µs | 102 µs | n/a |
| gather, small (64K from 1K) | 12.6 µs | n/a | 11.6 µs | 11.5 µs | 19.6 µs |
| gather, large (64K from 1M) | 28.7 µs | n/a | 29.9 µs | 29.5 µs | 19.8 µs |
| SpMV, small (1K², 16/row) | 4.6–6.3 µs | 4.1–4.7 µs | 3.48 µs | **3.54 µs** | 6.9 µs |
| SpMV, large (64K², 16/row) | 404–426 µs | 341–374 µs | 325 µs | **328 µs** | 411 µs |
| union-find, small (4K) | 14.8 µs | n/a | 14.0 µs | 14.2 µs | 2.4 µs |
| union-find, large (1M) | 16.6 ms | n/a | 16.9 ms | 16.1 ms | 0.49 ms |
| 5-point stencil, small (64²) | 1.19 µs | 0.97 µs | 1.10 µs | 1.10 µs | n/a |
| 5-point stencil, large (2048²) | 2.25 ms | 2.25 ms | 2.24 ms | 2.25 ms | n/a |
| stencil small, `target-cpu=native` | 1.00 µs | 0.71 µs | 0.74 µs | 0.75 µs | n/a |
| stencil large, `target-cpu=native` | 1.69 ms | 1.59 ms | 1.87 ms | 1.85 ms | n/a |

A range means the result moved between runs with code layout, even though that code didn't
change. `unchecked` and `branded` stayed stable.

### Findings

1. **Branded matches `unchecked` within noise (about 2%) on every kernel.** It never loses to
   it.
2. **Check elimination only matters when compute-bound.** SpMV small is the clear case:
   checked is 30–80% slower than unchecked. In DRAM-bound cases (gather large, union-find
   large), memory latency hides the checks entirely.
3. **LLVM already handles the easy cases.** In `add` the checked index loop was versioned and
   vectorized, so it runs as fast as `unchecked`. Writing `unsafe` there buys nothing.
4. **Branded beats the best safe code available today.** In SpMV, idiomatic Rust (per-row
   sub-slices plus `zip`) still pays for `x[col]` on every element. No local rewrite removes it,
   because the proof that `col < x.len()` comes from validating the matrix, not from the loop.
   Branding carries that proof in the type.
5. **Validation has to be amortized.** A separate validation pass costs more than one checked
   run. Break-even is about 5 reuses for SpMV, about 4 for union-find, and about 19 for gather.
   For data used once, plain checked code is the right call. Branding pays off for structures
   validated once and used many times: iterative solvers, graphs, permutations, embedding
   tables.
6. **Layout matters as much as checks.** The first branded SpMV stored rows as `Vec<Range>`, 16
   bytes per row, and ran 15% behind `unchecked`. Switching to `Offsets` over the original
   `row_ptr` didn't help. What closed the gap was the `Ranges` iterator carrying the previous
   boundary in a register, so each row loads one offset instead of two, the same as the
   hand-written `unchecked` loop. The inner loops were instruction-identical throughout; the
   difference was per-row load traffic.

7. **For stencils, bounds checks are only part of the story.** With AVX-512, checked code is
   33% slower than branded in cache. But the safe idiomatic version (row slices plus `zip` plus
   `windows(3)`) is as fast as or faster than `unchecked`. Exact slice lengths give LLVM an
   exact trip count, so it unrolls the vector loop 2x. Branded indexing is a plain index loop
   like `unchecked`, and matches it. Branding's win here is that arbitrary neighbours
   (diagonals, radius-`R` stencils, irregular access inside the interior) can be written as
   indices, without hand-slicing three rows. It is not a speedup over the best safe slicing.
   The `large` native row is DRAM-bound and favours the checked/idiomatic loop shapes, probably
   through prefetch behaviour. It shows the same "layout over checks" theme.

### Reassociated SpMV (opt-in, `kernels::spmv::branded_reassoc::<K>`)

Each row's sum is split into `K` independent running sums, so the additions run as `K` parallel
chains instead of one serial chain. This is safe, stable Rust, and gets what `-ffast-math` would
give C. Results differ from the serial sum in the last bits (tested to within `1e-12` relative).

| Matrix (rows = cols, nonzeros/row) | build | unchecked | branded | K=2 | K=4 | K=8 |
|---|---|---|---|---|---|---|
| 1K, 16 | default | 3.49 µs | 3.55 µs | 3.47 µs | **3.03 µs** | 3.35 µs |
| 64K, 16 | default | 319 µs | 320 µs | 321 µs | 318 µs | 330 µs |
| 4K, 256 | default | 309 µs | 309 µs | 204 µs | 196 µs | **194 µs** |
| 1K, 16 | native | 3.36 µs | 3.36 µs | 3.48 µs | 3.52 µs | 5.46 µs |
| 64K, 16 | native | 326 µs | 324 µs | 345 µs | 348 µs | 485 µs |
| 4K, 256 | native | 311 µs | 311 µs | 190 µs | **172 µs** | 305 µs |

- **Long rows: 1.6–1.8x faster than `unchecked`.** The serial addition chain was the bottleneck,
  and safe reassociation beats the hand-written `unsafe` loop.
- **Short rows: little or no gain, sometimes a loss.** With 16 nonzeros per row the CPU already
  overlaps different rows, so independent chains add little. Memory-bound cases don't move.
- **Wider is not better.** At `target-cpu=native`, K=8 compiles to AVX-512 `vgatherdpd`, and K=4
  packs scalar loads into `ymm` registers. Both cost more than they save on short rows (Zen 5).
- **The fixed-`K` choice is a trap under `target-cpu=native`**, which is exactly the flag people
  reach for when they want speed.

### Adaptive SpMV (`kernels::spmv::branded_adaptive`), the robust choice

Per row: 4 running sums when the row has at least `ADAPTIVE_MIN_ROW` (32) nonzeros, otherwise
the serial branded index loop. Short rows keep the exact serial result.

| Matrix | build | unchecked | branded | K=4 | **adaptive** |
|---|---|---|---|---|---|
| 1K, 16 | default | 3.50 µs | 3.54 µs | 3.02 µs | 3.51 µs |
| 64K, 16 | default | 318 µs | 320 µs | 316 µs | 320 µs |
| 4K, 256 | default | 310 µs | 310 µs | 198 µs | **197 µs** |
| 1K, 16 | native | 3.35 µs | 3.35 µs | 3.51 µs | 3.36 µs |
| 64K, 16 | native | 321 µs | 317 µs | 332 µs | 318 µs |
| 4K, 256 | native | 311 µs | 312 µs | 175 µs | **175 µs** |

- Never slower than `unchecked` in either build, and 1.6–1.8x faster on long rows.
- It gives up K=4's 14% on short rows in the default build, which is the price of never
  regressing under `native`.
- The serial path must be the branded index loop. A first version used `zip` there and was 20%
  slower under `native` on short rows: the loop-shape effect again.

## Limits

- Brands are closure-scoped, so validated structures cannot be stored in a long-lived struct
  without threading `'id` through it.
- Arbitrary index arithmetic (`i + k`) still needs a fresh check. Only fixed grid offsets
  within the interior margin are free.
- A language feature (pattern types or refinement types) could make this ergonomic. This crate
  shows the codegen side is already solved.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.
