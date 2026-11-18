//! Benchmark kernels, each written several ways:
//!
//! - `checked`: the natural safe code, with a bounds check on every access.
//! - `hoisted` / `idiomatic`: the best safe code available without this crate.
//! - `unchecked`: `get_unchecked` everywhere. Needs `unsafe` and trusts the input.
//! - `branded`: this crate. Safe; checks paid once at validation (`validate_*`).
//!
//! Functions are `#[inline(never)]` so each variant can be read in the disassembly.
//! Benchmark-only crate; not published.

#![allow(missing_docs)]

use branded_index::{
    Grid, Index, Len, Offsets, Slice, SliceMut, ValidIndices, ValidOffsets, with_len,
};

/// splitmix64: small deterministic RNG so inputs need no extra dependency.
pub fn rng(mut state: u64) -> impl FnMut() -> u64 {
    move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// `c[i] = a[i] + b[i]` over an index loop. Three slices of unrelated length, as a caller
/// would hand them in.
pub mod add {
    #[inline(never)]
    pub fn checked(a: &[f32], b: &[f32], c: &mut [f32]) {
        for i in 0..c.len() {
            c[i] = a[i] + b[i];
        }
    }

    /// The "assert first" trick: tell LLVM the lengths up front so it can drop the checks.
    #[inline(never)]
    pub fn hoisted(a: &[f32], b: &[f32], c: &mut [f32]) {
        let n = c.len();
        let (a, b) = (&a[..n], &b[..n]);
        for i in 0..n {
            c[i] = a[i] + b[i];
        }
    }

    #[inline(never)]
    pub fn idiomatic(a: &[f32], b: &[f32], c: &mut [f32]) {
        for ((c, a), b) in c.iter_mut().zip(a).zip(b) {
            *c = a + b;
        }
    }

    #[inline(never)]
    pub fn unchecked(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() >= c.len() && b.len() >= c.len());
        for i in 0..c.len() {
            // SAFETY: `i < c.len() <= a.len(), b.len()` by the assert above.
            unsafe { *c.get_unchecked_mut(i) = a.get_unchecked(i) + b.get_unchecked(i) };
        }
    }

    #[inline(never)]
    pub fn branded(a: &[f32], b: &[f32], c: &mut [f32]) {
        branded_index::with_len(c.len(), |n| {
            let a = n.brand(&a[..n.get()]).unwrap();
            let b = n.brand(&b[..n.get()]).unwrap();
            let mut c = n.brand_mut(c).unwrap();
            for i in n.indices() {
                c[i] = a[i] + b[i];
            }
        })
    }
}

/// `out[k] = data[idx[k]]` with arbitrary indices: permutations, embedding lookups,
/// column reorders. The same index list is typically applied many times.
pub mod gather {
    use super::*;

    #[inline(never)]
    pub fn checked(data: &[f64], idx: &[u32], out: &mut [f64]) {
        for (o, &i) in out.iter_mut().zip(idx) {
            *o = data[i as usize];
        }
    }

    /// # Safety
    /// Every element of `idx` must be `< data.len()`.
    #[inline(never)]
    pub unsafe fn unchecked(data: &[f64], idx: &[u32], out: &mut [f64]) {
        for (o, &i) in out.iter_mut().zip(idx) {
            // SAFETY: caller guarantees `i < data.len()`.
            *o = unsafe { *data.get_unchecked(i as usize) };
        }
    }

    #[inline(never)]
    pub fn branded<'id>(data: Slice<'id, '_, f64>, idx: &[Index<'id, u32>], out: &mut [f64]) {
        for (o, &i) in out.iter_mut().zip(idx) {
            *o = data[i];
        }
    }
}

/// Sparse matrix times dense vector, CSR format. The matrix is validated once and multiplied
/// many times (iterative solvers, PageRank).
pub mod spmv {
    use super::*;

    pub struct Csr {
        pub nrows: usize,
        pub ncols: usize,
        /// `nrows + 1` offsets into `cols` / `vals`.
        pub row_ptr: Vec<usize>,
        pub cols: Vec<u32>,
        pub vals: Vec<f64>,
    }

    impl Csr {
        pub fn random(nrows: usize, ncols: usize, per_row: usize, seed: u64) -> Csr {
            let mut r = rng(seed);
            let mut row_ptr = Vec::with_capacity(nrows + 1);
            let mut cols = Vec::with_capacity(nrows * per_row);
            let mut vals = Vec::with_capacity(nrows * per_row);
            row_ptr.push(0);
            for _ in 0..nrows {
                for _ in 0..per_row {
                    cols.push((r() % ncols as u64) as u32);
                    vals.push((r() >> 11) as f64 / (1u64 << 53) as f64);
                }
                row_ptr.push(cols.len());
            }
            Csr {
                nrows,
                ncols,
                row_ptr,
                cols,
                vals,
            }
        }
    }

    #[inline(never)]
    pub fn checked(a: &Csr, x: &[f64], y: &mut [f64]) {
        for r in 0..a.nrows {
            let mut s = 0.0;
            for k in a.row_ptr[r]..a.row_ptr[r + 1] {
                s += a.vals[k] * x[a.cols[k] as usize];
            }
            y[r] = s;
        }
    }

    /// Best safe code without branding: per-row sub-slices and `zip` remove every check except
    /// `x[col]`, which no local rewrite can remove.
    #[inline(never)]
    pub fn idiomatic(a: &Csr, x: &[f64], y: &mut [f64]) {
        for (y, w) in y.iter_mut().zip(a.row_ptr.windows(2)) {
            let (lo, hi) = (w[0], w[1]);
            *y = a.vals[lo..hi]
                .iter()
                .zip(&a.cols[lo..hi])
                .fold(0.0, |s, (v, &c)| s + v * x[c as usize]);
        }
    }

    /// # Safety
    /// `a` must be well-formed: `row_ptr` non-decreasing with last element `<= cols.len(),
    /// vals.len()`, every column `< x.len()`, and `y.len() >= nrows`.
    #[inline(never)]
    pub unsafe fn unchecked(a: &Csr, x: &[f64], y: &mut [f64]) {
        for r in 0..a.nrows {
            let mut s = 0.0;
            // SAFETY: well-formedness from the caller.
            unsafe {
                let (lo, hi) = (*a.row_ptr.get_unchecked(r), *a.row_ptr.get_unchecked(r + 1));
                for k in lo..hi {
                    s += a.vals.get_unchecked(k)
                        * x.get_unchecked(*a.cols.get_unchecked(k) as usize);
                }
                *y.get_unchecked_mut(r) = s;
            }
        }
    }

    /// A branded SpMV kernel: row offsets and columns into the nonzeros (`'z`), column indices
    /// into `x` (`'c`).
    pub type Kernel = for<'z, 'c, 'a> fn(
        Offsets<'z, 'a>,
        Slice<'z, 'a, Index<'c, u32>>,
        Slice<'z, 'a, f64>,
        Slice<'c, 'a, f64>,
        &'a mut [f64],
    );

    /// A CSR matrix validated once, at construction, that lives in ordinary code: no brand
    /// lifetimes in its type, and no `unsafe` anywhere in its implementation. Each
    /// multiplication re-brands it with O(1) checks.
    #[derive(Clone, Debug)]
    pub struct ValidCsr {
        nrows: usize,
        ncols: usize,
        row_ptr: ValidOffsets,
        cols: ValidIndices<u32>,
        vals: Vec<f64>,
    }

    impl ValidCsr {
        /// Checks every row boundary and every column index once.
        pub fn new(a: &Csr) -> Option<ValidCsr> {
            if a.row_ptr.len() != a.nrows + 1 || a.cols.len() != a.vals.len() {
                return None;
            }
            let nnz = a.vals.len();
            Some(ValidCsr {
                nrows: a.nrows,
                ncols: a.ncols,
                row_ptr: ValidOffsets::new(a.row_ptr.clone(), nnz).ok()?,
                cols: ValidIndices::new(a.cols.clone(), a.ncols).ok()?,
                vals: a.vals.clone(),
            })
        }

        pub fn nrows(&self) -> usize {
            self.nrows
        }

        pub fn ncols(&self) -> usize {
            self.ncols
        }

        /// `y = A x` using `kernel`. Panics unless `x.len() == ncols` and `y.len() == nrows`.
        #[inline]
        pub fn mul_with(&self, kernel: Kernel, x: &[f64], y: &mut [f64]) {
            assert_eq!(x.len(), self.ncols, "x has the wrong length");
            assert_eq!(y.len(), self.nrows, "y has the wrong length");
            with_len(self.vals.len(), |z| {
                with_len(self.ncols, |c| {
                    // All O(1): bound comparisons and length checks, never a pass over the data.
                    let rows = self.row_ptr.view(z).expect("row_ptr bound is nnz");
                    let cols = self.cols.view(c).expect("column bound is ncols");
                    let cols = z.brand(cols).expect("cols and vals have equal length");
                    let vals = z.brand(&self.vals).expect("vals has length nnz");
                    kernel(rows, cols, vals, c.brand(x).expect("checked above"), y)
                })
            })
        }
    }

    #[inline(never)]
    pub fn branded<'z, 'c>(
        rows: Offsets<'z, '_>,
        cols: Slice<'z, '_, Index<'c, u32>>,
        vals: Slice<'z, '_, f64>,
        x: Slice<'c, '_, f64>,
        y: &mut [f64],
    ) {
        for (y, r) in y.iter_mut().zip(rows.ranges()) {
            let mut s = 0.0;
            for k in r {
                s += vals[k] * x[cols[k]];
            }
            *y = s;
        }
    }

    /// Like [`branded`], but each row's dot product uses `K` independent accumulators, so the
    /// additions form `K` parallel chains instead of one serial chain. This reassociates the
    /// floating-point sum: results differ from the other variants in the last bits.
    #[inline(never)]
    pub fn branded_reassoc<'z, 'c, const K: usize>(
        rows: Offsets<'z, '_>,
        cols: Slice<'z, '_, Index<'c, u32>>,
        vals: Slice<'z, '_, f64>,
        x: Slice<'c, '_, f64>,
        y: &mut [f64],
    ) {
        for (y, r) in y.iter_mut().zip(rows.ranges()) {
            let (v, c) = (&vals[r.clone()], &cols[r]);
            let (v_chunks, v_rest) = v.as_chunks::<K>();
            let (c_chunks, c_rest) = c.as_chunks::<K>();
            let mut acc = [0.0f64; K];
            for (vk, ck) in v_chunks.iter().zip(c_chunks) {
                for j in 0..K {
                    acc[j] += vk[j] * x[ck[j]];
                }
            }
            let mut s = acc.iter().sum::<f64>();
            for (v, &c) in v_rest.iter().zip(c_rest) {
                s += v * x[c];
            }
            *y = s;
        }
    }

    /// Rows at least this long use [`branded_adaptive`]'s 4-way reassociated path.
    pub const ADAPTIVE_MIN_ROW: usize = 32;

    /// Picks per row: 4 accumulators for long rows, the serial loop for short ones. Short rows
    /// gain nothing from extra chains (the CPU already overlaps rows), and wider `K` regresses
    /// at `target-cpu=native` through AVX-512 gathers. Reassociates long rows only.
    #[inline(never)]
    pub fn branded_adaptive<'z, 'c>(
        rows: Offsets<'z, '_>,
        cols: Slice<'z, '_, Index<'c, u32>>,
        vals: Slice<'z, '_, f64>,
        x: Slice<'c, '_, f64>,
        y: &mut [f64],
    ) {
        for (y, r) in y.iter_mut().zip(rows.ranges()) {
            *y = if r.len() >= ADAPTIVE_MIN_ROW {
                dot_reassoc::<4, _>(&vals[r.clone()], &cols[r], x)
            } else {
                // Same loop shape as `branded`: a branded index loop, not `zip`.
                let mut s = 0.0;
                for k in r {
                    s += vals[k] * x[cols[k]];
                }
                s
            };
        }
    }

    #[inline(always)]
    fn dot_reassoc<'c, const K: usize, X>(v: &[f64], c: &[Index<'c, u32>], x: X) -> f64
    where
        X: core::ops::Index<Index<'c, u32>, Output = f64>,
    {
        let (v_chunks, v_rest) = v.as_chunks::<K>();
        let (c_chunks, c_rest) = c.as_chunks::<K>();
        let mut acc = [0.0f64; K];
        for (vk, ck) in v_chunks.iter().zip(c_chunks) {
            for j in 0..K {
                acc[j] += vk[j] * x[ck[j]];
            }
        }
        let mut s = acc.iter().sum::<f64>();
        for (v, &c) in v_rest.iter().zip(c_rest) {
            s += v * x[c];
        }
        s
    }
}

/// Union-find with path halving. The parent array stores indices into itself, so the data
/// structure's invariant ("every parent is a valid node") can be put in the type.
pub mod union_find {
    use super::*;

    pub fn random_ops(n: usize, count: usize, seed: u64) -> Vec<(u32, u32)> {
        let mut r = rng(seed);
        (0..count)
            .map(|_| ((r() % n as u64) as u32, (r() % n as u64) as u32))
            .collect()
    }

    #[inline(always)]
    fn find_checked(p: &mut [u32], mut x: u32) -> u32 {
        loop {
            let px = p[x as usize];
            if px == x {
                return x;
            }
            let gp = p[px as usize];
            p[x as usize] = gp;
            x = gp;
        }
    }

    /// Returns the number of successful unions.
    #[inline(never)]
    pub fn checked(p: &mut [u32], ops: &[(u32, u32)]) -> usize {
        let mut unions = 0;
        for &(a, b) in ops {
            let (ra, rb) = (find_checked(p, a), find_checked(p, b));
            if ra != rb {
                p[ra as usize] = rb;
                unions += 1;
            }
        }
        unions
    }

    #[inline(always)]
    unsafe fn find_unchecked(p: &mut [u32], mut x: u32) -> u32 {
        // SAFETY (whole body): caller guarantees `x` and every parent are `< p.len()`.
        unsafe {
            loop {
                let px = *p.get_unchecked(x as usize);
                if px == x {
                    return x;
                }
                let gp = *p.get_unchecked(px as usize);
                *p.get_unchecked_mut(x as usize) = gp;
                x = gp;
            }
        }
    }

    /// # Safety
    /// Every element of `p` and every node in `ops` must be `< p.len()`.
    #[inline(never)]
    pub unsafe fn unchecked(p: &mut [u32], ops: &[(u32, u32)]) -> usize {
        let mut unions = 0;
        for &(a, b) in ops {
            // SAFETY: caller's precondition; roots are parents, so also in range.
            unsafe {
                let (ra, rb) = (find_unchecked(p, a), find_unchecked(p, b));
                if ra != rb {
                    *p.get_unchecked_mut(ra as usize) = rb;
                    unions += 1;
                }
            }
        }
        unions
    }

    pub type Node<'id> = Index<'id, u32>;

    /// The identity parent array: every node its own root. `None` if `n` exceeds `u32`.
    pub fn identity<'id>(n: Len<'id>) -> Option<Vec<Node<'id>>> {
        Some(n.indices_as::<u32>()?.collect())
    }

    #[inline(never)]
    pub fn validate_ops<'id>(
        n: Len<'id>,
        ops: &[(u32, u32)],
    ) -> Option<Vec<(Node<'id>, Node<'id>)>> {
        ops.iter()
            .map(|&(a, b)| Some((n.check_as(a)?, n.check_as(b)?)))
            .collect()
    }

    #[inline(always)]
    fn find_branded<'id>(p: &mut SliceMut<'id, '_, Node<'id>>, mut x: Node<'id>) -> Node<'id> {
        loop {
            let px = p[x];
            if px == x {
                return x;
            }
            let gp = p[px];
            p[x] = gp;
            x = gp;
        }
    }

    #[inline(never)]
    pub fn branded<'id>(
        mut p: SliceMut<'id, '_, Node<'id>>,
        ops: &[(Node<'id>, Node<'id>)],
    ) -> usize {
        let mut unions = 0;
        for &(a, b) in ops {
            let (ra, rb) = (find_branded(&mut p, a), find_branded(&mut p, b));
            if ra != rb {
                p[ra] = rb;
                unions += 1;
            }
        }
        unions
    }
}

/// 5-point Jacobi sweep on a row-major grid:
/// `out[r][c] = 0.25 * (a[r-1][c] + a[r+1][c] + a[r][c-1] + a[r][c+1])` for interior cells.
/// Boundary cells of `out` are left untouched.
pub mod stencil {
    use super::*;

    #[inline(never)]
    pub fn checked(a: &[f64], out: &mut [f64], w: usize) {
        let h = a.len() / w;
        for r in 1..h.saturating_sub(1) {
            for c in 1..w.saturating_sub(1) {
                let i = r * w + c;
                out[i] = 0.25 * (a[i - w] + a[i + w] + a[i - 1] + a[i + 1]);
            }
        }
    }

    /// Best safe code without branding: three row slices, `zip` and `windows(3)`.
    #[inline(never)]
    pub fn idiomatic(a: &[f64], out: &mut [f64], w: usize) {
        let h = a.len() / w;
        if w < 3 || h < 3 {
            return;
        }
        for r in 1..h - 1 {
            let up = &a[(r - 1) * w + 1..r * w - 1];
            let mid = &a[r * w..(r + 1) * w];
            let down = &a[(r + 1) * w + 1..(r + 2) * w - 1];
            let o = &mut out[r * w + 1..(r + 1) * w - 1];
            for (((o, u), d), m) in o.iter_mut().zip(up).zip(down).zip(mid.windows(3)) {
                *o = 0.25 * (u + d + m[0] + m[2]);
            }
        }
    }

    /// # Safety
    /// `a.len() == out.len()`, and both a multiple of `w`.
    #[inline(never)]
    pub unsafe fn unchecked(a: &[f64], out: &mut [f64], w: usize) {
        let h = a.len() / w;
        for r in 1..h.saturating_sub(1) {
            for c in 1..w.saturating_sub(1) {
                let i = r * w + c;
                // SAFETY: interior cell, so `i - w`, `i + w`, `i - 1`, `i + 1` are all `< h * w`.
                unsafe {
                    *out.get_unchecked_mut(i) = 0.25
                        * (a.get_unchecked(i - w)
                            + a.get_unchecked(i + w)
                            + a.get_unchecked(i - 1)
                            + a.get_unchecked(i + 1));
                }
            }
        }
    }

    #[inline(never)]
    pub fn branded<'id>(g: Grid<'id>, a: Slice<'id, '_, f64>, mut out: SliceMut<'id, '_, f64>) {
        for row in g.interior_rows::<1>() {
            for i in row {
                out[i.center()] = 0.25 * (a[i.up()] + a[i.down()] + a[i.left()] + a[i.right()]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use branded_index::with_len;

    fn floats(n: usize, seed: u64) -> Vec<f32> {
        let mut r = rng(seed);
        (0..n).map(|_| (r() % 1000) as f32 / 7.0).collect()
    }

    #[test]
    fn add_variants_agree() {
        let n = 1000;
        let (a, b) = (floats(n, 1), floats(n, 2));
        let mut want = vec![0.0; n];
        add::checked(&a, &b, &mut want);
        for f in [add::hoisted, add::idiomatic, add::unchecked, add::branded] {
            let mut got = vec![0.0; n];
            f(&a, &b, &mut got);
            assert_eq!(got, want);
        }
    }

    #[test]
    fn gather_variants_agree() {
        let data: Vec<f64> = (0..500).map(|i| i as f64 * 1.5).collect();
        let mut r = rng(3);
        let idx: Vec<u32> = (0..2000).map(|_| (r() % 500) as u32).collect();
        let mut want = vec![0.0; idx.len()];
        gather::checked(&data, &idx, &mut want);

        let mut got = vec![0.0; idx.len()];
        // SAFETY: every index was generated `< 500 == data.len()`.
        unsafe { gather::unchecked(&data, &idx, &mut got) };
        assert_eq!(got, want);

        let v = ValidIndices::new(idx.clone(), data.len()).unwrap();
        branded_index::scope(&data, |d| {
            let mut got = vec![0.0; idx.len()];
            gather::branded(d, v.view(d.len()).unwrap(), &mut got);
            assert_eq!(got, want);
        });
    }

    #[test]
    fn spmv_variants_agree() {
        let a = spmv::Csr::random(300, 200, 7, 4);
        let x: Vec<f64> = (0..a.ncols).map(|i| i as f64 / 3.0).collect();
        let mut want = vec![0.0; a.nrows];
        spmv::checked(&a, &x, &mut want);

        let mut got = vec![0.0; a.nrows];
        spmv::idiomatic(&a, &x, &mut got);
        assert_eq!(got, want);

        let mut got = vec![0.0; a.nrows];
        // SAFETY: `Csr::random` builds a well-formed matrix and `x.len() == ncols`.
        unsafe { spmv::unchecked(&a, &x, &mut got) };
        assert_eq!(got, want);

        let m = spmv::ValidCsr::new(&a).unwrap();
        let mut got = vec![0.0; a.nrows];
        m.mul_with(spmv::branded, &x, &mut got);
        assert_eq!(got, want);
    }

    #[test]
    fn spmv_reassoc_matches_within_rounding() {
        // Row lengths 0..=39 cover empty rows, remainders, and rows on both sides of the
        // adaptive threshold.
        let mut a = spmv::Csr::random(40, 50, 39, 8);
        let mut row_ptr = vec![0];
        for r in 0..a.nrows {
            row_ptr.push(row_ptr[r] + r);
        }
        let nnz = *row_ptr.last().unwrap();
        a.cols.truncate(nnz);
        a.vals.truncate(nnz);
        a.row_ptr = row_ptr;

        let x: Vec<f64> = (0..a.ncols).map(|i| 1.0 + i as f64 / 7.0).collect();
        let mut want = vec![0.0; a.nrows];
        spmv::checked(&a, &x, &mut want);

        let m = spmv::ValidCsr::new(&a).unwrap();
        let run = |kernel: spmv::Kernel| {
            let mut got = vec![0.0; a.nrows];
            m.mul_with(kernel, &x, &mut got);
            got
        };
        let adaptive = run(spmv::branded_adaptive);
        for (r, (g, w)) in adaptive.iter().zip(&want).enumerate() {
            if a.row_ptr[r + 1] - a.row_ptr[r] < spmv::ADAPTIVE_MIN_ROW {
                assert_eq!(g, w, "short rows keep the serial order");
            }
        }
        let variants = [
            run(spmv::branded_reassoc::<1>),
            run(spmv::branded_reassoc::<2>),
            run(spmv::branded_reassoc::<4>),
            run(spmv::branded_reassoc::<8>),
            adaptive,
        ];
        for got in variants {
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() <= 1e-12 * w.abs().max(1.0), "{g} vs {w}");
            }
        }
        assert_eq!(
            run(spmv::branded_reassoc::<1>),
            want,
            "K = 1 is the serial order"
        );
    }

    #[test]
    fn spmv_validation_rejects_bad_matrix() {
        let mut a = spmv::Csr::random(4, 4, 2, 5);
        a.cols[3] = 4;
        assert!(spmv::ValidCsr::new(&a).is_none());

        let mut a = spmv::Csr::random(4, 4, 2, 5);
        a.row_ptr[2] = 100;
        assert!(spmv::ValidCsr::new(&a).is_none());

        let mut a = spmv::Csr::random(4, 4, 2, 5);
        a.row_ptr[2] = 1; // decreasing
        assert!(spmv::ValidCsr::new(&a).is_none());
    }

    #[test]
    fn union_find_variants_agree() {
        let n = 1000;
        let ops = union_find::random_ops(n, 800, 6);
        let init: Vec<u32> = (0..n as u32).collect();

        let mut p = init.clone();
        let want = union_find::checked(&mut p, &ops);

        let mut q = init.clone();
        // SAFETY: identity parents and ops generated `< n`.
        assert_eq!(unsafe { union_find::unchecked(&mut q, &ops) }, want);
        assert_eq!(q, p);

        with_len(n, |len| {
            let mut storage = union_find::identity(len).unwrap();
            let bops = union_find::validate_ops(len, &ops).unwrap();
            let got = union_find::branded(len.brand_mut(&mut storage).unwrap(), &bops);
            assert_eq!(got, want);
            let raw: Vec<u32> = storage.iter().map(|i| i.raw()).collect();
            assert_eq!(raw, p);
        });
    }

    #[test]
    fn stencil_variants_agree() {
        for (h, w) in [(1, 1), (2, 5), (3, 3), (17, 23), (64, 64)] {
            let n = h * w;
            let mut r = rng(7);
            let a: Vec<f64> = (0..n).map(|_| (r() % 1000) as f64).collect();
            let mut want = vec![-1.0; n];
            stencil::checked(&a, &mut want, w);

            let mut got = vec![-1.0; n];
            stencil::idiomatic(&a, &mut got, w);
            assert_eq!(got, want, "idiomatic {h}x{w}");

            let mut got = vec![-1.0; n];
            // SAFETY: equal lengths, multiple of `w`.
            unsafe { stencil::unchecked(&a, &mut got, w) };
            assert_eq!(got, want, "unchecked {h}x{w}");

            let mut got = vec![-1.0; n];
            with_len(n, |len| {
                let g = len.grid(w).unwrap();
                stencil::branded(g, len.brand(&a).unwrap(), len.brand_mut(&mut got).unwrap());
            });
            assert_eq!(got, want, "branded {h}x{w}");
        }
    }
}
