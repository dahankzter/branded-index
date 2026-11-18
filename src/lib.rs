//! # branded-index
//!
//! Bounds checks paid once, at validation, instead of on every access.
//!
//! [`with_len`] creates a fresh, unnameable lifetime `'id` (a *brand*) tied to one length `n`.
//! Every [`Index<'id>`] is proven `< n` when it is created, and every [`Slice<'id, '_, T>`] is
//! proven to have length exactly `n`. Indexing a branded slice with a branded index therefore
//! needs no runtime check, and using an index of one brand on a slice of another is a compile
//! error. The only `unsafe` lives inside this crate; user code stays safe.
//!
//! ```
//! let data = [10, 20, 30, 40];
//! let total = branded_index::scope(&data, |s| s.len().indices().map(|i| s[i]).sum::<i32>());
//! assert_eq!(total, 100);
//! ```
//!
//! Indices validated once can be reused any number of times without further checks:
//!
//! ```
//! let data = [1.0, 2.0, 3.0];
//! let raw = [2u32, 0, 1, 2];
//! branded_index::scope(&data, |s| {
//!     let idx: Vec<_> = raw.iter().map(|&i| s.len().check_as(i).unwrap()).collect();
//!     for _ in 0..3 {
//!         let sum: f64 = idx.iter().map(|&i| s[i]).sum(); // no bounds checks
//!         assert_eq!(sum, 9.0);
//!     }
//! });
//! ```
//!
//! Several slices can share a brand when their lengths match, so one index works for all of them:
//!
//! ```
//! let (a, b) = ([1, 2, 3], [4, 5, 6]);
//! let mut c = [0; 3];
//! branded_index::with_len(3, |n| {
//!     let (a, b) = (n.brand(&a).unwrap(), n.brand(&b).unwrap());
//!     let mut c = n.brand_mut(&mut c).unwrap();
//!     for i in n.indices() {
//!         c[i] = a[i] + b[i];
//!     }
//! });
//! assert_eq!(c, [5, 7, 9]);
//! ```
//!
//! ## What the compiler rejects
//!
//! An index of one brand cannot index a slice of another brand, even if the lengths are equal:
//!
//! ```compile_fail,E0521
//! let (a, b) = ([1, 2, 3], [4, 5, 6]);
//! branded_index::scope(&a, |a| {
//!     branded_index::scope(&b, |b| {
//!         let i = a.len().check(2).unwrap();
//!         b[i]
//!     })
//! });
//! ```
//!
//! An index cannot escape the scope that created it:
//!
//! ```compile_fail
//! let i = branded_index::with_len(3, |n| n.check(0).unwrap());
//! ```
//!
//! Nor can it be smuggled out through captured storage:
//!
//! ```compile_fail,E0521
//! let mut out: Vec<branded_index::Index<'static>> = Vec::new();
//! branded_index::with_len(3, |n| out.push(n.check(0).unwrap()));
//! ```

#![no_std]
#![deny(unsafe_op_in_unsafe_fn, missing_docs)]

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;
use core::hash::Hash;
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::ops;

/// Invariant lifetime marker. Invariance stops the compiler from shrinking or growing `'id` to
/// unify two different brands, so each brand stays distinct.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Brand<'id>(PhantomData<fn(&'id ()) -> &'id ()>);

impl Brand<'_> {
    #[inline(always)]
    const fn new() -> Self {
        Brand(PhantomData)
    }
}

/// Opens a scope with a fresh brand `'id` for the length `n`.
///
/// The closure must work for *every* lifetime `'id`, so it cannot name the brand, return
/// anything carrying it, or store branded values anywhere outside the closure.
#[inline(always)]
pub fn with_len<R>(n: usize, f: impl for<'id> FnOnce(Len<'id>) -> R) -> R {
    f(Len {
        n,
        brand: Brand::new(),
    })
}

/// Brands `slice` and runs `f` with it. Shorthand for [`with_len`] plus [`Len::brand`].
#[inline(always)]
pub fn scope<'a, T, R>(slice: &'a [T], f: impl for<'id> FnOnce(Slice<'id, 'a, T>) -> R) -> R {
    f(Slice {
        s: slice,
        brand: Brand::new(),
    })
}

/// Mutable counterpart of [`scope`].
#[inline(always)]
pub fn scope_mut<'a, T, R>(
    slice: &'a mut [T],
    f: impl for<'id> FnOnce(SliceMut<'id, 'a, T>) -> R,
) -> R {
    f(SliceMut {
        s: slice,
        brand: Brand::new(),
    })
}

// ---------------------------------------------------------------------------------------------
// Len
// ---------------------------------------------------------------------------------------------

/// A length `n`, carrying the brand `'id`. Everything else in the crate is validated against it.
#[derive(Clone, Copy)]
pub struct Len<'id> {
    n: usize,
    brand: Brand<'id>,
}

impl<'id> Len<'id> {
    /// The length as a plain number.
    #[inline(always)]
    pub fn get(self) -> usize {
        self.n
    }

    /// `true` when the length is zero, in which case no [`Index<'id>`] can exist.
    #[inline(always)]
    pub fn is_empty(self) -> bool {
        self.n == 0
    }

    /// Validates `i < n`. This is the one bounds check an index ever pays.
    #[inline(always)]
    pub fn check(self, i: usize) -> Option<Index<'id>> {
        (i < self.n).then_some(Index {
            raw: i,
            brand: self.brand,
        })
    }

    /// Like [`check`](Self::check), but keeps a compact representation (for example `u32`), so
    /// arrays of indices use less memory.
    #[inline(always)]
    pub fn check_as<I: Repr>(self, i: I) -> Option<Index<'id, I>> {
        match i.try_to_usize() {
            Some(u) if u < self.n => Some(Index {
                raw: i,
                brand: self.brand,
            }),
            _ => None,
        }
    }

    /// Iterates over every index `0..n`. No per-element checks.
    #[inline(always)]
    pub fn indices(self) -> Range<'id> {
        Range {
            start: 0,
            end: self.n,
            brand: self.brand,
        }
    }

    /// Iterates over every index `0..n` in the compact representation `I`. Returns `None` when
    /// `n - 1` does not fit in `I`; that single check covers every index produced.
    #[inline]
    pub fn indices_as<I: Repr>(
        self,
    ) -> Option<impl DoubleEndedIterator<Item = Index<'id, I>> + ExactSizeIterator> {
        let fits = self.n == 0 || I::try_from_usize(self.n - 1).is_some();
        // `narrow` is lossless here: every value is `< n` and `n - 1` fits in `I`.
        fits.then(|| {
            self.indices().map(|i| Index {
                raw: I::narrow(i.raw),
                brand: i.brand,
            })
        })
    }

    /// Validates `start <= end <= n` and returns the branded range.
    #[inline(always)]
    pub fn range(self, start: usize, end: usize) -> Option<Range<'id>> {
        (start <= end && end <= self.n).then_some(Range {
            start,
            end,
            brand: self.brand,
        })
    }

    /// Validates fenceposts: `s` must be non-decreasing with every element `<= n`. Each adjacent
    /// pair is then a valid [`Range<'id>`] (CSR row pointers, ragged arrays, string tables),
    /// and the offsets stay in their compact form, one `usize` per boundary.
    #[inline]
    pub fn offsets<'a>(self, s: &'a [usize]) -> Option<Offsets<'id, 'a>> {
        let sorted = s.windows(2).all(|w| w[0] <= w[1]);
        let in_range = s.last().is_none_or(|&e| e <= self.n);
        (sorted && in_range).then_some(Offsets {
            s,
            brand: self.brand,
        })
    }

    /// Views the `n` elements as a row-major grid `width` wide. Requires `width > 0` and `n` a
    /// multiple of `width`.
    #[inline(always)]
    #[allow(clippy::manual_is_multiple_of)] // `is_multiple_of` needs 1.87; MSRV is 1.85
    pub fn grid(self, width: usize) -> Option<Grid<'id>> {
        (width > 0 && self.n % width == 0).then(|| Grid {
            w: width,
            h: self.n / width,
            brand: self.brand,
        })
    }

    /// Brands `s` if its length is exactly `n`.
    #[inline(always)]
    pub fn brand<'a, T>(self, s: &'a [T]) -> Option<Slice<'id, 'a, T>> {
        (s.len() == self.n).then_some(Slice {
            s,
            brand: self.brand,
        })
    }

    /// Brands `s` mutably if its length is exactly `n`.
    #[inline(always)]
    pub fn brand_mut<'a, T>(self, s: &'a mut [T]) -> Option<SliceMut<'id, 'a, T>> {
        if s.len() == self.n {
            Some(SliceMut {
                s,
                brand: self.brand,
            })
        } else {
            None
        }
    }
}

impl fmt::Debug for Len<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Len({})", self.n)
    }
}

// ---------------------------------------------------------------------------------------------
// Index representation
// ---------------------------------------------------------------------------------------------

mod sealed {
    pub trait Sealed {}
}

/// Integer types an [`Index`] can be stored as. Sealed: implemented for `u8`, `u16`, `u32`,
/// `u64` and `usize`.
pub trait Repr: sealed::Sealed + Copy + Eq + Ord + Hash + fmt::Debug {
    #[doc(hidden)]
    fn try_to_usize(self) -> Option<usize>;
    #[doc(hidden)]
    fn try_from_usize(u: usize) -> Option<Self>;
    /// Widening cast. Lossless for every value this crate validated, because validation proved
    /// the value is `< n <= usize::MAX`.
    #[doc(hidden)]
    fn widen(self) -> usize;
    /// Narrowing cast. Only called after proving the value fits.
    #[doc(hidden)]
    fn narrow(u: usize) -> Self;
}

macro_rules! impl_repr {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl Repr for $t {
            #[inline(always)]
            fn try_to_usize(self) -> Option<usize> { usize::try_from(self).ok() }
            #[inline(always)]
            fn try_from_usize(u: usize) -> Option<Self> { <$t>::try_from(u).ok() }
            #[inline(always)]
            fn widen(self) -> usize { self as usize }
            #[inline(always)]
            fn narrow(u: usize) -> Self { u as $t }
        }
    )*};
}
impl_repr!(u8, u16, u32, u64, usize);

// ---------------------------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------------------------

/// An index proven to be `< n` for the [`Len<'id>`] of the same brand.
///
/// `I` is the stored integer type; `Index<'id, u32>` is 4 bytes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Index<'id, I: Repr = usize> {
    raw: I,
    brand: Brand<'id>,
}

impl<'id, I: Repr> Index<'id, I> {
    /// The index as a `usize`.
    #[inline(always)]
    pub fn get(self) -> usize {
        self.raw.widen()
    }

    /// The index in its stored representation.
    #[inline(always)]
    pub fn raw(self) -> I {
        self.raw
    }

    /// Converts to another representation, checking that the value fits.
    #[inline(always)]
    pub fn to_repr<J: Repr>(self) -> Option<Index<'id, J>> {
        J::try_from_usize(self.get()).map(|raw| Index {
            raw,
            brand: self.brand,
        })
    }
}

impl<'id> Index<'id> {
    /// The previous index. Never needs a bound: `i - 1 < i < n`.
    #[inline(always)]
    pub fn prev(self) -> Option<Index<'id>> {
        self.checked_sub(1)
    }

    /// `i - k`, if it does not underflow. Never needs a bound: `i - k <= i < n`.
    #[inline(always)]
    pub fn checked_sub(self, k: usize) -> Option<Index<'id>> {
        self.raw.checked_sub(k).map(|raw| Index {
            raw,
            brand: self.brand,
        })
    }

    /// The next index, if it is `< n`.
    #[inline(always)]
    pub fn next(self, len: Len<'id>) -> Option<Index<'id>> {
        self.checked_add(1, len)
    }

    /// `i + k`, if it is `< n`.
    #[inline(always)]
    pub fn checked_add(self, k: usize, len: Len<'id>) -> Option<Index<'id>> {
        // `i < n`, so `i + k` overflows only if `k > usize::MAX - i`; checked_add handles it.
        self.raw.checked_add(k).and_then(|u| len.check(u))
    }
}

impl<I: Repr> fmt::Debug for Index<'_, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Index({:?})", self.raw)
    }
}

// ---------------------------------------------------------------------------------------------
// Range
// ---------------------------------------------------------------------------------------------

/// A range `start..end` with `start <= end <= n`. Iterates as [`Index<'id>`], and slices a
/// branded slice without a check.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Range<'id> {
    start: usize,
    end: usize,
    brand: Brand<'id>,
}

impl<'id> Range<'id> {
    /// Lower bound (inclusive).
    #[inline(always)]
    pub fn start(&self) -> usize {
        self.start
    }

    /// Upper bound (exclusive).
    #[inline(always)]
    pub fn end(&self) -> usize {
        self.end
    }

    /// Number of indices remaining.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// `true` when no indices remain.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// `true` if `i` lies in `start..end`.
    #[inline(always)]
    pub fn contains(&self, i: Index<'id>) -> bool {
        self.start <= i.raw && i.raw < self.end
    }

    /// Splits into `start..start + k` and `start + k..end`, if `k <= len`.
    #[inline(always)]
    pub fn split_at(self, k: usize) -> Option<(Range<'id>, Range<'id>)> {
        // `k <= end - start`, so `start <= mid <= end <= n`.
        (k <= self.len()).then(|| {
            let mid = self.start + k;
            let lo = Range {
                start: self.start,
                end: mid,
                brand: self.brand,
            };
            (
                lo,
                Range {
                    start: mid,
                    end: self.end,
                    brand: self.brand,
                },
            )
        })
    }

    /// The first index and the rest, if not empty.
    #[inline(always)]
    pub fn split_first(self) -> Option<(Index<'id>, Range<'id>)> {
        let brand = self.brand;
        let (first, rest) = self.split_at(1)?;
        Some((
            Index {
                raw: first.start,
                brand,
            },
            rest,
        ))
    }

    /// The rest and the last index, if not empty.
    #[inline(always)]
    pub fn split_last(self) -> Option<(Range<'id>, Index<'id>)> {
        let brand = self.brand;
        let k = self.len().checked_sub(1)?;
        let (rest, last) = self.split_at(k)?;
        Some((
            rest,
            Index {
                raw: last.start,
                brand,
            },
        ))
    }
}

impl<'id> Iterator for Range<'id> {
    type Item = Index<'id>;

    #[inline(always)]
    fn next(&mut self) -> Option<Index<'id>> {
        if self.start < self.end {
            let raw = self.start;
            self.start += 1;
            Some(Index {
                raw,
                brand: self.brand,
            })
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl DoubleEndedIterator for Range<'_> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            self.end -= 1;
            Some(Index {
                raw: self.end,
                brand: self.brand,
            })
        } else {
            None
        }
    }
}

impl ExactSizeIterator for Range<'_> {}
impl FusedIterator for Range<'_> {}

impl fmt::Debug for Range<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Range({}..{})", self.start, self.end)
    }
}

// ---------------------------------------------------------------------------------------------
// Offsets
// ---------------------------------------------------------------------------------------------

/// Non-decreasing offsets, each `<= n`, borrowed from the caller. Created by [`Len::offsets`].
///
/// The shared borrow keeps the offsets from changing after validation.
pub struct Offsets<'id, 'a> {
    s: &'a [usize],
    brand: Brand<'id>,
}

impl Clone for Offsets<'_, '_> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for Offsets<'_, '_> {}

impl<'id, 'a> Offsets<'id, 'a> {
    /// The validated offsets.
    #[inline(always)]
    pub fn as_slice(self) -> &'a [usize] {
        self.s
    }

    /// Ranges between adjacent offsets: `s[0]..s[1]`, `s[1]..s[2]`, and so on.
    #[inline(always)]
    pub fn ranges(self) -> Ranges<'id, 'a> {
        let (prev, rest) = match self.s.split_first() {
            Some((&first, rest)) => (first, rest),
            None => (0, &[][..]),
        };
        Ranges {
            prev,
            rest: rest.iter(),
            brand: self.brand,
        }
    }
}

/// Iterator over the ranges between adjacent [`Offsets`].
///
/// Carries the previous boundary in a register, so each step loads one offset, not two.
#[derive(Clone)]
pub struct Ranges<'id, 'a> {
    prev: usize,
    rest: core::slice::Iter<'a, usize>,
    brand: Brand<'id>,
}

impl<'id> Iterator for Ranges<'id, '_> {
    type Item = Range<'id>;

    #[inline(always)]
    fn next(&mut self) -> Option<Range<'id>> {
        let &end = self.rest.next()?;
        // Validated by `Len::offsets`: adjacent offsets satisfy `prev <= end <= n`.
        let r = Range {
            start: self.prev,
            end,
            brand: self.brand,
        };
        self.prev = end;
        Some(r)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.rest.size_hint()
    }
}

impl DoubleEndedIterator for Ranges<'_, '_> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        let &end = self.rest.next_back()?;
        let start = self.rest.as_slice().last().copied().unwrap_or(self.prev);
        Some(Range {
            start,
            end,
            brand: self.brand,
        })
    }
}

impl ExactSizeIterator for Ranges<'_, '_> {}
impl FusedIterator for Ranges<'_, '_> {}

impl fmt::Debug for Offsets<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Offsets").field(&self.s).finish()
    }
}

// ---------------------------------------------------------------------------------------------
// Persistent validated data
// ---------------------------------------------------------------------------------------------

/// Validation failure: the rejected input, returned unchanged, and where it failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Invalid<T> {
    /// The input, handed back so it is not lost.
    pub input: T,
    /// Position of the first element that failed validation.
    pub position: usize,
}

impl<T> fmt::Display for Invalid<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "validation failed at position {}", self.position)
    }
}

impl<T: fmt::Debug> core::error::Error for Invalid<T> {}

/// Owned indices, each validated `< bound`, that live outside any brand scope.
///
/// Validation is `O(n)`, once, at construction. Afterwards [`view`](Self::view) re-brands the
/// whole list for any [`Len<'id>`] with `n >= bound`, at the cost of one comparison. Keep it in
/// your own structs, across calls; open a scope only where you index.
///
/// The indices are immutable apart from checked [`push`](Self::push), so the invariant cannot
/// be broken after validation.
///
/// ```
/// use branded_index::{ValidIndices, scope};
///
/// struct Lookup {
///     idx: ValidIndices<u32>,
/// }
///
/// let data = [1.0, 2.0, 3.0];
/// let lookup = Lookup { idx: ValidIndices::new(vec![2u32, 0, 1, 2], data.len()).unwrap() };
///
/// // Later, any number of times:
/// let sum: f64 = scope(&data, |d| {
///     let idx = lookup.idx.view(d.len()).unwrap(); // one comparison
///     idx.iter().map(|&i| d[i]).sum()              // no bounds checks
/// });
/// assert_eq!(sum, 9.0);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValidIndices<I: Repr = usize> {
    v: Vec<I>,
    bound: usize,
}

impl<I: Repr> ValidIndices<I> {
    /// Validates that every element is `< bound`.
    pub fn new(v: Vec<I>, bound: usize) -> Result<Self, Invalid<Vec<I>>> {
        match v
            .iter()
            .position(|&i| !i.try_to_usize().is_some_and(|u| u < bound))
        {
            Some(position) => Err(Invalid { input: v, position }),
            None => Ok(ValidIndices { v, bound }),
        }
    }

    /// An empty list for indices `< bound`.
    pub fn empty(bound: usize) -> Self {
        ValidIndices {
            v: Vec::new(),
            bound,
        }
    }

    /// Persists indices that were already validated inside a scope. No checks.
    pub fn from_branded<'id>(len: Len<'id>, v: Vec<Index<'id, I>>) -> Self {
        ValidIndices {
            v: v.into_iter().map(Index::raw).collect(),
            bound: len.n,
        }
    }

    /// Appends `i` if it is `< bound`; otherwise hands it back.
    pub fn push(&mut self, i: I) -> Result<(), I> {
        if i.try_to_usize().is_some_and(|u| u < self.bound) {
            self.v.push(i);
            Ok(())
        } else {
            Err(i)
        }
    }

    /// The bound every index is below.
    #[inline(always)]
    pub fn bound(&self) -> usize {
        self.bound
    }

    /// Number of indices.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.v.len()
    }

    /// `true` when there are no indices.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }

    /// The raw indices.
    #[inline(always)]
    pub fn as_slice(&self) -> &[I] {
        &self.v
    }

    /// The raw indices, by value.
    pub fn into_inner(self) -> Vec<I> {
        self.v
    }

    /// The indices branded for `len`, if `len.get() >= bound`. One comparison, no copy.
    #[inline(always)]
    pub fn view<'id>(&self, len: Len<'id>) -> Option<&[Index<'id, I>]> {
        if self.bound <= len.n {
            // SAFETY: `Index<'id, I>` is `repr(transparent)` over `I` (the brand is a 1-aligned
            // ZST), so `[I]` and `[Index<'id, I>]` have identical layout. Every element is
            // `< bound <= n` (validated by `new`/`push`, or already branded in `from_branded`),
            // which is exactly the invariant `Index<'id, I>` requires. The shared borrow of
            // `self` keeps the elements from changing while the view lives.
            Some(unsafe { core::slice::from_raw_parts(self.v.as_ptr().cast(), self.v.len()) })
        } else {
            None
        }
    }
}

/// Owned fenceposts, validated non-decreasing with every element `<= bound`, that live
/// outside any brand scope. The persistent form of [`Offsets`].
///
/// ```
/// use branded_index::{ValidOffsets, scope};
///
/// let words = ValidOffsets::new(vec![0, 5, 5, 11], 11).unwrap(); // in a long-lived struct
/// let text = *b"helloworld!";
/// scope(&text, |t| {
///     let parts: Vec<&[u8]> = words.view(t.len()).unwrap().ranges().map(|r| &t[r]).collect();
///     assert_eq!(parts, [&b"hello"[..], b"", b"world!"]);
/// });
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValidOffsets {
    v: Vec<usize>,
    bound: usize,
}

impl ValidOffsets {
    /// Validates that `v` is non-decreasing and every element is `<= bound`.
    pub fn new(v: Vec<usize>, bound: usize) -> Result<Self, Invalid<Vec<usize>>> {
        let bad = (0..v.len()).find(|&i| v[i] > bound || (i > 0 && v[i] < v[i - 1]));
        match bad {
            Some(position) => Err(Invalid { input: v, position }),
            None => Ok(ValidOffsets { v, bound }),
        }
    }

    /// The bound every offset is at most.
    #[inline(always)]
    pub fn bound(&self) -> usize {
        self.bound
    }

    /// Number of ranges, one less than the number of offsets.
    #[inline(always)]
    pub fn ranges_len(&self) -> usize {
        self.v.len().saturating_sub(1)
    }

    /// The raw offsets.
    #[inline(always)]
    pub fn as_slice(&self) -> &[usize] {
        &self.v
    }

    /// The raw offsets, by value.
    pub fn into_inner(self) -> Vec<usize> {
        self.v
    }

    /// The offsets branded for `len`, if `len.get() >= bound`. One comparison, no copy.
    #[inline(always)]
    pub fn view<'id>(&self, len: Len<'id>) -> Option<Offsets<'id, '_>> {
        // Non-decreasing and `<= bound <= n`: exactly what `Len::offsets` would check.
        (self.bound <= len.n).then_some(Offsets {
            s: &self.v,
            brand: len.brand,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// AppendVec
// ---------------------------------------------------------------------------------------------

/// Single-use permission to create one [`AppendVec`] for brand `'id`. Not `Clone`: one vector
/// per brand is what keeps its indices sound.
pub struct AppendToken<'id> {
    brand: Brand<'id>,
}

impl fmt::Debug for AppendToken<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AppendToken")
    }
}

/// Opens a scope with a fresh brand for one [`AppendVec`].
///
/// A token rather than a ready-made vector lets the element type mention the brand, so
/// elements can hold indices into their own vector.
#[inline(always)]
pub fn append_scope<R>(f: impl for<'id> FnOnce(AppendToken<'id>) -> R) -> R {
    f(AppendToken {
        brand: Brand::new(),
    })
}

/// A vector that only grows. Every [`Index<'id>`] it hands out stays valid for the rest of the
/// scope, so elements can refer to each other by index (graphs, arenas, interners) and every
/// access is check-free.
///
/// Nothing removes elements, and no [`Len<'id>`] exists while the vector can still grow:
/// indices come only from [`push`](Self::push), [`extend`](Self::extend), [`check`](Self::check)
/// and [`indices`](Self::indices). [`freeze`](Self::freeze) ends growth and returns the final
/// `Len<'id>`, which connects to the rest of the crate.
///
/// ```
/// use branded_index::{AppendVec, Index, append_scope};
///
/// // Edges point at nodes of the same vector.
/// struct Node<'id> {
///     name: &'static str,
///     edges: Vec<Index<'id>>,
/// }
///
/// let names = append_scope(|token| {
///     let mut g = AppendVec::new(token);
///     let a = g.push(Node { name: "a", edges: vec![] });
///     let b = g.push(Node { name: "b", edges: vec![a] });
///     let c = g.push(Node { name: "c", edges: vec![a, b] });
///     g[a].edges.push(c); // cycles are fine
///     g[c].edges.iter().map(|&e| g[e].name).collect::<Vec<_>>() // no bounds checks
/// });
/// assert_eq!(names, ["a", "b"]);
/// ```
///
/// The token cannot be duplicated to make a second vector with the same brand:
///
/// ```compile_fail,E0382
/// branded_index::append_scope(|token| {
///     let mut a = branded_index::AppendVec::new(token);
///     let mut b = branded_index::AppendVec::new(token);
///     a.push(1);
///     b.push(2);
/// });
/// ```
pub struct AppendVec<'id, T> {
    v: Vec<T>,
    brand: Brand<'id>,
}

impl<'id, T> AppendVec<'id, T> {
    /// An empty vector for the token's brand.
    #[inline]
    pub fn new(token: AppendToken<'id>) -> Self {
        Self::from_vec(token, Vec::new())
    }

    /// Takes over `v`; its existing elements are reachable through [`indices`](Self::indices).
    #[inline]
    pub fn from_vec(token: AppendToken<'id>, v: Vec<T>) -> Self {
        AppendVec {
            v,
            brand: token.brand,
        }
    }

    /// Appends `x` and returns its index, valid for the rest of the scope.
    #[inline]
    pub fn push(&mut self, x: T) -> Index<'id> {
        let raw = self.v.len();
        self.v.push(x);
        Index {
            raw,
            brand: self.brand,
        }
    }

    /// Appends every item and returns the range of their indices.
    pub fn extend(&mut self, items: impl IntoIterator<Item = T>) -> Range<'id> {
        let start = self.v.len();
        self.v.extend(items);
        Range {
            start,
            end: self.v.len(),
            brand: self.brand,
        }
    }

    /// Validates `i < len`. Valid forever after, since the length never shrinks.
    #[inline(always)]
    pub fn check(&self, i: usize) -> Option<Index<'id>> {
        (i < self.v.len()).then_some(Index {
            raw: i,
            brand: self.brand,
        })
    }

    /// All current indices. They stay valid while more elements are pushed.
    #[inline(always)]
    pub fn indices(&self) -> Range<'id> {
        Range {
            start: 0,
            end: self.v.len(),
            brand: self.brand,
        }
    }

    /// The index of the last element, if any.
    #[inline(always)]
    pub fn last_index(&self) -> Option<Index<'id>> {
        self.v.len().checked_sub(1).map(|raw| Index {
            raw,
            brand: self.brand,
        })
    }

    /// Current number of elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.v.len()
    }

    /// `true` when there are no elements yet.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }

    /// Reserves capacity for at least `additional` more elements.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.v.reserve(additional);
    }

    /// Element at `i`. Never checks.
    #[inline(always)]
    pub fn get<I: Repr>(&self, i: Index<'id, I>) -> &T {
        // SAFETY: while this vector exists, indices of brand `'id` come only from its own
        // `push`, `extend`, `check`, `indices` and `last_index`, each `< len` at creation (the
        // token is consumed, so no second vector and no `Len<'id>` can exist). The length never
        // decreases: nothing removes elements and no `&mut Vec<T>` is handed out. So
        // `i < len` now.
        unsafe { self.v.get_unchecked(i.get()) }
    }

    /// Mutable element at `i`. Never checks.
    #[inline(always)]
    pub fn get_mut<I: Repr>(&mut self, i: Index<'id, I>) -> &mut T {
        // SAFETY: as in `get`.
        unsafe { self.v.get_unchecked_mut(i.get()) }
    }

    /// The elements as a slice.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        &self.v
    }

    /// The elements as a mutable slice. Its length cannot change through it.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.v
    }

    /// Elements paired with their indices.
    #[inline(always)]
    pub fn iter_indexed(
        &self,
    ) -> impl DoubleEndedIterator<Item = (Index<'id>, &T)> + ExactSizeIterator {
        self.indices().zip(self.v.iter())
    }

    /// Ends growth. Returns the final branded length and the elements; every index handed out
    /// is `< n`, so they work with [`Len::brand`] slices and [`ValidIndices::from_branded`].
    #[inline]
    pub fn freeze(self) -> (Len<'id>, Vec<T>) {
        (
            Len {
                n: self.v.len(),
                brand: self.brand,
            },
            self.v,
        )
    }
}

impl<T: fmt::Debug> fmt::Debug for AppendVec<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AppendVec").field(&self.v).finish()
    }
}

impl<'id, T, I: Repr> ops::Index<Index<'id, I>> for AppendVec<'id, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: Index<'id, I>) -> &T {
        self.get(i)
    }
}

impl<'id, T, I: Repr> ops::IndexMut<Index<'id, I>> for AppendVec<'id, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: Index<'id, I>) -> &mut T {
        self.get_mut(i)
    }
}

// ---------------------------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------------------------

/// The `n` elements of brand `'id` viewed as a row-major `height x width` grid.
/// Created by [`Len::grid`]; `width * height == n` exactly.
#[derive(Clone, Copy)]
pub struct Grid<'id> {
    w: usize,
    h: usize,
    brand: Brand<'id>,
}

impl<'id> Grid<'id> {
    /// Number of columns.
    #[inline(always)]
    pub fn width(self) -> usize {
        self.w
    }

    /// Number of rows.
    #[inline(always)]
    pub fn height(self) -> usize {
        self.h
    }

    /// The branded length `width * height`.
    #[inline(always)]
    pub fn len(self) -> Len<'id> {
        Len {
            n: self.w * self.h,
            brand: self.brand,
        }
    }

    /// Index of cell (`row`, `col`), checked.
    #[inline(always)]
    pub fn at(self, row: usize, col: usize) -> Option<Index<'id>> {
        // `row * w + col < h * w == n`, so no overflow.
        (row < self.h && col < self.w).then(|| Index {
            raw: row * self.w + col,
            brand: self.brand,
        })
    }

    /// The interior cells at distance `>= R` from every edge, one row at a time. Every cell
    /// produced can reach all neighbours within `R` rows and `R` columns without a check.
    /// Empty when the grid is narrower or shorter than `2 * R + 1`.
    #[inline(always)]
    pub fn interior_rows<const R: usize>(self) -> InteriorRows<'id, R> {
        let fits = R.checked_mul(2).is_some_and(|d| self.w > d && self.h > d);
        let (row, row_end) = if fits { (R, self.h - R) } else { (0, 0) };
        InteriorRows {
            row,
            row_end,
            w: self.w,
            brand: self.brand,
        }
    }
}

impl fmt::Debug for Grid<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Grid({}x{})", self.h, self.w)
    }
}

/// A grid cell with at least `R` cells of margin on every side. Created by
/// [`Grid::interior_rows`].
///
/// Invariant: `raw == row * w + col` with `R <= row < h - R` and `R <= col < w - R`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Interior<'id, const R: usize> {
    raw: usize,
    w: usize,
    brand: Brand<'id>,
}

impl<'id, const R: usize> Interior<'id, R> {
    /// The cell itself.
    #[inline(always)]
    pub fn center(self) -> Index<'id> {
        Index {
            raw: self.raw,
            brand: self.brand,
        }
    }

    /// The cell `DR` rows and `DC` columns away. `|DR|` and `|DC|` must be `<= R`; this is
    /// checked at compile time, so the access itself needs no runtime check.
    ///
    /// ```compile_fail,E0080
    /// branded_index::with_len(25, |n| {
    ///     let g = n.grid(5).unwrap();
    ///     for row in g.interior_rows::<1>() {
    ///         for i in row {
    ///             let _ = i.at::<2, 0>(); // two rows away, but the margin is only 1
    ///         }
    ///     }
    /// });
    /// ```
    #[inline(always)]
    pub fn at<const DR: isize, const DC: isize>(self) -> Index<'id> {
        const {
            assert!(
                DR.unsigned_abs() <= R && DC.unsigned_abs() <= R,
                "stencil offset exceeds the interior margin R"
            )
        };
        // The target is `(row + DR) * w + (col + DC)`. With `|DR|, |DC| <= R` and the invariant,
        // `0 <= row + DR < h` and `0 <= col + DC < w`, so the true value lies in `0..n`.
        // Wrapping arithmetic computes it modulo 2^usize::BITS, which is then exact.
        let offset = (DR as usize).wrapping_mul(self.w).wrapping_add(DC as usize);
        Index {
            raw: self.raw.wrapping_add(offset),
            brand: self.brand,
        }
    }

    /// One row up. Needs `R >= 1`.
    #[inline(always)]
    pub fn up(self) -> Index<'id> {
        self.at::<-1, 0>()
    }

    /// One row down. Needs `R >= 1`.
    #[inline(always)]
    pub fn down(self) -> Index<'id> {
        self.at::<1, 0>()
    }

    /// One column left. Needs `R >= 1`.
    #[inline(always)]
    pub fn left(self) -> Index<'id> {
        self.at::<0, -1>()
    }

    /// One column right. Needs `R >= 1`.
    #[inline(always)]
    pub fn right(self) -> Index<'id> {
        self.at::<0, 1>()
    }

    /// Row of the cell.
    #[inline(always)]
    pub fn row(self) -> usize {
        self.raw / self.w
    }

    /// Column of the cell.
    #[inline(always)]
    pub fn col(self) -> usize {
        self.raw % self.w
    }
}

impl<const R: usize> fmt::Debug for Interior<'_, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Interior<{R}>({}, {})", self.row(), self.col())
    }
}

/// Iterator over interior rows; see [`Grid::interior_rows`].
#[derive(Clone)]
pub struct InteriorRows<'id, const R: usize> {
    row: usize,
    row_end: usize,
    w: usize,
    brand: Brand<'id>,
}

impl<'id, const R: usize> Iterator for InteriorRows<'id, R> {
    type Item = InteriorRow<'id, R>;

    #[inline(always)]
    fn next(&mut self) -> Option<InteriorRow<'id, R>> {
        if self.row < self.row_end {
            // Nonempty only if `w > 2R` (checked in `interior_rows`), so `R < w - R`.
            let base = self.row * self.w;
            self.row += 1;
            Some(InteriorRow {
                cur: base + R,
                end: base + self.w - R,
                w: self.w,
                brand: self.brand,
            })
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.row_end - self.row;
        (n, Some(n))
    }
}

impl<const R: usize> ExactSizeIterator for InteriorRows<'_, R> {}
impl<const R: usize> FusedIterator for InteriorRows<'_, R> {}

/// The interior cells of one row; see [`Grid::interior_rows`].
#[derive(Clone)]
pub struct InteriorRow<'id, const R: usize> {
    cur: usize,
    end: usize,
    w: usize,
    brand: Brand<'id>,
}

impl<'id, const R: usize> Iterator for InteriorRow<'id, R> {
    type Item = Interior<'id, R>;

    #[inline(always)]
    fn next(&mut self) -> Option<Interior<'id, R>> {
        if self.cur < self.end {
            let raw = self.cur;
            self.cur += 1;
            Some(Interior {
                raw,
                w: self.w,
                brand: self.brand,
            })
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.end - self.cur;
        (n, Some(n))
    }
}

impl<const R: usize> DoubleEndedIterator for InteriorRow<'_, R> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.cur < self.end {
            self.end -= 1;
            Some(Interior {
                raw: self.end,
                w: self.w,
                brand: self.brand,
            })
        } else {
            None
        }
    }
}

impl<const R: usize> ExactSizeIterator for InteriorRow<'_, R> {}
impl<const R: usize> FusedIterator for InteriorRow<'_, R> {}

// ---------------------------------------------------------------------------------------------
// Slice / SliceMut
// ---------------------------------------------------------------------------------------------

/// A shared slice proven to have length exactly `n` for brand `'id`.
pub struct Slice<'id, 'a, T> {
    s: &'a [T],
    brand: Brand<'id>,
}

impl<T> Clone for Slice<'_, '_, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Slice<'_, '_, T> {}

impl<'id, 'a, T> Slice<'id, 'a, T> {
    /// The branded length.
    #[inline(always)]
    pub fn len(self) -> Len<'id> {
        Len {
            n: self.s.len(),
            brand: self.brand,
        }
    }

    /// Elements paired with their branded indices.
    #[inline(always)]
    pub fn iter_indexed(
        self,
    ) -> impl DoubleEndedIterator<Item = (Index<'id>, &'a T)> + ExactSizeIterator {
        self.len().indices().zip(self.s)
    }

    /// Binary search for `x` in a sorted slice. `Ok` carries a branded index of a match; `Err`
    /// the insertion point in `0..=n`, as in [`slice::binary_search`].
    #[inline]
    pub fn binary_search(self, x: &T) -> Result<Index<'id>, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|e| e.cmp(x))
    }

    /// [`binary_search`](Self::binary_search) with a comparator.
    #[inline]
    pub fn binary_search_by(
        self,
        f: impl FnMut(&'a T) -> core::cmp::Ordering,
    ) -> Result<Index<'id>, usize> {
        // One extra comparison after an O(log n) search, instead of trusting std's result for
        // soundness.
        let len = self.len();
        self.s.binary_search_by(f).map(|i| {
            len.check(i)
                .expect("binary_search returned an in-bounds index")
        })
    }

    /// Splits `0..n` at the first element where `pred` is false, for a slice partitioned by
    /// `pred` (as in [`slice::partition_point`]). Returns the branded ranges where `pred`
    /// holds and where it does not.
    #[inline]
    pub fn partition_point(self, pred: impl FnMut(&T) -> bool) -> (Range<'id>, Range<'id>) {
        let p = self.s.partition_point(pred);
        self.len()
            .indices()
            .split_at(p)
            .expect("partition_point is at most the length")
    }

    /// The underlying slice.
    #[inline(always)]
    pub fn as_slice(self) -> &'a [T] {
        self.s
    }

    /// Element at `i`, with the lifetime of the underlying slice. Never checks.
    #[inline(always)]
    pub fn get<I: Repr>(self, i: Index<'id, I>) -> &'a T {
        // SAFETY: `i` was validated `< n` against the `Len<'id>` of this brand (`Len::check*`,
        // `Range` iteration), and this slice was validated to have length exactly `n` for the
        // same brand (`Len::brand`, `scope`). Brands are invariant and unique per scope, so
        // both facts refer to the same `n`. Hence `i < self.s.len()`.
        unsafe { self.s.get_unchecked(i.get()) }
    }

    /// Sub-slice for `r`. Never checks.
    #[inline(always)]
    pub fn slice(self, r: Range<'id>) -> &'a [T] {
        // SAFETY: `r.start <= r.end <= n` (validated by `Len::range` or derived from
        // `Len::indices`), and `self.s.len() == n` for the same brand.
        unsafe { self.s.get_unchecked(r.start..r.end) }
    }
}

impl<'id, T, I: Repr> ops::Index<Index<'id, I>> for Slice<'id, '_, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: Index<'id, I>) -> &T {
        self.get(i)
    }
}

impl<'id, T> ops::Index<Range<'id>> for Slice<'id, '_, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, r: Range<'id>) -> &[T] {
        self.slice(r)
    }
}

/// A mutable slice proven to have length exactly `n` for brand `'id`.
///
/// Handing out `&mut [T]` is fine: a slice's length cannot change through it.
pub struct SliceMut<'id, 'a, T> {
    s: &'a mut [T],
    brand: Brand<'id>,
}

impl<'id, 'a, T> SliceMut<'id, 'a, T> {
    /// The branded length.
    #[inline(always)]
    pub fn len(&self) -> Len<'id> {
        Len {
            n: self.s.len(),
            brand: self.brand,
        }
    }

    /// Mutable elements paired with their branded indices.
    #[inline(always)]
    pub fn iter_indexed_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (Index<'id>, &mut T)> + ExactSizeIterator {
        self.len().indices().zip(self.s.iter_mut())
    }

    /// The underlying slice, shared.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.s
    }

    /// The underlying slice, mutable.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.s
    }

    /// A shorter-lived branded view, so the slice can be passed on and used again afterwards.
    #[inline(always)]
    pub fn reborrow(&mut self) -> SliceMut<'id, '_, T> {
        SliceMut {
            s: self.s,
            brand: self.brand,
        }
    }

    /// A shared branded view.
    #[inline(always)]
    pub fn as_shared(&self) -> Slice<'id, '_, T> {
        Slice {
            s: self.s,
            brand: self.brand,
        }
    }

    /// Mutable element at `i`. Never checks.
    #[inline(always)]
    pub fn get_mut<I: Repr>(&mut self, i: Index<'id, I>) -> &mut T {
        // SAFETY: same argument as `Slice::get`: `i < n == self.s.len()`.
        unsafe { self.s.get_unchecked_mut(i.get()) }
    }

    /// Mutable sub-slice for `r`. Never checks.
    #[inline(always)]
    pub fn slice_mut(&mut self, r: Range<'id>) -> &mut [T] {
        // SAFETY: same argument as `Slice::slice`: `r.start <= r.end <= n == self.s.len()`.
        unsafe { self.s.get_unchecked_mut(r.start..r.end) }
    }

    /// Swaps two elements. Never checks.
    #[inline(always)]
    pub fn swap<I: Repr>(&mut self, a: Index<'id, I>, b: Index<'id, I>) {
        let p = self.s.as_mut_ptr();
        // SAFETY: both indices are `< n == self.s.len()`, so both pointers are in bounds of the
        // same allocation. `ptr::swap` allows `a == b` (overlapping).
        unsafe { core::ptr::swap(p.add(a.get()), p.add(b.get())) }
    }
}

impl<'id, T, I: Repr> ops::Index<Index<'id, I>> for SliceMut<'id, '_, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: Index<'id, I>) -> &T {
        // SAFETY: `i < n == self.s.len()`, see `Slice::get`.
        unsafe { self.s.get_unchecked(i.get()) }
    }
}

impl<'id, T, I: Repr> ops::IndexMut<Index<'id, I>> for SliceMut<'id, '_, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: Index<'id, I>) -> &mut T {
        self.get_mut(i)
    }
}

impl<'id, T> ops::Index<Range<'id>> for SliceMut<'id, '_, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, r: Range<'id>) -> &[T] {
        // SAFETY: `r.start <= r.end <= n == self.s.len()`, see `Slice::slice`.
        unsafe { self.s.get_unchecked(r.start..r.end) }
    }
}

impl<'id, T> ops::IndexMut<Range<'id>> for SliceMut<'id, '_, T> {
    #[inline(always)]
    fn index_mut(&mut self, r: Range<'id>) -> &mut [T] {
        self.slice_mut(r)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn check_boundaries() {
        with_len(3, |n| {
            assert_eq!(n.check(0).map(Index::get), Some(0));
            assert_eq!(n.check(2).map(Index::get), Some(2));
            assert!(n.check(3).is_none());
            assert!(n.check(usize::MAX).is_none());
        });
        with_len(0, |n| {
            assert!(n.check(0).is_none());
            assert_eq!(n.indices().count(), 0);
        });
    }

    #[test]
    fn check_as_boundaries() {
        with_len(300, |n| {
            assert!(n.check_as(255u8).is_some());
            assert!(n.check_as(299u16).is_some());
            assert!(n.check_as(300u16).is_none());
            assert!(n.check_as(u64::MAX).is_none());
        });
    }

    #[test]
    fn indices_as_fit() {
        with_len(256, |n| {
            let v: Vec<_> = n.indices_as::<u8>().unwrap().map(Index::get).collect();
            assert_eq!(v.len(), 256);
            assert_eq!(v[255], 255);
        });
        with_len(257, |n| assert!(n.indices_as::<u8>().is_none()));
        with_len(0, |n| assert_eq!(n.indices_as::<u8>().unwrap().count(), 0));
    }

    #[test]
    fn to_repr_checks_fit() {
        with_len(1000, |n| {
            assert!(n.check(255).unwrap().to_repr::<u8>().is_some());
            assert!(n.check(256).unwrap().to_repr::<u8>().is_none());
        });
    }

    #[test]
    fn range_validation() {
        with_len(5, |n| {
            assert!(n.range(0, 5).is_some());
            assert!(n.range(5, 5).is_some());
            assert!(n.range(3, 2).is_none());
            assert!(n.range(0, 6).is_none());
            let r = n.range(1, 4).unwrap();
            assert_eq!(r.clone().map(Index::get).collect::<Vec<_>>(), [1, 2, 3]);
            assert_eq!(r.rev().map(Index::get).collect::<Vec<_>>(), [3, 2, 1]);
        });
    }

    #[test]
    fn offsets_validation() {
        with_len(10, |n| {
            let o = n.offsets(&[0, 3, 3, 10]).unwrap();
            let got: Vec<_> = o.ranges().map(|r| (r.start(), r.end())).collect();
            assert_eq!(got, [(0, 3), (3, 3), (3, 10)]);
            let back: Vec<_> = o.ranges().rev().map(|r| (r.start(), r.end())).collect();
            assert_eq!(back, [(3, 10), (3, 3), (0, 3)]);
            let mut it = o.ranges();
            assert_eq!(it.len(), 3);
            assert_eq!(it.next_back().map(|r| r.start()), Some(3));
            assert_eq!(it.next().map(|r| r.end()), Some(3));
            assert_eq!(it.next().map(|r| (r.start(), r.end())), Some((3, 3)));
            assert!(it.next().is_none() && it.next_back().is_none());
            assert!(n.offsets(&[]).unwrap().ranges().next().is_none());
            assert!(n.offsets(&[4]).unwrap().ranges().next().is_none());
            assert!(n.offsets(&[0, 5, 4]).is_none());
            assert!(n.offsets(&[0, 11]).is_none());
            assert!(n.offsets(&[11]).is_none());
        });
    }

    #[test]
    fn grid_validation() {
        with_len(12, |n| {
            let g = n.grid(4).unwrap();
            assert_eq!((g.height(), g.width()), (3, 4));
            assert_eq!(g.at(2, 3).map(Index::get), Some(11));
            assert!(g.at(3, 0).is_none() && g.at(0, 4).is_none());
            assert!(n.grid(5).is_none());
            assert!(n.grid(0).is_none());
        });
    }

    #[test]
    fn interior_cells_and_neighbours() {
        // 4 rows x 5 cols; radius-1 interior is rows 1..3, cols 1..4.
        with_len(20, |n| {
            let g = n.grid(5).unwrap();
            let cells: Vec<_> = g
                .interior_rows::<1>()
                .flatten()
                .map(|i| (i.row(), i.col()))
                .collect();
            assert_eq!(cells, [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]);
            for i in g.interior_rows::<1>().flatten() {
                let c = i.center().get();
                assert_eq!(i.up().get(), c - 5);
                assert_eq!(i.down().get(), c + 5);
                assert_eq!(i.left().get(), c - 1);
                assert_eq!(i.right().get(), c + 1);
                assert_eq!(i.at::<-1, 1>().get(), c - 4);
                assert_eq!(i.at::<1, -1>().get(), c + 4);
            }
            let rev: Vec<_> = g
                .interior_rows::<1>()
                .next()
                .unwrap()
                .rev()
                .map(|i| i.col())
                .collect();
            assert_eq!(rev, [3, 2, 1]);
        });
    }

    #[test]
    fn interior_radius_two_reaches_corners() {
        with_len(7 * 6, |n| {
            let g = n.grid(6).unwrap();
            let cells: Vec<_> = g.interior_rows::<2>().flatten().collect();
            assert_eq!(cells.len(), 3 * 2);
            for i in cells {
                assert!(i.at::<-2, -2>().get() < 42 && i.at::<2, 2>().get() < 42);
                assert_eq!(i.at::<-2, -2>().get(), i.center().get() - 2 * 6 - 2);
            }
        });
    }

    #[test]
    fn interior_empty_when_too_small() {
        with_len(6, |n| {
            assert_eq!(n.grid(3).unwrap().interior_rows::<1>().count(), 0); // 2 rows
            assert_eq!(n.grid(2).unwrap().interior_rows::<1>().count(), 0); // 2 cols
            assert_eq!(n.grid(6).unwrap().interior_rows::<0>().flatten().count(), 6);
        });
        with_len(0, |n| {
            assert_eq!(n.grid(1).unwrap().interior_rows::<1>().count(), 0)
        });
        with_len(9, |n| {
            let g = n.grid(3).unwrap();
            assert_eq!(g.interior_rows::<{ usize::MAX }>().count(), 0);
            assert_eq!(
                g.interior_rows::<1>()
                    .flatten()
                    .map(|i| i.center().get())
                    .collect::<Vec<_>>(),
                [4]
            );
        });
    }

    #[test]
    fn valid_indices_validation() {
        let err = ValidIndices::new(alloc::vec![0u32, 4, 5, 1], 5).unwrap_err();
        assert_eq!((err.position, err.input.len()), (2, 4));
        assert!(ValidIndices::new(alloc::vec![u64::MAX], usize::MAX).is_err());
        assert!(ValidIndices::<u8>::new(Vec::new(), 0).unwrap().is_empty());

        let mut v = ValidIndices::new(alloc::vec![0u32, 4], 5).unwrap();
        assert_eq!(v.push(5), Err(5));
        assert_eq!(v.push(3), Ok(()));
        assert_eq!((v.as_slice(), v.bound()), (&[0, 4, 3][..], 5));
    }

    #[test]
    fn valid_indices_view() {
        let v = ValidIndices::new(alloc::vec![0u32, 4, 2], 5).unwrap();
        let data = [10, 11, 12, 13, 14, 15];
        with_len(4, |n| assert!(v.view(n).is_none()));
        // A larger `n` is fine: `i < 5 <= 6`.
        scope(&data, |d| {
            let idx = v.view(d.len()).unwrap();
            assert_eq!(idx.iter().map(|&i| d[i]).collect::<Vec<_>>(), [10, 14, 12]);
        });
    }

    #[test]
    fn valid_indices_from_branded_round_trip() {
        let v = with_len(7, |n| {
            let idx: Vec<_> = [6usize, 0, 3]
                .iter()
                .map(|&i| n.check_as(i as u16).unwrap())
                .collect();
            ValidIndices::from_branded(n, idx)
        });
        assert_eq!((v.as_slice(), v.bound()), (&[6u16, 0, 3][..], 7));
        with_len(7, |n| assert_eq!(v.view(n).unwrap()[0].get(), 6));
    }

    #[test]
    fn valid_offsets() {
        assert_eq!(
            ValidOffsets::new(alloc::vec![0, 3, 2], 5)
                .unwrap_err()
                .position,
            2
        );
        assert_eq!(
            ValidOffsets::new(alloc::vec![0, 6], 5)
                .unwrap_err()
                .position,
            1
        );
        assert_eq!(
            ValidOffsets::new(alloc::vec![7], 5).unwrap_err().position,
            0
        );
        let o = ValidOffsets::new(alloc::vec![0, 2, 2, 5], 5).unwrap();
        assert_eq!(o.ranges_len(), 3);
        with_len(4, |n| assert!(o.view(n).is_none()));
        with_len(5, |n| {
            let r: Vec<_> = o
                .view(n)
                .unwrap()
                .ranges()
                .map(|r| (r.start(), r.end()))
                .collect();
            assert_eq!(r, [(0, 2), (2, 2), (2, 5)]);
        });
        assert_eq!(ValidOffsets::new(Vec::new(), 0).unwrap().ranges_len(), 0);
    }

    #[test]
    fn index_arithmetic() {
        with_len(5, |n| {
            let (first, last) = (n.check(0).unwrap(), n.check(4).unwrap());
            assert!(first.prev().is_none());
            assert_eq!(last.prev().map(Index::get), Some(3));
            assert_eq!(last.checked_sub(4).map(Index::get), Some(0));
            assert!(last.checked_sub(5).is_none());
            assert!(last.next(n).is_none());
            assert_eq!(first.next(n).map(Index::get), Some(1));
            assert_eq!(first.checked_add(4, n).map(Index::get), Some(4));
            assert!(first.checked_add(5, n).is_none());
            assert!(last.checked_add(usize::MAX, n).is_none());
        });
    }

    #[test]
    fn range_splitting() {
        with_len(6, |n| {
            let r = n.range(1, 5).unwrap();
            assert!(r.contains(n.check(1).unwrap()) && r.contains(n.check(4).unwrap()));
            assert!(!r.contains(n.check(0).unwrap()) && !r.contains(n.check(5).unwrap()));
            let (a, b) = r.clone().split_at(3).unwrap();
            assert_eq!((a.start(), a.end(), b.start(), b.end()), (1, 4, 4, 5));
            let (a, b) = r.clone().split_at(4).unwrap();
            assert!(b.is_empty() && a.len() == 4);
            assert!(r.clone().split_at(5).is_none());
            let (f, rest) = r.clone().split_first().unwrap();
            assert_eq!((f.get(), rest.start()), (1, 2));
            let (rest, l) = r.split_last().unwrap();
            assert_eq!((rest.end(), l.get()), (4, 4));
            let empty = n.range(3, 3).unwrap();
            assert!(empty.clone().split_first().is_none() && empty.split_last().is_none());
        });
    }

    #[test]
    fn search_and_partition() {
        let v = [1, 3, 3, 7, 9];
        scope(&v, |s| {
            assert_eq!(s.binary_search(&7).map(|i| s[i]), Ok(7));
            assert_eq!(s.binary_search(&4), Err(3));
            assert_eq!(s.binary_search(&10), Err(5));
            let (lo, hi) = s.partition_point(|&x| x < 5);
            assert_eq!(&s[lo], &[1, 3, 3]);
            assert_eq!(&s[hi], &[7, 9]);
            let (lo, hi) = s.partition_point(|_| true);
            assert_eq!((lo.len(), hi.len()), (5, 0));
        });
        scope(&[] as &[i32], |s| {
            assert_eq!(s.binary_search(&1), Err(0));
            let (lo, hi) = s.partition_point(|_| false);
            assert!(lo.is_empty() && hi.is_empty());
        });
    }

    #[test]
    fn indexed_iteration() {
        let mut v = [10, 20, 30];
        scope_mut(&mut v, |mut s| {
            for (i, x) in s.iter_indexed_mut() {
                *x += i.get();
            }
            let pairs: Vec<_> = s
                .as_shared()
                .iter_indexed()
                .map(|(i, &x)| (i.get(), x))
                .collect();
            assert_eq!(pairs, [(0, 10), (1, 21), (2, 32)]);
            assert_eq!(
                s.as_shared()
                    .iter_indexed()
                    .next_back()
                    .map(|(i, _)| i.get()),
                Some(2)
            );
        });
    }

    #[test]
    fn append_vec_indices_survive_growth() {
        append_scope(|token| {
            let mut v = AppendVec::from_vec(token, alloc::vec![10, 11]);
            let early = v.indices();
            let a = v.push(12);
            let r = v.extend([13, 14, 15]);
            assert_eq!((r.start(), r.end()), (3, 6));
            v.reserve(100);
            for i in 0..1000 {
                v.push(i); // reallocations do not invalidate indices
            }
            assert_eq!(early.map(|i| v[i]).collect::<Vec<_>>(), [10, 11]);
            assert_eq!(v[a], 12);
            v[a] += 1;
            assert_eq!(*v.get(a), 13);
            assert!(v.check(1006).is_none());
            assert_eq!(v.check(1005).map(|i| v[i]), Some(999));
            assert_eq!(v.last_index().map(Index::get), Some(1005));
            assert_eq!(
                v.iter_indexed().nth(3).map(|(i, &x)| (i.get(), x)),
                Some((3, 13))
            );
            assert_eq!(v.len(), 1006);
        });
        append_scope(|token| {
            let v: AppendVec<'_, u8> = AppendVec::new(token);
            assert!(v.is_empty() && v.last_index().is_none() && v.check(0).is_none());
        });
    }

    #[test]
    fn append_vec_compact_indices_and_freeze() {
        let kept = append_scope(|token| {
            let mut v = AppendVec::new(token);
            let ids: Vec<Index<'_, u32>> = (0..5)
                .map(|i| v.push(i * 2).to_repr::<u32>().unwrap())
                .collect();
            assert_eq!(v[ids[3]], 6);
            let (len, data) = v.freeze();
            let s = len.brand(&data).unwrap();
            assert_eq!(ids.iter().map(|&i| s[i]).sum::<i32>(), 20);
            ValidIndices::from_branded(len, ids)
        });
        assert_eq!(
            (kept.as_slice(), kept.bound()),
            (&[0u32, 1, 2, 3, 4][..], 5)
        );
    }

    #[test]
    fn append_vec_self_referential_graph() {
        struct Node<'id> {
            edges: Vec<Index<'id>>,
        }
        let reachable = append_scope(|token| {
            let mut g = AppendVec::new(token);
            let root = g.push(Node { edges: Vec::new() });
            let mut prev = root;
            for _ in 0..10 {
                let n = g.push(Node {
                    edges: alloc::vec![prev],
                });
                prev = n;
            }
            g[root].edges.push(prev); // cycle
            let mut seen = alloc::vec![false; g.len()];
            let mut stack = alloc::vec![root];
            while let Some(n) = stack.pop() {
                if !core::mem::replace(&mut seen[n.get()], true) {
                    stack.extend(g[n].edges.iter().copied());
                }
            }
            seen.iter().filter(|&&s| s).count()
        });
        assert_eq!(reachable, 11);
    }

    #[test]
    fn brand_requires_exact_length() {
        let v = [1, 2, 3];
        with_len(3, |n| assert!(n.brand(&v).is_some()));
        with_len(2, |n| assert!(n.brand(&v).is_none()));
        with_len(4, |n| assert!(n.brand(&v).is_none()));
    }

    #[test]
    fn index_and_slice() {
        let v = [10, 20, 30, 40, 50];
        scope(&v, |s| {
            let i = s.len().check(3).unwrap();
            assert_eq!(s[i], 40);
            let r = s.len().range(1, 3).unwrap();
            assert_eq!(&s[r], &[20, 30]);
            assert_eq!(s.len().indices().map(|i| s[i]).sum::<i32>(), 150);
        });
    }

    #[test]
    fn mutation() {
        let mut v = [1, 2, 3, 4];
        scope_mut(&mut v, |mut s| {
            let n = s.len();
            for i in n.indices() {
                s[i] *= 10;
            }
            let (a, b) = (n.check(0).unwrap(), n.check(3).unwrap());
            s.swap(a, b);
            s.swap(a, a);
            let r = n.range(1, 3).unwrap();
            s[r].reverse();
        });
        assert_eq!(v, [40, 30, 20, 10]);
    }

    #[test]
    fn compact_index_is_small() {
        assert_eq!(core::mem::size_of::<Index<'static, u32>>(), 4);
        assert_eq!(
            core::mem::size_of::<Index<'static>>(),
            core::mem::size_of::<usize>()
        );
        assert_eq!(core::mem::size_of::<Option<Index<'static, u32>>>(), 8);
    }
}
