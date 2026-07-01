//! StaticDeque: a compile-time fixed-size circular buffer.
//! Copied verbatim from https://github.com/zama-ai/tfhe-rs/blob/main/apps/trivium/src/static_deque/static_deque.rs

use core::ops::{Index, IndexMut};

/// StaticDeque: a struct implementing a deque whose size is known at compile time.
/// It has 2 members: the static array containing the data (never empty), and a cursor
/// equal to the index of the oldest element (and the next one to be overwritten).
#[derive(Clone)]
pub struct StaticDeque<const N: usize, T> {
    arr: [T; N],
    cursor: usize,
}

impl<const N: usize, T> StaticDeque<N, T> {
    /// Constructor always uses a fully initialized array, the first element of
    /// which is oldest, the last is newest
    pub fn new(_arr: [T; N]) -> Self {
        Self {
            arr: _arr,
            cursor: 0,
        }
    }

    /// Push a new element to the deque, overwriting the oldest at the same time.
    pub fn push(&mut self, val: T) {
        self.arr[self.cursor] = val;
        self.shift();
    }

    /// Shift: equivalent to pushing the oldest element
    pub fn shift(&mut self) {
        self.n_shifts(1);
    }

    /// computes n shift in a row
    pub fn n_shifts(&mut self, n: usize) {
        self.cursor += n;
        self.cursor %= N;
    }

    /// Getter for the internal memory
    #[allow(dead_code)]
    pub fn get_arr(&self) -> &[T; N] {
        &self.arr
    }
}

/// Index trait for the StaticDeque: 0 is the youngest element, N-1 is the oldest,
/// and above N will panic.
impl<const N: usize, T> Index<usize> for StaticDeque<N, T> {
    type Output = T;

    /// 0 is youngest
    fn index(&self, i: usize) -> &T {
        if i >= N {
            panic!("Index {i:?} too high for size {N:?}");
        }
        &self.arr[(N + self.cursor - i - 1) % N]
    }
}
/// IndexMut trait for the StaticDeque: 0 is the youngest element, N-1 is the oldest,
/// and above N will panic.
impl<const N: usize, T> IndexMut<usize> for StaticDeque<N, T> {
    /// 0 is youngest
    fn index_mut(&mut self, i: usize) -> &mut T {
        if i >= N {
            panic!("Index {i:?} too high for size {N:?}");
        }
        &mut self.arr[(N + self.cursor - i - 1) % N]
    }
}