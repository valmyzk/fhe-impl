//! Trivium stream cipher with bool or FheBool internal representation.
//! Ported from https://github.com/zama-ai/tfhe-rs/tree/main/apps/trivium

use crate::static_deque::StaticDeque;
use rayon::prelude::*;
use tfhe::prelude::*;
use tfhe::{set_server_key, unset_server_key, FheBool, ServerKey};

pub const TRIVIUM_KEY_BITS: usize = 80;

/// Marker trait: types that can be used as Trivium register bits.
pub trait TriviumBoolInput<OpOutput>:
    Sized
    + Clone
    + std::ops::BitXor<Output = OpOutput>
    + std::ops::BitAnd<Output = OpOutput>
    + std::ops::Not<Output = OpOutput>
{
}
impl TriviumBoolInput<bool> for bool {}
impl TriviumBoolInput<bool> for &bool {}
impl TriviumBoolInput<FheBool> for FheBool {}
impl TriviumBoolInput<FheBool> for &FheBool {}

/// Trivium stream cipher parameterised over `T` (bool or FheBool).
/// The 1152 warmup rounds run inside the constructor before it returns.
pub struct TriviumStream<T> {
    a: StaticDeque<93, T>,
    b: StaticDeque<84, T>,
    c: StaticDeque<111, T>,
    fhe_key: Option<ServerKey>,
}

impl TriviumStream<bool> {
    pub fn new(key: [bool; TRIVIUM_KEY_BITS], iv: [bool; TRIVIUM_KEY_BITS]) -> TriviumStream<bool> {
        let mut a = [false; 93];
        let mut b = [false; 84];
        let mut c = [false; 111];

        for i in 0..TRIVIUM_KEY_BITS {
            a[93 - TRIVIUM_KEY_BITS + i] = key[i];
            b[84 - TRIVIUM_KEY_BITS + i] = iv[i];
        }
        c[0] = true;
        c[1] = true;
        c[2] = true;

        TriviumStream::<bool>::new_from_registers(a, b, c, None)
    }
}

impl TriviumStream<FheBool> {
    pub fn new(key: [FheBool; TRIVIUM_KEY_BITS], iv: [bool; TRIVIUM_KEY_BITS], sk: &ServerKey) -> TriviumStream<FheBool> {
        set_server_key(sk.clone());

        let mut a = [false; 93].map(FheBool::encrypt_trivial);
        let mut b = [false; 84].map(FheBool::encrypt_trivial);
        let mut c = [false; 111].map(FheBool::encrypt_trivial);

        for i in 0..TRIVIUM_KEY_BITS {
            a[93 - TRIVIUM_KEY_BITS + i] = key[i].clone();
            b[84 - TRIVIUM_KEY_BITS + i] = FheBool::encrypt_trivial(iv[i]);
        }
        c[0] = FheBool::encrypt_trivial(true);
        c[1] = FheBool::encrypt_trivial(true);
        c[2] = FheBool::encrypt_trivial(true);

        unset_server_key();
        TriviumStream::<FheBool>::new_from_registers(a, b, c, Some(sk.clone()))
    }
}

impl<T> TriviumStream<T>
where
    T: TriviumBoolInput<T> + Send + Sync,
    for<'a> &'a T: TriviumBoolInput<T>,
{
    fn new_from_registers(
        a: [T; 93],
        b: [T; 84],
        c: [T; 111],
        fhe_key: Option<ServerKey>,
    ) -> Self {
        let mut s = Self {
            a: StaticDeque::new(a),
            b: StaticDeque::new(b),
            c: StaticDeque::new(c),
            fhe_key,
        };
        s.init();
        s
    }

    /// 1152 = 18 × 64 warmup rounds (Trivium spec).
    fn init(&mut self) {
        for _ in 0..18 {
            self.next_64();
        }
    }

    /// Advance one clock step, return the output bit. Does NOT broadcast the
    /// server key to rayon workers — prefer `next_64` for FheBool streams.
    pub fn next_bool(&mut self) -> T {
        if let Some(sk) = &self.fhe_key {
            set_server_key(sk.clone());
        }
        let [o, a, b, c] = self.get_output_and_values(0);
        self.a.push(a);
        self.b.push(b);
        self.c.push(c);
        o
    }

    /// Compute the output + register-update values for a future step that is
    /// `n` steps ahead, without mutating state. Used for parallel batching.
    fn get_output_and_values(&self, n: usize) -> [T; 4] {
        assert!(n < 65);

        let (((temp_a, temp_b), (temp_c, a_and)), (b_and, c_and)) = rayon::join(
            || {
                rayon::join(
                    || {
                        rayon::join(
                            || &self.a[65 - n] ^ &self.a[92 - n],
                            || &self.b[68 - n] ^ &self.b[83 - n],
                        )
                    },
                    || {
                        rayon::join(
                            || &self.c[65 - n] ^ &self.c[110 - n],
                            || &self.a[91 - n] & &self.a[90 - n],
                        )
                    },
                )
            },
            || {
                rayon::join(
                    || &self.b[82 - n] & &self.b[81 - n],
                    || &self.c[109 - n] & &self.c[108 - n],
                )
            },
        );

        let ((o, a), (b, c)) = rayon::join(
            || {
                rayon::join(
                    || &(&temp_a ^ &temp_b) ^ &temp_c,
                    || &temp_c ^ &(&c_and ^ &self.a[68 - n]),
                )
            },
            || {
                rayon::join(
                    || &temp_a ^ &(&a_and ^ &self.b[77 - n]),
                    || &temp_b ^ &(&b_and ^ &self.c[86 - n]),
                )
            },
        );

        [o, a, b, c]
    }

    fn get_64_output_and_values(&self) -> Vec<[T; 4]> {
        (0..64)
            .into_par_iter()
            .map(|x| self.get_output_and_values(x))
            .rev()
            .collect()
    }

    /// Compute 64 clock steps in parallel. Returns keystream bits oldest-first.
    /// For FheBool streams this broadcasts the server key to all rayon workers.
    pub fn next_64(&mut self) -> Vec<T> {
        if let Some(sk) = &self.fhe_key {
            rayon::broadcast(|_| set_server_key(sk.clone()));
        }
        let mut values = self.get_64_output_and_values();
        if self.fhe_key.is_some() {
            rayon::broadcast(|_| unset_server_key());
        }

        let mut ret = Vec::<T>::with_capacity(64);
        while let Some([o, a, b, c]) = values.pop() {
            ret.push(o);
            self.a.push(a);
            self.b.push(b);
            self.c.push(c);
        }
        ret
    }
}
