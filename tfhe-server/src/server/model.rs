use tfhe::{FheInt16, FheUint16, prelude::*};

/// A model for homomorphically predicting glucose levels, given various samples with a 5-minute window.
pub trait GlucoseModel {
    fn predict(&self, samples: &[FheUint16]) -> FheUint16;
}

pub struct NaiveLinearRegression;
impl GlucoseModel for NaiveLinearRegression {
    fn predict(&self, samples: &[FheUint16]) -> FheUint16 {
        // Calculate the mean slope between samples and add it to the latest sample, "following the trend".
        // This can be simplified to: prediction = last + (last - first) / (n - 1).
        // Arithmetic is done in signed 16-bit to handle decreasing trends without wrap-around.
        let first = FheInt16::cast_from(samples.first().expect("nonempty slice").clone());
        let last = FheInt16::cast_from(samples.last().unwrap().clone());
        let n = FheInt16::encrypt_trivial(samples.len() as i16 - 1);
        let result = &last + (&last - &first) / n;
        FheUint16::cast_from(result)
    }
}