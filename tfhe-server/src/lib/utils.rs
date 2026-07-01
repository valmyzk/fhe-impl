use std::{io::{self, Write}, time::{Duration, Instant}};

use tfhe::{FheBool, FheUint16, core_crypto::commons::traits::CastFrom, prelude::FheTrivialEncrypt};

use crate::trivium_bool::TriviumStream;

/// Convert a 10-byte slice (80 bits) into a `[bool; 80]` array (LSB-first per byte).
pub fn bytes_to_bits(bytes: &[u8; 10]) -> [bool; 80] {
    std::array::from_fn(|i| (bytes[i / 8] >> (i % 8)) & 1 != 0)
}

/// A wrapper around a [`std::io::Write`] handler which executes a callback once a threshold has been reached.
pub struct ProgressWriter<W: Write, F: Fn(u64) -> ()> {
    inner: W,
    bytes_sent: u64,
    last_report: Instant,
    interval: Duration,
    f: F
}

impl<W: Write, F: Fn(u64) -> ()> ProgressWriter<W, F> {
    pub fn new(inner: W, interval: Duration, f: F) -> Self {
        Self {
            inner,
            bytes_sent: 0,
            last_report: Instant::now(),
            interval,
            f,
        }
    }
    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent
    } 
}

impl<W: Write, F: Fn(u64) -> ()> Write for ProgressWriter<W, F> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.bytes_sent += n as u64;
        if self.last_report.elapsed() >= self.interval {
            (self.f)(self.bytes_sent);
            self.last_report = Instant::now();
        }
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

pub fn encrypt_samples(trivium_stream: &mut TriviumStream<bool>, samples: &[u16; 4]) -> [u16; 4] {
    let keystream = trivium_stream.next_64();
    let keystream = keystream.as_chunks::<16>().0
        .into_iter()
        .map(|chunk| chunk.iter().enumerate().fold(0u16, |acc, (i, bit_i)| acc | ((*bit_i as u16) << i)));

    keystream
        .zip(samples.into_iter())
        .map(|(trivium, sample)| trivium ^ (*sample as u16))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

pub fn decrypt_samples(trivium_stream: &mut TriviumStream<FheBool>, samples: &[u16]) -> [FheUint16; 4] {
    let keystream = trivium_stream.next_64();
    let keystream = keystream.as_chunks::<16>().0
        .into_iter()
        .map(|chunk| chunk.into_iter().enumerate().fold(FheUint16::encrypt_trivial(0u8), |acc, (i, bit_i)| acc | (FheUint16::cast_from(bit_i.clone()) << i as u16)));

    keystream
        .zip(samples.into_iter())
        .map(|(trivium, sample)| trivium ^ (*sample as u16))
        .collect::<Vec<_>>()
        .try_into()
        .ok()
        .unwrap()
}