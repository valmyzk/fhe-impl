use std::{cell::Cell, f32::consts::PI};

/// A Glucose monitor that emits the glucose level in blood, measure in dg/mL.
pub trait GlucoseMonitor {
    fn sample(&self) -> f32;
}

/// A basic, synthetic glucose monitor which samples a sine function to approximate samples.
pub struct SyntheticGcm(Cell<u16>);

impl SyntheticGcm {
    pub fn new() -> Self {
        Self(Cell::new(0))
    }
}

impl GlucoseMonitor for SyntheticGcm {
    fn sample(&self) -> f32 {
        // Example parameters
        const GLUCOSE_MIN: f32 = 40.0;
        const GLUCOSE_MAX: f32 = 400.0;
        const BASAL_LEVEL: f32 = 120.0;
        const WAVE_HEIGHT: f32 = 30.0;

        let n = f32::from(self.0.get());
        self.0.update(|n| n + 1);

        (WAVE_HEIGHT * (2.0 * PI * n / (24.0 * 60.0 / 5.0)).sin()
            + BASAL_LEVEL).clamp(GLUCOSE_MIN, GLUCOSE_MAX)
    }
}

