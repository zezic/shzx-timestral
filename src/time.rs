use std::{
    num::{NonZeroU64, NonZeroUsize},
    time::Duration,
};

use crate::{Error, Result};

/// A regular sampling grid in UTC Unix milliseconds. Missing observations remain NaN;
/// this grid never fills prices or assumes an exchange calendar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeGrid {
    start_ms: i64,
    step_ms: NonZeroU64,
    len: NonZeroUsize,
}

impl TimeGrid {
    pub fn new(start_ms: i64, interval: Duration, len: usize) -> Result<Self> {
        let millis = interval.as_millis();
        if millis == 0
            || millis > i64::MAX as u128
            || !interval.subsec_nanos().is_multiple_of(1_000_000)
        {
            return Err(Error::InvalidInput(
                "sampling interval must be a positive, whole number of milliseconds fitting i64"
                    .into(),
            ));
        }
        let grid = Self {
            start_ms,
            step_ms: NonZeroU64::new(millis as u64).expect("positive interval"),
            len: NonZeroUsize::new(len)
                .ok_or_else(|| Error::InvalidInput("time grid must not be empty".into()))?,
        };
        grid.timestamp_at(len - 1)?;
        Ok(grid)
    }

    #[must_use]
    pub fn len(self) -> usize {
        self.len.get()
    }
    #[must_use]
    pub fn is_empty(self) -> bool {
        false
    }
    #[must_use]
    pub fn start_ms(self) -> i64 {
        self.start_ms
    }
    #[must_use]
    pub fn interval(self) -> Duration {
        Duration::from_millis(self.step_ms.get())
    }
    #[must_use]
    pub fn timestamp(self, index: usize) -> Option<i64> {
        (index < self.len()).then(|| self.timestamp_at(index).expect("validated grid"))
    }
    pub fn timestamps(self) -> impl ExactSizeIterator<Item = i64> {
        (0..self.len()).map(move |i| self.timestamp(i).expect("index in grid"))
    }

    pub fn following(self, len: usize) -> Result<Self> {
        Self::new(self.timestamp_at(self.len())?, self.interval(), len)
    }

    /// Align strictly increasing `(timestamp_ms, value)` observations. Values on neither
    /// the grid nor its covered range are errors. f64 is converted to the model's f32;
    /// values outside its range are errors, and NaN retains its missing-value meaning.
    pub fn align_numeric(
        self,
        observations: impl IntoIterator<Item = (i64, f64)>,
    ) -> Result<Vec<f32>> {
        let mut aligned = Vec::new();
        aligned.try_reserve_exact(self.len()).map_err(|error| {
            Error::InvalidInput(format!(
                "cannot allocate {} grid observations: {error}",
                self.len()
            ))
        })?;
        aligned.resize(self.len(), f32::NAN);
        let mut previous = None;
        for (timestamp, value) in observations {
            if previous.is_some_and(|last| timestamp <= last) {
                return Err(Error::InvalidInput(
                    "observation timestamps must be strictly increasing".into(),
                ));
            }
            let offset = i128::from(timestamp) - i128::from(self.start_ms);
            let step = i128::from(self.step_ms.get());
            if offset < 0 || offset % step != 0 || offset / step >= self.len() as i128 {
                return Err(Error::InvalidInput(format!(
                    "timestamp {timestamp} is outside the sampling grid"
                )));
            }
            let numeric = value as f32;
            if numeric.is_infinite() {
                return Err(Error::InvalidInput(format!(
                    "observation at {timestamp} does not fit finite f32"
                )));
            }
            aligned[(offset / step) as usize] = numeric;
            previous = Some(timestamp);
        }
        Ok(aligned)
    }

    fn timestamp_at(self, index: usize) -> Result<i64> {
        let timestamp = i128::from(self.start_ms) + index as i128 * i128::from(self.step_ms.get());
        i64::try_from(timestamp)
            .map_err(|_| Error::InvalidInput("time grid overflows Unix milliseconds".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_gaps_and_rejects_ambiguous_alignment() {
        let grid = TimeGrid::new(1000, Duration::from_millis(100), 3).unwrap();
        let values = grid.align_numeric([(1000, 5.0), (1200, 7.0)]).unwrap();
        assert_eq!(values[0], 5.0);
        assert!(values[1].is_nan());
        assert_eq!(values[2], 7.0);
        assert_eq!(
            grid.following(2).unwrap().timestamps().collect::<Vec<_>>(),
            [1300, 1400]
        );
        assert!(grid.align_numeric([(1001, 1.0)]).is_err());
        assert!(grid.align_numeric([(1000, 1.0), (1000, 2.0)]).is_err());
        assert!(grid.align_numeric([(900, 1.0)]).is_err());
        assert!(grid.align_numeric([(1000, f64::MAX)]).is_err());
    }

    #[test]
    fn rejects_overflow_and_fractional_milliseconds() {
        assert!(TimeGrid::new(i64::MAX, Duration::from_millis(1), 2).is_err());
        assert!(TimeGrid::new(0, Duration::from_nanos(1), 2).is_err());
        assert!(TimeGrid::new(0, Duration::from_millis(1), 0).is_err());
    }

    #[test]
    fn rejects_unaddressable_alignment_buffers() {
        let grid = TimeGrid::new(i64::MIN, Duration::from_millis(1), usize::MAX).unwrap();
        assert!(grid.align_numeric([]).is_err());
    }
}
