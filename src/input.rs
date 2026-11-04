use crate::{Error, Result, TimeGrid, categorical};

/// One aligned covariate channel. Category identity is local to this channel.
#[derive(Clone, Copy, Debug)]
pub enum Covariate<'a> {
    /// NaN denotes a missing observation. Infinity is invalid.
    Numeric(&'a [f32]),
    /// Missing categories use `None`; unseen future labels use the fitted target mean
    /// for a univariate target. Multiple targets use deterministic numeric label codes.
    Categorical(&'a [Option<&'a str>]),
}

impl Covariate<'_> {
    fn len(self) -> usize {
        match self {
            Self::Numeric(values) => values.len(),
            Self::Categorical(values) => values.len(),
        }
    }

    fn validate(self, length: usize, name: &str) -> Result<()> {
        if self.len() != length {
            return Err(Error::InvalidInput(format!(
                "{name}: expected {length} steps, received {}",
                self.len()
            )));
        }
        if let Self::Numeric(values) = self {
            validate_numeric(values, name)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Target<'a> {
    name: String,
    values: &'a [f32],
}

#[derive(Debug)]
struct Channel<'a> {
    name: String,
    past: Covariate<'a>,
    future: Option<Covariate<'a>>,
}

/// One aligned multivariate forecast problem, with a positive horizon and at least
/// one target. Values are borrowed; names and channel descriptions are owned.
#[derive(Debug)]
pub struct ForecastRequest<'a> {
    targets: Vec<Target<'a>>,
    past: Vec<Channel<'a>>,
    known: Vec<Channel<'a>>,
    horizon: usize,
    grid: Option<TimeGrid>,
}

impl<'a> ForecastRequest<'a> {
    pub fn new(name: impl Into<String>, values: &'a [f32], horizon: usize) -> Result<Self> {
        if values.is_empty() || horizon == 0 || values.len().checked_add(horizon).is_none() {
            return Err(Error::InvalidInput(
                "history and horizon must be positive and their sum must fit usize".into(),
            ));
        }
        validate_numeric(values, "target")?;
        let mut request = Self {
            targets: Vec::new(),
            past: Vec::new(),
            known: Vec::new(),
            horizon,
            grid: None,
        };
        let name = name.into();
        request.check_name(&name)?;
        request.targets.push(Target { name, values });
        Ok(request)
    }

    /// Add another target to forecast jointly, such as another instrument or volume.
    pub fn with_target(mut self, name: impl Into<String>, values: &'a [f32]) -> Result<Self> {
        let name = name.into();
        self.check_name(&name)?;
        Covariate::Numeric(values).validate(self.context_len(), &name)?;
        self.targets.push(Target { name, values });
        Ok(self)
    }

    /// Add observations available only up to the forecast origin.
    pub fn with_past_covariate(
        mut self,
        name: impl Into<String>,
        values: Covariate<'a>,
    ) -> Result<Self> {
        let name = name.into();
        self.check_name(&name)?;
        values.validate(self.context_len(), &name)?;
        self.past.push(Channel {
            name,
            past: values,
            future: None,
        });
        Ok(self)
    }

    /// Add history and future values known at the forecast origin, for example a
    /// session calendar. Both halves must use the same numeric/categorical kind.
    pub fn with_known_covariate(
        mut self,
        name: impl Into<String>,
        past: Covariate<'a>,
        future: Covariate<'a>,
    ) -> Result<Self> {
        let name = name.into();
        self.check_name(&name)?;
        past.validate(self.context_len(), &name)?;
        future.validate(self.horizon, &name)?;
        if !matches!(
            (past, future),
            (Covariate::Numeric(_), Covariate::Numeric(_))
                | (Covariate::Categorical(_), Covariate::Categorical(_))
        ) {
            return Err(Error::InvalidInput(format!(
                "{name}: past and future covariate kinds differ"
            )));
        }
        self.known.push(Channel {
            name,
            past,
            future: Some(future),
        });
        Ok(self)
    }

    /// Attach a regular UTC millisecond grid. The future grid continues this interval;
    /// applications with exchange-session time must supply their own timestamp mapping.
    pub fn with_time_grid(mut self, grid: TimeGrid) -> Result<Self> {
        if grid.len() != self.context_len() {
            return Err(Error::InvalidInput(
                "time grid must cover the target history".into(),
            ));
        }
        grid.following(self.horizon)?;
        self.grid = Some(grid);
        Ok(self)
    }

    #[must_use]
    pub fn context_len(&self) -> usize {
        self.targets[0].values.len()
    }
    #[must_use]
    pub fn horizon(&self) -> usize {
        self.horizon
    }
    pub fn target_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.targets.iter().map(|target| target.name.as_str())
    }
    #[must_use]
    pub fn time_grid(&self) -> Option<TimeGrid> {
        self.grid
    }

    fn check_name(&self, name: &str) -> Result<()> {
        if name.trim().is_empty() {
            return Err(Error::InvalidInput(
                "channel names must not be empty".into(),
            ));
        }
        if self
            .target_names()
            .chain(self.past.iter().chain(&self.known).map(|c| c.name.as_str()))
            .any(|n| n == name)
        {
            return Err(Error::InvalidInput(format!(
                "duplicate channel name {name:?}"
            )));
        }
        Ok(())
    }

    pub(crate) fn prepare(&self, max_context: usize) -> Result<Prepared> {
        let t = self.context_len();
        let v = self.targets.len();
        let rows = v
            .checked_add(self.past.len())
            .and_then(|r| r.checked_add(self.known.len()));
        // The backend pads context/horizon to patches and uses products of these dimensions.
        let length = t.checked_add(self.horizon).and_then(|n| n.checked_add(64));
        let fits = rows
            .zip(length)
            .and_then(|(r, n)| r.checked_mul(n))
            .and_then(|n| n.checked_mul(21))
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .is_some_and(|bytes| bytes <= isize::MAX as usize);
        if !fits {
            return Err(Error::InvalidInput(
                "forecast dimensions exceed the addressable buffer size".into(),
            ));
        }
        for target in &self.targets {
            if !target.values[t.saturating_sub(max_context)..]
                .iter()
                .any(|x| x.is_finite())
            {
                return Err(Error::InvalidInput(format!(
                    "{}: retained target history contains no observations",
                    target.name
                )));
            }
        }
        let targets: Vec<_> = self.targets.iter().map(|target| target.values).collect();
        Ok(Prepared {
            context: targets.iter().flat_map(|row| row.iter().copied()).collect(),
            variates: v,
            past: encode_channels(&self.past, &targets, max_context),
            past_channels: self.past.len(),
            known: encode_channels(&self.known, &targets, max_context),
            known_channels: self.known.len(),
        })
    }
}

fn validate_numeric(values: &[f32], name: &str) -> Result<()> {
    if let Some(index) = values.iter().position(|value| value.is_infinite()) {
        return Err(Error::InvalidInput(format!(
            "{name}: infinite observation at step {index}; use NaN for missing values"
        )));
    }
    Ok(())
}

fn encode_channels(channels: &[Channel<'_>], targets: &[&[f32]], max_context: usize) -> Vec<f32> {
    let capacity = channels
        .iter()
        .map(|channel| channel.past.len() + channel.future.map_or(0, Covariate::len))
        .sum();
    let mut encoded = Vec::with_capacity(capacity);
    // Match the reference row order: numeric channels precede categorical channels.
    for channel in channels
        .iter()
        .filter(|c| matches!(c.past, Covariate::Numeric(_)))
        .chain(
            channels
                .iter()
                .filter(|c| matches!(c.past, Covariate::Categorical(_))),
        )
    {
        match (channel.past, channel.future) {
            (Covariate::Numeric(past), None) => encoded.extend_from_slice(past),
            (Covariate::Numeric(past), Some(Covariate::Numeric(future))) => {
                encoded.extend_from_slice(past);
                encoded.extend_from_slice(future);
            }
            (Covariate::Categorical(past), None) => {
                encoded.extend(categorical::encode(past, &[], targets, max_context))
            }
            (Covariate::Categorical(past), Some(Covariate::Categorical(future))) => {
                encoded.extend(categorical::encode(past, future, targets, max_context))
            }
            _ => unreachable!("request constructors enforce matching covariate kinds"),
        }
    }
    encoded
}

pub(crate) struct Prepared {
    pub context: Vec<f32>,
    pub variates: usize,
    pub past: Vec<f32>,
    pub past_channels: usize,
    pub known: Vec<f32>,
    pub known_channels: usize,
}

impl Prepared {
    pub fn input(&self) -> candle_timestral::ForecastInput<'_> {
        let mut input = candle_timestral::ForecastInput::new(&self.context, 1, self.variates);
        if self.past_channels > 0 {
            input = input.with_past_covariates(&self.past, self.past_channels);
        }
        if self.known_channels > 0 {
            input = input.with_known_covariates(&self.known, self.known_channels);
        }
        input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_external_inputs() {
        assert!(ForecastRequest::new("close", &[], 1).is_err());
        assert!(ForecastRequest::new("close", &[1.0], 0).is_err());
        assert!(ForecastRequest::new("close", &[f32::INFINITY], 1).is_err());
        assert!(ForecastRequest::new("close", &[1.0], usize::MAX).is_err());
        assert!(ForecastRequest::new("", &[1.0], 1).is_err());
        assert!(
            ForecastRequest::new("close", &[1.0], 1)
                .unwrap()
                .with_target("close", &[2.0])
                .is_err()
        );
        assert!(
            ForecastRequest::new("close", &[1.0], 1)
                .unwrap()
                .with_past_covariate("volume", Covariate::Numeric(&[1.0, 2.0]))
                .is_err()
        );
        assert!(
            ForecastRequest::new("close", &[f32::NAN], 1)
                .unwrap()
                .prepare(32)
                .is_err()
        );
    }

    #[test]
    fn rejects_unaddressable_forecast_buffers() {
        let request = ForecastRequest::new("close", &[1.0], isize::MAX as usize / 21).unwrap();
        assert!(request.prepare(32).is_err());
    }

    #[test]
    fn packs_multivariate_targets_and_numeric_covariates_in_row_order() {
        let request = ForecastRequest::new("close", &[1.0, f32::NAN], 1)
            .unwrap()
            .with_target("volume", &[3.0, 4.0])
            .unwrap()
            .with_known_covariate(
                "session",
                Covariate::Numeric(&[0.0, 1.0]),
                Covariate::Numeric(&[1.0]),
            )
            .unwrap();
        let prepared = request.prepare(32).unwrap();
        assert_eq!(prepared.variates, 2);
        assert!(prepared.context[1].is_nan());
        assert_eq!(&prepared.context[2..], &[3.0, 4.0]);
        assert_eq!(prepared.known, [0.0, 1.0, 1.0]);
    }
}
