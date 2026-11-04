use crate::{Backend, Error, ForecastRequest, ModelSource, Result, Settings, TimeGrid};

/// The 21 probabilities emitted by every released Timestral checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantile {
    P01,
    P05,
    P10,
    P15,
    P20,
    P25,
    P30,
    P35,
    P40,
    P45,
    P50,
    P55,
    P60,
    P65,
    P70,
    P75,
    P80,
    P85,
    P90,
    P95,
    P99,
}

impl Quantile {
    pub const ALL: [Self; 21] = [
        Self::P01,
        Self::P05,
        Self::P10,
        Self::P15,
        Self::P20,
        Self::P25,
        Self::P30,
        Self::P35,
        Self::P40,
        Self::P45,
        Self::P50,
        Self::P55,
        Self::P60,
        Self::P65,
        Self::P70,
        Self::P75,
        Self::P80,
        Self::P85,
        Self::P90,
        Self::P95,
        Self::P99,
    ];

    #[must_use]
    pub const fn level(self) -> f64 {
        match self {
            Self::P01 => 0.01,
            Self::P05 => 0.05,
            Self::P10 => 0.10,
            Self::P15 => 0.15,
            Self::P20 => 0.20,
            Self::P25 => 0.25,
            Self::P30 => 0.30,
            Self::P35 => 0.35,
            Self::P40 => 0.40,
            Self::P45 => 0.45,
            Self::P50 => 0.50,
            Self::P55 => 0.55,
            Self::P60 => 0.60,
            Self::P65 => 0.65,
            Self::P70 => 0.70,
            Self::P75 => 0.75,
            Self::P80 => 0.80,
            Self::P85 => 0.85,
            Self::P90 => 0.90,
            Self::P95 => 0.95,
            Self::P99 => 0.99,
        }
    }

    fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|&q| q == self)
            .expect("all supported quantiles")
    }
}

/// Inference choices attached to a result. Local checkpoint revision, input data cutoff,
/// sampling/session policy and generation time should also be persisted by the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    pub source: ModelSource,
    pub backend: Backend,
    pub settings: Settings,
    pub max_context: usize,
    pub context_steps: usize,
}

/// Marginal quantiles in the original units of each named target. These are neither
/// samples from a joint future path nor OHLC-consistent future candles.
#[derive(Debug, Clone)]
pub struct Forecast {
    raw: candle_timestral::Forecast,
    targets: Vec<String>,
    provenance: Provenance,
    future_grid: Option<TimeGrid>,
}

impl Forecast {
    pub(crate) fn new(
        raw: candle_timestral::Forecast,
        request: &ForecastRequest<'_>,
        provenance: Provenance,
    ) -> Result<Self> {
        let targets: Vec<_> = request.target_names().map(str::to_owned).collect();
        let expected = targets
            .len()
            .checked_mul(request.horizon())
            .and_then(|n| n.checked_mul(Quantile::ALL.len()));
        if raw.batch != 1
            || raw.variates != targets.len()
            || raw.horizon != request.horizon()
            || expected != Some(raw.quantiles.len())
            || !raw
                .quantile_levels
                .iter()
                .copied()
                .eq(Quantile::ALL.map(Quantile::level))
        {
            return Err(Error::InvalidOutput(
                "checkpoint must emit the released 21-quantile target shape".into(),
            ));
        }
        if raw.quantiles.iter().any(|q| !q.is_finite()) {
            return Err(Error::InvalidOutput(
                "forecast contains nonfinite values".into(),
            ));
        }
        Ok(Self {
            raw,
            targets,
            provenance,
            future_grid: request
                .time_grid()
                .map(|grid| grid.following(request.horizon()))
                .transpose()?,
        })
    }

    #[must_use]
    pub fn horizon(&self) -> usize {
        self.raw.horizon
    }
    #[must_use]
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
    #[must_use]
    pub fn future_grid(&self) -> Option<TimeGrid> {
        self.future_grid
    }
    #[must_use]
    pub fn target(&self, name: &str) -> Option<TargetForecast<'_>> {
        self.targets
            .iter()
            .position(|target| target == name)
            .map(|index| self.at(index))
    }
    #[must_use]
    pub fn targets(&self) -> impl ExactSizeIterator<Item = TargetForecast<'_>> {
        (0..self.targets.len()).map(|index| self.at(index))
    }
    /// Flat `(1, targets, horizon, 21)` output and the model's probability levels.
    #[must_use]
    pub fn as_native(&self) -> &candle_timestral::Forecast {
        &self.raw
    }

    fn at(&self, index: usize) -> TargetForecast<'_> {
        let width = self.horizon() * Quantile::ALL.len();
        TargetForecast {
            name: &self.targets[index],
            values: &self.raw.quantiles[index * width..(index + 1) * width],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TargetForecast<'a> {
    name: &'a str,
    values: &'a [f32],
}

impl<'a> TargetForecast<'a> {
    #[must_use]
    pub fn name(self) -> &'a str {
        self.name
    }
    /// All 21 quantiles at one future step, ordered like [`Quantile::ALL`].
    #[must_use]
    pub fn step(self, index: usize) -> Option<&'a [f32]> {
        self.values
            .as_chunks::<{ Quantile::ALL.len() }>()
            .0
            .get(index)
            .map(|step| step.as_slice())
    }
    #[must_use]
    pub fn quantile(self, quantile: Quantile) -> impl ExactSizeIterator<Item = f32> + 'a {
        let index = quantile.index();
        self.values
            .as_chunks::<{ Quantile::ALL.len() }>()
            .0
            .iter()
            .map(move |step| step[index])
    }
    #[must_use]
    pub fn median(self) -> impl ExactSizeIterator<Item = f32> + 'a {
        self.quantile(Quantile::P50)
    }
    /// Pointwise lower/upper bounds. Bounds may be equal for a degenerate interval.
    pub fn interval(
        self,
        lower: Quantile,
        upper: Quantile,
    ) -> Result<impl ExactSizeIterator<Item = (f32, f32)> + 'a> {
        if lower.level() > upper.level() {
            return Err(Error::InvalidInput(
                "lower quantile exceeds upper quantile".into(),
            ));
        }
        Ok(self.quantile(lower).zip(self.quantile(upper)))
    }
}
