//! Native, probabilistic Timestral forecasts for aligned market observations.
//!
//! Targets are predicted jointly. Past covariates cover the history; known covariates
//! also cover the entire requested horizon. Use NaN for missing numeric observations
//! and `None` for missing categories. All numeric values must otherwise be finite.
//!
//! ```no_run
//! use shzx_timestral::{Backend, ForecastRequest, Forecaster, Quantile};
//!
//! let close = [100.0, 101.0, f32::NAN, 102.0];
//! let request = ForecastRequest::new("close", &close, 32)?;
//! let model = Forecaster::from_dir("models/Timestral-5M", Backend::Cpu)?;
//! let forecast = model.forecast(&request)?;
//! let close = forecast.target("close").expect("requested target");
//! let median: Vec<_> = close.quantile(Quantile::P50).collect();
//! # Ok::<(), shzx_timestral::Error>(())
//! ```
//!
//! Loading and inference are blocking. Keep a forecaster on a worker, reuse it between
//! requests, and publish results to the terminal with their original data cutoff.

#![forbid(unsafe_code)]

mod categorical;
mod checkpoint;
mod forecast;
mod input;
mod runtime;
mod time;

pub use checkpoint::Model;
pub use forecast::{Forecast, Provenance, Quantile, TargetForecast};
pub use input::{Covariate, ForecastRequest};
pub use runtime::{Backend, Forecaster, ModelSource, Nonnegative, Settings, Symmetry};
pub use time::TimeGrid;

/// Low-level batched tensors and the raw block model, for callers that already own
/// alignment, categorical encoding and dimension validation.
pub use candle_timestral as native;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("invalid forecast input: {0}")]
    InvalidInput(String),
    #[error("invalid model output: {0}")]
    InvalidOutput(String),
    #[error("backend {0:?} was not enabled at compile time")]
    BackendUnavailable(Backend),
    #[error(
        "backend {backend:?} requested a device outside the {available_devices} available devices"
    )]
    DeviceUnavailable {
        backend: Backend,
        available_devices: usize,
    },
    #[error(transparent)]
    Inference(#[from] candle_timestral::Error),
    #[error(transparent)]
    Device(#[from] candle_core::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[cfg(feature = "hub")]
    #[error(transparent)]
    Hub(#[from] hf_hub::HFError),
}

pub type Result<T> = std::result::Result<T, Error>;
