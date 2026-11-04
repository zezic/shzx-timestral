use std::path::{Path, PathBuf};

use candle_core::Device;
use candle_timestral::{ForecastOptions, TimestralForecaster};

use crate::{Error, Forecast, ForecastRequest, Model, Provenance, Result};

/// CPU always works without accelerator features. Accelerator selection is explicit;
/// an unavailable backend returns an error instead of silently changing devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    #[default]
    Cpu,
    Metal {
        ordinal: usize,
    },
    Cuda {
        ordinal: usize,
    },
}

impl Backend {
    pub(crate) fn device(self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            #[cfg(feature = "metal")]
            Self::Metal { ordinal } => {
                // Candle 0.11 indexes the device list unchecked. Validate an external
                // device ordinal, including a machine with no visible Metal devices.
                let available_devices = candle_metal_kernels::metal::Device::all().len();
                if ordinal >= available_devices {
                    return Err(Error::DeviceUnavailable {
                        backend: self,
                        available_devices,
                    });
                }
                Ok(Device::new_metal(ordinal)?)
            }
            #[cfg(feature = "cuda")]
            Self::Cuda { ordinal } => Ok(Device::new_cuda(ordinal)?),
            #[allow(unreachable_patterns)]
            backend => Err(Error::BackendUnavailable(backend)),
        }
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;

    #[test]
    fn invalid_metal_ordinal_returns_an_error() {
        assert!(matches!(
            Backend::Metal {
                ordinal: usize::MAX
            }
            .device(),
            Err(Error::DeviceUnavailable { .. })
        ));
    }
}

/// Averaging performs two model passes, reflecting the negated pass back to the
/// original scale. Direct inference reduces latency but changes forecast values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Symmetry {
    #[default]
    Average,
    Direct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Nonnegative {
    /// Clip at zero only for targets with entirely nonnegative observed history.
    #[default]
    Clip,
    Unconstrained,
}

/// Reference inference defaults. Normalization and missing-value handling remain in
/// the native backend; these options control reference postprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Settings {
    pub symmetry: Symmetry,
    pub nonnegative: Nonnegative,
}

impl Settings {
    pub(crate) fn options(self, horizon: usize) -> ForecastOptions {
        ForecastOptions::new(horizon)
            .with_symmetric_averaging(self.symmetry == Symmetry::Average)
            .with_clip_nonnegative(self.nonnegative == Nonnegative::Clip)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// Local weights may be caller-supplied. The library cannot infer their revision.
    Directory(PathBuf),
    HuggingFace(Model),
}

/// A loaded float32 model. Construct once and reuse it on a worker thread. The default
/// configuration supports every released Timestral size and a trained context of
/// 32,768 time steps. Longer horizons use median-fed block autoregression.
#[derive(Debug)]
pub struct Forecaster {
    native: TimestralForecaster,
    backend: Backend,
    source: ModelSource,
    settings: Settings,
}

impl Forecaster {
    /// Load `config.json` and `model.safetensors` from a local directory, entirely offline.
    pub fn from_dir(directory: impl AsRef<Path>, backend: Backend) -> Result<Self> {
        let directory = directory.as_ref();
        let native = TimestralForecaster::from_dir(directory, &backend.device()?)?;
        Ok(Self {
            native,
            backend,
            source: ModelSource::Directory(directory.to_path_buf()),
            settings: Settings::default(),
        })
    }

    /// Download and load a pinned official revision, using the normal HF credentials/cache.
    #[cfg(feature = "hub")]
    pub fn from_hub(model: Model, backend: Backend) -> Result<Self> {
        // Resolve the device first so an invalid backend does not trigger a download.
        let device = backend.device()?;
        Ok(Self {
            native: TimestralForecaster::from_dir(model.download()?, &device)?,
            backend,
            source: ModelSource::HuggingFace(model),
            settings: Settings::default(),
        })
    }

    #[must_use]
    pub fn with_settings(mut self, settings: Settings) -> Self {
        self.settings = settings;
        self
    }

    /// Keep the most recent history up to this cap. Whole patches avoid ambiguity in
    /// the backend's autoregressive truncation. Must be within the trained context.
    pub fn with_max_context(mut self, steps: usize) -> Result<Self> {
        let config = self.native.config();
        if steps == 0 || steps > config.context_length || !steps.is_multiple_of(config.patch_size) {
            return Err(Error::InvalidInput(format!(
                "context cap must be a positive multiple of {} no larger than {}",
                config.patch_size, config.context_length
            )));
        }
        self.native = self.native.with_max_context(steps);
        Ok(self)
    }

    pub fn forecast(&self, request: &ForecastRequest<'_>) -> Result<Forecast> {
        let input = request.prepare(self.native.max_context())?;
        let raw = self
            .native
            .forecast(&input.input(), &self.settings.options(request.horizon()))?;
        Forecast::new(
            raw,
            request,
            Provenance {
                source: self.source.clone(),
                backend: self.backend,
                settings: self.settings,
                max_context: self.native.max_context(),
                context_steps: request.context_len(),
            },
        )
    }

    #[must_use]
    pub fn backend(&self) -> Backend {
        self.backend
    }
    #[must_use]
    pub fn source(&self) -> &ModelSource {
        &self.source
    }
    #[must_use]
    pub fn settings(&self) -> Settings {
        self.settings
    }
    #[must_use]
    pub fn max_context(&self) -> usize {
        self.native.max_context()
    }

    /// Access batched forecasting and raw single-block inference. Inputs passed here
    /// follow the native crate's contract rather than this crate's named request checks.
    #[must_use]
    pub fn as_native(&self) -> &TimestralForecaster {
        &self.native
    }
}
