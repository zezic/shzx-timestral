# Native Timestral research

Checked 4 November 2026 against upstream model cards and source. Architecture facts
below describe the released models; market mappings and integration decisions are
our engineering recommendations. Runtime measurements belong in the README after
actual execution.

## Released models

| Checkpoint | Transformer layers | Embedding width | FP32 checkpoint, approximately |
|---|---:|---:|---:|
| [Timestral-5M](https://huggingface.co/mistralai/Timestral-5M) | 8 | 192 | 21 MB |
| [Timestral-28M](https://huggingface.co/mistralai/Timestral-28M) | 12 | 384 | 111 MB |
| [Timestral-123M](https://huggingface.co/mistralai/Timestral-123M) | 20 | 640 | 490 MB |
| [Timestral-209M](https://huggingface.co/mistralai/Timestral-209M) | 24 | 768 | 837 MB |

All four use 32-step patches, up to 32,768 context steps, and up to 1,024 future
steps per decode block. The checkpoint consists of `config.json` and
`model.safetensors`; weights are FP32. Runtime memory also includes tensors and
attention buffers. Checkpoint sizes come from the
[official repository](https://github.com/mistralai/timestral).

## Inputs and outputs

Targets have shape `(batch, variates, context_steps)` and variates are forecast
jointly. Past covariates cover context; known-future covariates cover context plus
horizon. Covariates condition targets without being forecast. NaN or an
observation mask marks missing targets. The API accepts sampled arrays;
timestamps, calendars, units and bar formation belong to the caller.
[Official forecaster source](https://github.com/mistralai/timestral/blob/main/src/timestral/forecaster.py)

Output shape is `(batch, target_variates, horizon_steps, quantiles)`. The 21 levels
are `0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99`. The median is
the 0.50 channel. Longer horizons use repeated blocks with median feedback.
[209M model contract](https://huggingface.co/mistralai/Timestral-209M)

The head builds non-crossing quantiles around a free median through cumulative
positive increments. Its public prediction output supplies per-step,
per-variate quantiles; it does not supply a mean, sampled joint future paths,
trade actions, or constrained OHLC candles. A quantile fan therefore fits the
terminal better than drawing its marginal bounds as future candle highs/lows.
Jointly forecasting open/high/low/close does not enforce their price inequalities.
These output limitations follow from inspecting the
[model and quantile head](https://github.com/mistralai/timestral/blob/main/src/timestral/model.py).

## Preprocessing that must survive native inference

Context scaling uses causal per-patch mean and sample standard deviation computed
in float64, then an `asinh` transform. Missing positions retain their missingness.
Forecasts invert that transform; later patches use advancing statistics incorporating
earlier predicted medians. Known-future numeric covariates use whole-row instance
statistics instead. Observed binary 0/1 indicator rows keep location zero and
scale one and skip `asinh`.
[Official scaler](https://github.com/mistralai/timestral/blob/main/src/timestral/scaler.py)

Defaults average the forecast with its reflected, negated-input forecast;
covariates are negated too. Nonnegative clipping applies when all observed target
values are nonnegative. Disabling symmetry removes one pass. Left padding aligns
patches and old context is trimmed. Block feedback uses actual known-future
covariates and marks past-only future rows missing.
[Forecaster implementation](https://github.com/mistralai/timestral/blob/main/src/timestral/forecaster.py)

### Categorical encoding

For a univariate target, forward-fill the original target using its observation
mask. Only positions after its first observation and within the selected context
window contribute to encoding statistics. Each category becomes
`(series_mean + category_sum) / (1 + category_count)` using float64 statistics and
finite target values. A missing category has its own bucket; unseen future
categories map to the series mean. Missing category entries outside the valid
target window remain missing. With multiple target variates the reference passes
numeric category codes through instead. Upstream keeps codes in float64 before
encoding so distinct identifiers larger than `2^24` remain distinct.
[Encoding algorithm](https://github.com/mistralai/timestral/blob/main/src/timestral/covariate_encoding.py),
[window and padding rules](https://github.com/mistralai/timestral/blob/main/src/timestral/forecaster.py)

## Market data mapping

The model accepts numeric sequences rather than market-specific schemas. The
following are possible representations, not evidence that adding a channel
improves accuracy.

| Market information | Possible role | Preparation requirement |
|---|---|---|
| Prices, returns, volume, realized volatility | Selected targets or numeric past covariates | Choose explicit units, transformations and a shared grid |
| Multiple instruments, OHLCV | Joint target variates or past covariates | Align observations and missing values; OHLC constraints are not guaranteed |
| Trades, aggressor flow, trade counts | Numeric past covariates | Aggregate timestamped observations with a documented rule |
| Spread, depth, imbalance, book pressure | Numeric past covariates | Record and sample a history; one snapshot is insufficient |
| Technical indicators, fundamentals, numeric sentiment | Numeric past covariates | Use values available at the recorded forecast cutoff |
| Session flags, weekday, holiday or event schedule | Known numeric or categorical covariates | Include only information knowable at the cutoff |
| Future prices, volume, event outcomes | Targets to predict | Realized future values cannot serve as known covariates |

Raw news and exchange objects need feature extraction outside this library.
Corporate-action policy, missing sessions, incomplete bars, quote currency, lots
and price precision remain data-adapter decisions. Keep the exact provider values
for accounting and transform only inference copies to FP32. The existing terminal
[data adapter](../../shizex/docs/native/data-adapter.md) already defines timestamp,
lot and completed-bar provenance. The terminal's earlier
[Timestral research](../../shizex/docs/timestral_integration_research.md) specifies
a worker boundary, frozen input cutoff, quantile fan and rolling-origin evaluation.
Its proposed Python runtime is superseded here by native Candle inference.

## Reuse, speed and verification

`candle-timestral` provides the native network, causal scaling, decode loop,
numeric covariates and clampsym behavior without DataFusion. It already processes
time attention in bounded blocks and variate attention per series. Its documented
gap is categorical encoding, which belongs in our input preparation. Reusing the
released crate avoids maintaining another transformer port.
[Native crate source](https://github.com/nfnt/datafusion-forecast/tree/main/crates/candle-timestral)

The upstream repository includes committed reference-output fixtures
`golden_{5m,28m,123m,209m}.safetensors`; its documented parity threshold is relative
error below `1e-4`. They can be consumed by Rust without running Python. The
documented 5M checkpoint revision is
`bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f`. Fixture generation is a separate upstream
Python operation and is not required to run existing fixtures.
[Upstream parity instructions](https://github.com/nfnt/datafusion-forecast/blob/main/crates/candle-timestral/README.md)

CPU inference is Candle's portable baseline. Apple Accelerate accelerates CPU
matrix multiplication; Metal targets Apple GPUs. CUDA and cuDNN require their
respective native toolchains. The native Timestral crate forwards these backend
features. Linux and Windows CPU operation is a supported implementation path,
but needs execution on those platforms before it can be described as locally
verified.
[Backend feature definitions](https://github.com/nfnt/datafusion-forecast/blob/main/crates/candle-timestral/Cargo.toml),
[Candle backends](https://github.com/huggingface/candle)

Start with 5M, a retained model instance, moderate history and a short forecast.
Benchmark representative target/covariate counts and compare CPU, Accelerate and
Metal on the actual Mac: kernel dispatch and transfers can outweigh a small
model's GPU compute savings. Keep loading and synchronous inference on a worker
outside the UI render loop. Record model revision, options, cutoff and timing.
Neither benchmark rankings nor numerical parity establish market predictive
quality; evaluate rolling origins and quantile coverage against a persistence
baseline before relying on richer feature sets.

## Licensing

The reference implementation and its native port use Apache 2.0. Model weights
have a separate Modified MIT license, including attribution/permission retention
and a restriction when company or employer consolidated monthly revenue exceeds
USD 20 million for the preceding month. Such use requires separate commercial
permission. Do not label the weights ordinary MIT or bundle them as this crate's
code license.
[Reference code license](https://github.com/mistralai/timestral/blob/main/LICENSE),
[native port license](https://github.com/nfnt/datafusion-forecast/blob/main/LICENSE),
[weight license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE)
