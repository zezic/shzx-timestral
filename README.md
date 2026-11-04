# shzx-timestral

An application-facing Rust API for Timestral forecasts, built on
[`candle-timestral`](https://github.com/nfnt/datafusion-forecast/tree/main/crates/candle-timestral),
the native inference engine from
[`nfnt/datafusion-forecast`](https://github.com/nfnt/datafusion-forecast). That
project also provides a separate `datafusion-forecast` crate for SQL forecasting
through Apache DataFusion.

This library reuses the upstream model implementation and adds categorical
covariate encoding, validated named inputs, explicit time grids, pinned/offline
model loading, forecast provenance and typed quantile access for direct Rust
application integration. No Python runtime or DataFusion dependency is required.
Requires Rust 1.95 or later.

## Quickstart

After publishing the repository, add this Git dependency to your application's
`Cargo.toml` for portable CPU inference:

```toml
[dependencies]
shzx-timestral = { git = "https://github.com/zezic/shzx-timerstal.git" }
```

For model downloads and Apple GPU/CPU acceleration on macOS:

```toml
[dependencies]
shzx-timestral = { git = "https://github.com/zezic/shzx-timerstal.git", features = ["hub", "metal", "accelerate"] }
```

The repository is named `shzx-timerstal`; the Cargo package remains
`shzx-timestral`, with Rust imports under `shzx_timestral`. The Git dependency
uses the repository's default branch. Commit the application's `Cargo.lock` to
retain its resolved revision, or add `rev = "<full-commit-sha>"` after publishing
to select a specific commit.

For local development alongside the sibling terminal, use a path dependency:

```toml
[dependencies]
shzx-timestral = { path = "../shzx-timestral" }
```

Load a local directory containing the released `config.json` and
`model.safetensors`. This works offline with the default CPU feature set:

```rust
use shzx_timestral::{Backend, ForecastRequest, Forecaster, Quantile};

fn main() -> shzx_timestral::Result<()> {
    let close = [100.0, 101.0, f32::NAN, 102.0];
    let request = ForecastRequest::new("close", &close, 32)?;
    let model = Forecaster::from_dir("models/Timestral-5M", Backend::Cpu)?;
    let forecast = model.forecast(&request)?;
    let close = forecast.target("close").expect("requested target");
    let median: Vec<_> = close.median().collect();
    let p90: Vec<_> = close.quantile(Quantile::P90).collect();
    let band: Vec<_> = close.interval(Quantile::P10, Quantile::P90)?.collect();
    println!("median={median:?}, p90={p90:?}, band={band:?}");
    Ok(())
}
```

Reuse the loaded forecaster on a worker. Loading and inference are synchronous.

## Pinned Hugging Face checkpoints

Enable `features = ["hub"]` on the dependency. The four `Model` variants pin
immutable official revisions rather than following `main`:

```rust
use shzx_timestral::{Backend, Forecaster, Model};

let model = Model::Timestral5M;
let forecaster = Forecaster::from_hub(model, Backend::Cpu)?;
// Or download separately, then load the returned snapshot directory:
let directory = model.download()?;
let offline = Forecaster::from_dir(model.cached()?, Backend::Cpu)?;
```

The excerpt uses `?` inside a function returning `shzx_timestral::Result<_>`.
The loader uses Hugging Face's token file or `HF_TOKEN`, honors `HF_HOME` and
`HF_HUB_CACHE`, and retrieves the config, weights and weight license.
`Model::cached()` permits no network access and fails if a required pinned file
is missing. Local-directory loads need no Hub feature or credentials.
Downloaded models and local captures are explicitly excluded from the crate package.

| Model variant | FP32 weights, approximately | Local inference validation |
|---|---:|---|
| `Timestral5M` | 21 MB | CPU/Metal reference parity and market snapshot |
| `Timestral28M` | 111 MB | CPU/Metal reference parity |
| `Timestral123M` | 490 MB | CPU/Metal reference parity, full context and market snapshot |
| `Timestral209M` | 837 MB | CPU/Metal reference parity, full context and market snapshot |

All four share 32-step patches, up to 32,768 retained historical steps and up to
1,024 forecast steps per block. Longer requests use successive blocks with median
feedback. See [research](docs/research.md) for sources and architecture details.

## Input and result contract

`ForecastRequest::new(name, history, horizon)` creates one series. Add joint
targets with `with_target`, history-only channels with `with_past_covariate`, and
context/future pairs with `with_known_covariate`. Channels use
`Covariate::Numeric(&[f32])` or `Covariate::Categorical(&[Option<&str>])`.
Names must be nonempty and unique. Target histories and past channels align;
known-future channels additionally cover the entire horizon.

Use NaN for missing numbers and `None` for missing category labels. Infinities,
empty histories, zero horizons and retained target rows without observations are
errors. Univariate categories use reference target encoding; joint multivariate
targets use deterministic lexical label codes. `TimeGrid` aligns strictly
increasing UTC millisecond observations onto a regular grid, preserves missing
slots and optionally attaches future timestamps.

Every target returns all 21 quantiles through `Quantile::ALL`:
`P01, P05, P10, P15, P20, P25, P30, P35, P40, P45, P50, P55, P60, P65, P70,
P75, P80, P85, P90, P95, P99`. Access a channel as an iterator, all probabilities
at one step with `step`, or flat output through `as_native()`.

Reference defaults use `Symmetry::Average` and `Nonnegative::Clip`.
`with_settings(Settings { symmetry: Symmetry::Direct, nonnegative:
Nonnegative::Unconstrained })` changes those choices. `with_max_context(steps)`
accepts positive 32-step multiples through 32,768. Results retain model source,
backend, settings and context lengths in `provenance()`.

## Run the examples

```sh
# Model, backend, context, horizon, timed repeats, scenario
cargo run --release --features hub --example forecast -- 5m cpu 512 96 3 uni
cargo run --release --features hub --example forecast -- 28m cpu 512 96 3 market

# Apple GPU; requires Metal build tools
cargo run --release --features hub,metal --example forecast -- 5m metal 512 96 3 multi

# Apple Accelerate CPU matrix multiplication
cargo run --release --features hub,accelerate --example forecast -- 5m cpu 512 96 3 uni

# Existing Shizex market snapshot; no provider connection or Hub access
cargo run --release --example market_snapshot -- \
  models/Timestral-5M ../shizex/captures/market-sandbox.json cpu 32
```

`forecast` arguments are
`[5m|28m|123m|209m|checkpoint_dir] [cpu|metal|cuda] [context=512]
[horizon=96] [repeats=3] [uni|market|multi]`; it emits timing JSON for synthetic
inputs. `market_snapshot` takes
`CHECKPOINT_DIR SNAPSHOT_JSON [cpu|metal] [HORIZON=32]` and emits forecast JSON.

CPU is the portable baseline for Linux and Windows; neither platform was executed
locally. CUDA is opt-in and needs the NVIDIA toolchain. Accelerator selection is
explicit and errors when unavailable. Current Apple backend, Hub and performance
evidence is recorded in [validation](docs/validation.md); feature availability
alone does not establish a speedup.

## Verification and boundaries

```sh
cargo test
TIMESTRAL_5M_DIR="$PWD/models/Timestral-5M" \
TIMESTRAL_28M_DIR="$PWD/models/Timestral-28M" \
TIMESTRAL_123M_DIR="$PWD/models/Timestral-123M" \
TIMESTRAL_209M_DIR="$PWD/models/Timestral-209M" \
cargo test --release --test parity -- --ignored --test-threads=1
```

Eleven default unit tests (twelve with Metal) and all eight enabled parity tests
on each of CPU and Metal passed locally. The
reference suite covers all four model sizes, missing multivariate data, covariates,
raw blocks and 1,100-step horizons at normalized maximum error below `1e-4`.
Fixtures and exact revision provenance are in [tests/fixtures](tests/fixtures/README.md).
The manual [reference workflow](.github/workflows/reference.yml) runs the full
model family on Linux CPU; regular CI keeps the smaller 5M reference check.

Inputs are numeric sequences, not raw text, news or exchange messages; extract
features first. Output supplies marginal quantiles, not a mean, joint sampled
paths, constrained OHLC candles or trading actions. Preserve exact provider
prices and quantities outside inference copies: the model uses FP32.
Numerical parity establishes implementation correspondence, not market forecast
quality. See the [Shizex handoff](docs/shizex-integration.md) for data preparation
and worker integration.

Code is Apache-2.0. Model weights have a separate
[Modified MIT license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE),
including attribution retention and a restriction when company/employer global
consolidated revenue exceeds USD 20 million in the preceding month. That condition
requires separate commercial permission; the weights are not ordinary MIT.
