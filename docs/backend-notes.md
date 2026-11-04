# Native Timestral backend review

Checked 4 November 2026. Source inspection is distinct from verification on a
particular operating system. No Python runtime is needed by the proposed backend.

## Reuse decision

`nfnt/datafusion-forecast` contains a standalone `candle-timestral` crate; the
SQL/DataFusion layer is a separate crate. Its 0.1.0 manifest uses Candle 0.11 and
Rust 1.95, with Apache-2.0 licensing. Reusing this backend avoids maintaining a
second implementation of the transformer, scaling, masks and decode loop.
[Workspace manifest](https://github.com/nfnt/datafusion-forecast/blob/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/Cargo.toml),
[crate manifest](https://github.com/nfnt/datafusion-forecast/blob/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/crates/candle-timestral/Cargo.toml).

The API supports batched multivariate targets, observation masks, numeric past
covariates, numeric known-future covariates, 21 quantiles and median feedback for
horizons longer than one block. Categorical target encoding is explicitly absent.
Application code should encode categories deliberately rather than imply that raw
integer category IDs reproduce the reference categorical preprocessing.
[Backend API and supported features](https://github.com/nfnt/datafusion-forecast/blob/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/crates/candle-timestral/README.md).

Checkpoint loading requires `config.json` and `model.safetensors`. It loads float32
tensors, rejects unused checkpoint weights, and relies on Candle to reject missing
or misshapen tensors. The backend's `from_hub` uses the default Hub revision; a
reproducible application should resolve a pinned revision and call `from_dir`.
[Loader](https://github.com/nfnt/datafusion-forecast/blob/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/crates/candle-timestral/src/loading.rs).

## Platform and performance implications

Candle's default feature set is empty, with CPU operations provided without a
CUDA dependency. Its optional backends include Metal, CUDA, Apple Accelerate and
Intel MKL. `candle-timestral` forwards those features. CPU is the portable baseline
for Linux and Windows; cross-platform runtime behavior still requires execution
on those systems. Metal and Accelerate require Apple platforms. CUDA requires a
working toolkit and driver rather than merely enabling a Cargo feature.
[Candle feature manifest](https://github.com/huggingface/candle/blob/main/candle-core/Cargo.toml),
[installation prerequisites](https://huggingface.github.io/candle/guide/installation.html).

Backend source inspection shows CPU-side preprocessing, masks, rotary tables and
postprocessing around the device-side network. Every decode block copies head
outputs back to the host. Symmetric averaging performs two complete inference
passes. Consequently, small models or short contexts may favor Accelerate CPU
over Metal; this is a performance hypothesis to measure, not a benchmark result.
The upstream implementation chunks attention scores to limit temporary memory.
[Network execution](https://github.com/nfnt/datafusion-forecast/blob/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/crates/candle-timestral/src/model.rs),
[attention implementation](https://github.com/nfnt/datafusion-forecast/blob/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/crates/candle-timestral/src/layers.rs),
[decode loop](https://github.com/nfnt/datafusion-forecast/blob/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/crates/candle-timestral/src/forecaster.rs).

## Existing numerical fixtures

The source repository commits safetensors fixtures for all four model sizes. Each
contains reference outputs for univariate inference, multivariate missing data,
numeric covariates, multiple decode blocks, capped contexts and raw network
blocks. These let Rust tests compare with recorded reference outputs without
installing or executing Python. The upstream package excludes those fixtures
from its published crate; consumers must obtain them from the source repository.
[Golden tests](https://github.com/nfnt/datafusion-forecast/blob/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/crates/candle-timestral/tests/golden.rs),
[fixture provenance generator](https://github.com/nfnt/datafusion-forecast/blob/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/crates/candle-timestral/scripts/make_golden.py).

The pinned model revisions associated with those fixtures are:

| Model | Revision |
| --- | --- |
| 5M | `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f` |
| 28M | `49b26409ac87b5935539a5d8ffe7f397b720db47` |
| 123M | `5a147287addda85f6acd8da3fb8858fe9af7e8f7` |
| 209M | `9874d252ab0aae16f7f7c999f98f98d6ad7d8365` |

[Upstream pinned golden workflow](https://github.com/nfnt/datafusion-forecast/blob/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/.github/workflows/golden.yml).

## Boundary review and Shizex integration

The backend accepts primitive dimensions and performs unchecked dimension
multiplications. The library boundary should validate dimensions and checked
arithmetic before calling it. It should also reject infinite observations:
covariate assembly uses NaN, rather than all nonfinite values, as missingness.
These observations concern malformed external inputs, not evidence of a failure
on valid released checkpoints. The upstream parity comparison helper also needs
an explicit finite-value assertion if copied, because floating-point NaNs do not
compare greater than its running maximum error.
[Forecast input validation](https://github.com/nfnt/datafusion-forecast/blob/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/crates/candle-timestral/src/forecaster.rs),
[comparison helper](https://github.com/nfnt/datafusion-forecast/blob/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/crates/candle-timestral/tests/golden.rs).

The sibling terminal currently stores exact provider quotations and UTC bar
timestamps in `src/provider.rs`; it deliberately preserves absent intervals.
Its render-facing `src/data.rs` holds numeric chart bars and synthetic forecast
data. Integration should convert closed provider bars onto an explicit sampling
grid in the application layer, preserve missing observations, and call this
library on a worker outside the rendering thread. Keep provider credentials and
transport dependencies out of the model library. Persist the model revision,
input cutoff, time grid and inference settings with each forecast snapshot.
This is an integration recommendation based on the sibling's local
`docs/native/data-adapter.md` and `docs/timestral_integration_research.md`, not an
upstream model contract.
