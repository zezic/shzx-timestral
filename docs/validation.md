# Local validation

Measured 4 November 2026 on an Apple M4 with 24 GiB RAM, Rust 1.98.1,
`candle-timestral` 0.1.0 and Candle 0.11.1. No Python inference, conversion or
fixture-generation code was executed. The authenticated `hf` CLI downloaded
the initial checkpoints; the library's Rust Hub downloader was also exercised.

## Correctness

- Eleven default unit tests passed; twelve passed with `hub,metal,accelerate`,
  including invalid Metal device selection without a GPU-visible environment.
- All eight recorded-reference tests passed on CPU and all eight passed on Metal.
  The 5M, 28M, 123M and 209M checkpoints were tested through the native backend
  and this library's API, with both portable CPU and Apple Accelerate CPU runs.
  Cases cover raw network blocks, default/direct inference, missing multivariate
  data, past/known numeric covariates, binary indicators, capped context,
  nonnegative clipping and 1,100-step autoregressive forecasts.
- Comparisons require every output to be finite. Maximum absolute error divided
  by `max(1, largest reference magnitude in that case)` remained below `1e-4`.
  The largest printed normalized error was about `3.31e-5`, including named
  median checks. This is a case-normalized bound, not a per-value relative bound.
  The 123M/209M comparisons remained below `1.18e-5` on both devices, without
  changing the tolerance. All four native/wrapper fixtures remain unchanged.
- Every wrapper suite also passed a synthetic regression check at the full
  32,768-step context on portable CPU and Metal, checking shape, provenance and
  every quantile for finite values and monotone ordering. These smoke assertions
  supplement the stored reference outputs.
- Categorical encoding has five focused unit cases and was checked against the
  official source. Mixed numeric/categorical forecasts ran on both devices with
  one and multiple targets. The recorded upstream fixtures cover numeric inputs;
  they do not establish end-to-end reference parity for categorical inputs.
- The Rust Hub path downloaded 5M config, weights and LICENSE at its pinned SHA.
  A subsequent run succeeded without network access using that cached revision.
- Formatting, Clippy with warnings denied, default/accelerator tests, doctests
  and documentation with warnings denied passed.
- Cargo's package list was checked with all four downloaded checkpoints present.
  Explicit manifest exclusions keep weights, model caches and local captures
  outside the crate package, including before the repository has an initial commit.

Pinned revisions and fixture checksums are recorded in
[fixture provenance](../tests/fixtures/README.md). The checkpoint loader rejects
missing, extra and incorrectly shaped tensors. A discovered Candle 0.11 panic
when selecting a nonexistent Metal device is prevented by checking the visible
device count before calling its constructor.

## Warm inference timings

These are release builds with `hub,metal,accelerate`. CPU therefore uses Apple
Accelerate. Each row is the median of seven synchronous calls after one warmup,
with one loaded model, default symmetric averaging and nonnegative clipping.
Model load and warmup are excluded from these medians. No compilation or other
inference benchmark ran concurrently. Other desktop activity was not controlled.

| Model | Context / horizon | Input | Accelerate CPU | Metal |
|---|---|---|---:|---:|
| 5M | 512 / 96 | One target | 4.36 ms | 5.05 ms |
| 5M | 512 / 96 | Market features | 19.78 ms | 15.39 ms |
| 5M | 512 / 96 | Two targets + features | 19.95 ms | 16.11 ms |
| 28M | 512 / 96 | One target | 12.03 ms | 11.45 ms |
| 28M | 512 / 96 | Market features | 60.82 ms | 44.04 ms |
| 28M | 512 / 96 | Two targets + features | 54.48 ms | 43.85 ms |
| 28M | 4,096 / 96 | One target | 46.96 ms | 28.79 ms |
| 123M | 512 / 96 | One target | 43.72 ms | 28.37 ms |
| 123M | 512 / 96 | Market features | 173.68 ms | 124.64 ms |
| 123M | 512 / 96 | Two targets + features | 170.56 ms | 124.97 ms |
| 123M | 4,096 / 96 | One target | 155.38 ms | 87.31 ms |
| 209M | 512 / 96 | One target | 65.66 ms | 41.06 ms |
| 209M | 512 / 96 | Market features | 257.52 ms | 197.40 ms |
| 209M | 512 / 96 | Two targets + features | 266.24 ms | 191.15 ms |
| 209M | 4,096 / 96 | One target | 226.41 ms | 128.69 ms |

Market features mean one price target, past spread and volume, known numeric
phase and known categorical session. The multivariate scenario forecasts volume
as a second target and keeps spread/phase/session as covariates. Inputs are
deterministic synthetic sequences, useful for runtime comparison rather than
predictive evaluation. Raw timing arrays and load/warmup times are in
[benchmarks-m4.json](benchmarks-m4.json).

The first 5M Metal inference in this timing pass took about 7.69 seconds while
compiling kernels. Later warmups varied by shape, from roughly 24 to 212 ms.
Initialize and warm the actual production shape on a worker before publishing
forecasts. For a short single-target 5M request, Accelerate CPU is a good starting
choice on this Mac; larger contexts and more rows benefited from Metal here.
These measurements do not guarantee the same crossover on another machine.

Both larger models also completed a 32,768-step history / 32-step forecast on
CPU and Metal. One timed call after one warmup took 1,160.83 / 1,010.13 ms for
123M and 1,759.91 / 1,405.30 ms for 209M (CPU / Metal). These single-call smoke
measurements exercise full-context chunked attention; they are not seven-call
medians and have no matching full-context reference fixture.

Across the measured larger-model processes, maximum resident set size reported
by `/usr/bin/time -l` was approximately 949 MiB for 123M and 1,610 MiB for 209M.
The JSON records both RSS and macOS peak memory footprint for each run. These
are process measurements, not a GPU allocation budget or a bound for every
possible target/covariate count. Keep model instances loaded between jobs.

Reproduce a row after building once:

```sh
cargo build --release --locked --features hub,metal,accelerate --examples
target/release/examples/forecast models/Timestral-5M cpu 512 96 7 market
target/release/examples/forecast models/Timestral-5M metal 512 96 7 market
```

## Terminal market snapshot

The example consumed the sibling terminal's existing sandbox snapshot, without
provider credentials or a network request: 203 complete GMKN five-minute bars,
close as target, past open/high/low/lot-volume covariates and horizon 32. All 21
quantiles at every step were finite and monotone. CPU and Metal produced nearly
identical price paths. In the final Accelerate build, one CPU call took 16.05 ms;
one Metal call including shape warmup took 94.19 ms. These are individual calls,
not the warm-call medians above. The initial portable CPU build took 26.4 ms.

The emitted JSON files are local ignored artifacts:
`captures/market-forecast-cpu.json` and `captures/market-forecast-metal.json`.
Their provenance records the frozen bar cutoff, units, grid and inference options.
Forecast timestamps continue a wall-clock grid; they do not infer exchange sessions.
The run demonstrates integration and numerical behavior, not forecasting quality.

The same 203-bar / 32-step OHLCV example also succeeded for 123M and 209M on
both devices, with all 21 output quantiles finite and monotone. Individual first
calls took 128.71 / 146.33 ms for 123M and 206.67 / 102.89 ms for 209M (CPU /
Metal), including shape warmup. CPU/Metal forecasts differed by at most
`0.00001526` in the original price units across all 672 output values per model.
Outputs are `captures/market-forecast-{123M,209M}-{cpu,metal}.json`; summaries
and timings are retained in [benchmarks-m4.json](benchmarks-m4.json).

## Platform scope

macOS portable CPU, Accelerate CPU and Metal were executed locally for every
model size. Linux and Windows
CPU builds and 5M Linux reference inference are configured in
[CI](../.github/workflows/ci.yml), with a separate manual
[all-model reference workflow](../.github/workflows/reference.yml). These
workflows have not been executed here.
CUDA is an opt-in feature and was not compiled or tested on this Mac. No
cross-platform GPU or reduced-precision performance claim is made.

Run reference parity after downloading the pinned checkpoints:

```sh
TIMESTRAL_5M_DIR="$PWD/models/Timestral-5M" \
TIMESTRAL_28M_DIR="$PWD/models/Timestral-28M" \
TIMESTRAL_123M_DIR="$PWD/models/Timestral-123M" \
TIMESTRAL_209M_DIR="$PWD/models/Timestral-209M" \
cargo test --release --locked --test parity -- --ignored --test-threads=1

TIMESTRAL_5M_DIR="$PWD/models/Timestral-5M" \
TIMESTRAL_28M_DIR="$PWD/models/Timestral-28M" \
TIMESTRAL_123M_DIR="$PWD/models/Timestral-123M" \
TIMESTRAL_209M_DIR="$PWD/models/Timestral-209M" TIMESTRAL_BACKEND=metal \
cargo test --release --locked --features metal,accelerate --test parity -- \
  --ignored --test-threads=1
```
