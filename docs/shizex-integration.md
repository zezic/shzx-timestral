# Shizex integration handoff

This repository supplies synchronous native inference behind the terminal's
existing worker boundary. It does not change the sibling terminal's UI, provider
adapter or account/order state. Its native Candle runtime supersedes the Python
worker proposal in the earlier
[terminal Timestral research](../../shizex/docs/timestral_integration_research.md).

## Dependency and ownership

Once published, add the Git dependency to the terminal's `Cargo.toml`:

```toml
[dependencies]
shzx-timestral = { git = "https://github.com/zezic/shzx-timerstal.git" }
```

For local development, use `shzx-timestral = { path = "../shzx-timestral" }`.
Keep the Rust import name `shzx_timestral` for either source. Commit the terminal's
`Cargo.lock`, or pin a published commit with `rev` in the Git dependency. See the
[quickstart](../README.md#quickstart) for a macOS feature example. Select
`hub` when downloading models, `metal` for Apple GPU inference, or `accelerate`
for Apple CPU matrix multiplication. The default CPU build needs none of these
features. See [validation](validation.md) for actual platform and latency results.

Keep one `Forecaster` on an inference worker and reuse it. Prepare an owned market
snapshot in the application, then borrow its arrays when constructing
`ForecastRequest` on that worker. The library does not fetch market data or need
provider credentials. Download a pinned `Model` separately or load its local
directory; retain the checkpoint identity with each job.

The terminal should own a bounded queue and a generation/job identity using its
existing task infrastructure. Replace superseded queued requests where appropriate.
Native inference is a blocking call without cooperative cancellation: discard a
completed result when its instrument, timeframe or request generation is stale.
Schedule jobs on explicit refresh or completed bars, independently of render
frames. No additional UI queue or channel protocol is implemented by this crate.

## Align data at a frozen cutoff

The provider adapter retains exact quotations, lots, timestamps and incomplete-bar
filtering; its [data contract](../../shizex/docs/native/data-adapter.md) remains the
source of truth. Construct separate FP32 inference arrays after selecting the
cutoff. Preserve the original integer/decimal data for accounting and display.

Choose one sampling policy for every target and covariate. `TimeGrid` represents
fixed wall-clock intervals in UTC milliseconds. `align_numeric` requires sorted,
unique observations on that grid and inserts NaN in missing slots. It neither
forward-fills bars nor knows exchange holidays. `with_time_grid` carries the grid
into the forecast; `future_grid()` continues wall-clock steps. A trading-session
index requires the terminal's own future timestamp mapping instead.

Avoid appending partial current bars without an explicit policy. Align additional
instruments and book/trade features to the same cutoff, including receipt-time
availability where publication or transport is delayed. A feature should use
only information available at that origin. Known-future covariates require real
advance knowledge, not realized future measurements.

## Represent market inputs

These are possible encodings, not claims that every additional feature improves
forecast accuracy. Compare alternatives through rolling-origin evaluation.

| Data | Library representation | Caller responsibility |
|---|---|---|
| Close, mid-price, returns, realized volatility | Named target or numeric past covariate | Define price basis, transform and units; invert transforms deliberately |
| OHLCV, several instruments | Joint targets or numeric past covariates | Align grids and adjustments; marginal forecasts do not enforce candle constraints |
| Trades, aggressor flow, volume, trade counts | Numeric past covariates | Aggregate recorded timestamped events; distinguish lots and shares |
| Order-book spread, imbalance, depth, slope | Numeric past covariates | Record historical snapshots and sample consistently; current books alone are insufficient |
| Indicators, fundamentals, macro observations | Numeric past covariates | Retain release times, revisions, warmup and missingness |
| Session, venue or event categories | Categorical past or known covariates | Use stable meaningful labels; only schedules belong in known future |
| Calendar position, holiday/session indicators | Numeric or categorical known covariates | Supply context plus the full requested horizon |
| News or other text | Externally extracted numeric/categorical features | Timestral takes no raw text; timestamp feature availability |
| Future price, volume or announcement outcome | Prediction target | Realized outcomes are not known-future conditioning inputs |

For a minimal richer request inside a function returning `shzx_timestral::Result<_>`:

```rust
use shzx_timestral::{Covariate, ForecastRequest};

let close = [100.0, 101.0, f32::NAN, 102.0];
let volume = [10.0, 12.0, f32::NAN, 9.0];
let session = [Some("open"), Some("open"), None, Some("open")];
let future_session = [Some("open"), Some("closed")];
let request = ForecastRequest::new("close", &close, 2)?
    .with_past_covariate("volume_lots", Covariate::Numeric(&volume))?
    .with_known_covariate(
        "session",
        Covariate::Categorical(&session),
        Covariate::Categorical(&future_session),
    )?;
```

Channel names are unique across targets and both covariate roles. Every history
matches target history length, while each known channel's future half matches the
horizon. Numbers use NaN for missingness and otherwise must be finite. Each target
needs at least one observation in the retained context. Mask a numeric target by
setting the corresponding inference copy to NaN; the lower-level native API also
exposes explicit observation masks.

Univariate categorical rows use smoothed target encoding over the retained
context, including forward-filled internal target gaps. Missing category labels
form a category; unseen future labels use the fitted mean. With multiple targets,
labels use deterministic lexical codes across that channel's context and future.
Adding a lexically earlier label can therefore change those numeric codes; callers
requiring externally fixed code values can supply numeric covariates directly.

## Publish forecasts with provenance

Read `forecast.target(name)` or iterate `forecast.targets()`. `median()` and
`quantile(Quantile)` return iterators in horizon order; `interval` pairs pointwise
bounds. `step(index)` supplies all 21 values in `Quantile::ALL` order. Native
batched/raw tensor access remains available through `shzx_timestral::native` and
`Forecaster::as_native()` when the application already owns dimension validation
and categorical preprocessing.

The result's provenance stores checkpoint source, backend, symmetry/nonnegative
settings, retained-context cap and original context length. The terminal should
persist the instrument, exact data cutoff, sampling/session policy, input version,
units, adjustments/transforms and generation time alongside it. A directory load
cannot infer its model revision; record that externally. Preserve each result's
original origin as later actual prices arrive.

Render the median and selected quantile bands as a projection. Quantiles are
marginal values at each step, not complete sampled future paths or a forecast
mean. Joint target inference does not produce coherent future OHLC candles or
trading actions. Measure errors, quantile loss and empirical coverage against a
persistence baseline; implementation parity alone does not validate trading use.

## Existing runnable bridge

From this repository:

```sh
cargo run --release --example market_snapshot -- \
  models/Timestral-5M ../shizex/captures/market-sandbox.json cpu 32
```

The example reads the terminal's existing market-only snapshot, checks instrument
identity and completed OHLC bar bounds, aligns history on a regular grid and
predicts close using OHLC/lot-volume past covariates. It emits all quantiles plus
P10/median/P90 and provenance as JSON. It opens no provider connection and performs
no account or order operation. It uses wall-clock future steps; it does not infer
a trading calendar. The locally successful 5M CPU run and reference parity results
are recorded in [validation](validation.md).

The library code's Apache-2.0 license and the downloaded weights' Modified MIT
license are distinct; see the [README licensing note](../README.md).
