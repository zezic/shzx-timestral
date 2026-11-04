//! Forecast a Shizex market snapshot without provider credentials or network access.
//!
//! cargo run --release --example market_snapshot -- \
//!   /path/to/Timestral-5M ../shizex/captures/market-sandbox.json cpu 32

use std::{env, error::Error, fs::File, io, io::Write, time::Duration, time::Instant};

use chrono::DateTime;
use serde::Deserialize;
use serde_json::json;
use shzx_timestral::{
    Backend, Covariate, ForecastRequest, Forecaster, Nonnegative, Quantile, Symmetry, TimeGrid,
};

#[derive(Deserialize)]
struct Snapshot {
    format_version: u32,
    environment: String,
    provenance: Provenance,
    instrument: Instrument,
    history: History,
}

#[derive(Deserialize)]
struct Provenance {
    provider: String,
    fetched_at: String,
}

#[derive(Deserialize)]
struct Instrument {
    uid: String,
    ticker: String,
    currency: String,
    lot_size: u32,
}

#[derive(Deserialize)]
struct History {
    instrument: String,
    timeframe: Timeframe,
    from: String,
    to: String,
    observed_at: String,
    bars: Vec<Candle>,
}

#[derive(Deserialize)]
enum Timeframe {
    #[serde(rename = "1")]
    M1,
    #[serde(rename = "5")]
    M5,
    #[serde(rename = "15")]
    M15,
    #[serde(rename = "30")]
    M30,
    #[serde(rename = "60")]
    H1,
    #[serde(rename = "120")]
    H2,
}

impl Timeframe {
    fn interval(&self) -> Duration {
        let minutes = match self {
            Self::M1 => 1,
            Self::M5 => 5,
            Self::M15 => 15,
            Self::M30 => 30,
            Self::H1 => 60,
            Self::H2 => 120,
        };
        Duration::from_secs(minutes * 60)
    }
}

#[derive(Deserialize)]
struct Candle {
    opens_at: String,
    closes_at: String,
    open: i64,
    high: i64,
    low: i64,
    close: i64,
    volume_lots: u64,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = env::args().skip(1).collect();
    if !(2..=4).contains(&arguments.len()) {
        return Err(invalid(
            "usage: market_snapshot CHECKPOINT_DIR SNAPSHOT_JSON [cpu|metal] [HORIZON=32]",
        )
        .into());
    }
    let backend_name = arguments.get(2).map_or("cpu", String::as_str);
    let backend = match backend_name {
        "cpu" => Backend::Cpu,
        "metal" => Backend::Metal { ordinal: 0 },
        _ => return Err(invalid("backend must be cpu or metal").into()),
    };
    let horizon = arguments
        .get(3)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(32);
    let snapshot: Snapshot = serde_json::from_reader(File::open(&arguments[1])?)?;
    if snapshot.format_version != 1 {
        return Err(invalid("unsupported market snapshot format").into());
    }
    if snapshot.history.instrument != snapshot.instrument.uid {
        return Err(invalid("snapshot history belongs to another instrument").into());
    }
    let bars = &snapshot.history.bars;
    if bars.is_empty() {
        return Err(invalid("snapshot contains no closed candles").into());
    }
    let interval = snapshot.history.timeframe.interval();
    let interval_ms = i64::try_from(interval.as_millis())?;
    let history_from = DateTime::parse_from_rfc3339(&snapshot.history.from)?;
    let history_to = DateTime::parse_from_rfc3339(&snapshot.history.to)?;
    let observed_at = DateTime::parse_from_rfc3339(&snapshot.history.observed_at)?;
    let mut timestamps = Vec::with_capacity(bars.len());
    for bar in bars {
        let opens_at = DateTime::parse_from_rfc3339(&bar.opens_at)?;
        let closes_at = DateTime::parse_from_rfc3339(&bar.closes_at)?;
        if (closes_at - opens_at).to_std()? != interval
            || opens_at < history_from
            || closes_at > history_to
            || closes_at > observed_at
            || !closes_at.timestamp_subsec_nanos().is_multiple_of(1_000_000)
        {
            return Err(invalid("candle is incomplete or outside the snapshot cutoff").into());
        }
        if bar.low > bar.open.min(bar.close)
            || bar.high < bar.open.max(bar.close)
            || bar.low > bar.high
        {
            return Err(invalid("candle has inconsistent OHLC bounds").into());
        }
        timestamps.push(closes_at.timestamp_millis());
    }
    let start = timestamps[0];
    let origin = timestamps[timestamps.len() - 1];
    let span = origin
        .checked_sub(start)
        .ok_or_else(|| invalid("candle history spans too many milliseconds"))?;
    let steps = usize::try_from(span / interval_ms)?
        .checked_add(1)
        .ok_or_else(|| invalid("candle history has too many steps"))?;
    let grid = TimeGrid::new(start, interval, steps)?;

    // Prices retain the snapshot's quotation units until conversion to model f32.
    // Missing time slots stay NaN, including any gaps between trading sessions.
    let align_price = |value: fn(&Candle) -> i64| {
        grid.align_numeric(
            timestamps
                .iter()
                .copied()
                .zip(bars.iter().map(|bar| value(bar) as f64 / 1_000_000_000.0)),
        )
    };
    let close = align_price(|bar| bar.close)?;
    let open = align_price(|bar| bar.open)?;
    let high = align_price(|bar| bar.high)?;
    let low = align_price(|bar| bar.low)?;
    let volume = grid.align_numeric(
        timestamps
            .iter()
            .copied()
            .zip(bars.iter().map(|bar| bar.volume_lots as f64)),
    )?;
    let request = ForecastRequest::new("close", &close, horizon)?
        .with_past_covariate("open", Covariate::Numeric(&open))?
        .with_past_covariate("high", Covariate::Numeric(&high))?
        .with_past_covariate("low", Covariate::Numeric(&low))?
        .with_past_covariate("volume_lots", Covariate::Numeric(&volume))?
        .with_time_grid(grid)?;

    let load_start = Instant::now();
    let forecaster = Forecaster::from_dir(&arguments[0], backend)?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let forecast_start = Instant::now();
    let forecast = forecaster.forecast(&request)?;
    let forecast_seconds = forecast_start.elapsed().as_secs_f64();
    let target = forecast.target("close").expect("requested close target");
    let future = forecast.future_grid().expect("request has a time grid");
    let points: Vec<_> = future
        .timestamps()
        .zip(target.quantile(Quantile::P10))
        .zip(target.quantile(Quantile::P50))
        .zip(target.quantile(Quantile::P90))
        .enumerate()
        .map(|(step, (((timestamp, p10), median), p90))| {
            json!({
                "closes_at_ms": timestamp,
                "p10": p10,
                "median": median,
                "p90": p90,
                "quantiles": target.step(step).expect("forecast step")
            })
        })
        .collect();
    let output = json!({
        "format_version": 1,
        "instrument": {
            "uid": snapshot.instrument.uid,
            "ticker": snapshot.instrument.ticker,
            "currency": snapshot.instrument.currency,
            "lot_size": snapshot.instrument.lot_size
        },
        "provenance": {
            "provider": snapshot.provenance.provider,
            "environment": snapshot.environment,
            "snapshot_fetched_at": snapshot.provenance.fetched_at,
            "history_observed_at": snapshot.history.observed_at,
            "forecast_origin_ms": origin,
            "timestamp_unit": "UTC Unix milliseconds at candle close",
            "sampling_interval_ms": interval_ms,
            "sampling_policy": "regular wall-clock grid; missing slots remain missing",
            "checkpoint_directory": arguments[0],
            "checkpoint_revision": null,
            "backend": backend_name,
            "symmetric_averaging": forecaster.settings().symmetry == Symmetry::Average,
            "clip_nonnegative": forecaster.settings().nonnegative == Nonnegative::Clip,
            "max_context": forecaster.max_context(),
            "closed_candles": bars.len(),
            "context_steps": grid.len(),
            "missing_steps": close.iter().filter(|value| value.is_nan()).count(),
            "target": "close",
            "price_unit": "currency per share",
            "past_covariates": ["open", "high", "low", "volume_lots"],
            "volume_unit": "lots",
            "known_future_covariates": [],
            "load_seconds": load_seconds,
            "forecast_seconds": forecast_seconds
        },
        "quantile_levels": Quantile::ALL.map(Quantile::level),
        "points": points
    });
    let mut stdout = io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, &output)?;
    writeln!(stdout)?;
    Ok(())
}
