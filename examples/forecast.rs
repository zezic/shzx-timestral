//! Reproducible synthetic input and a small native inference timing harness.
use std::{error::Error, path::Path, time::Instant};

use serde_json::json;
use shzx_timestral::{Backend, Covariate, ForecastRequest, Forecaster, Model, Quantile};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() > 6 || args.first().is_some_and(|arg| arg == "--help") {
        eprintln!(
            "usage: forecast [5m|28m|123m|209m|checkpoint_dir] [cpu|metal|cuda] [context=512] [horizon=96] [repeats=3] [uni|market|multi]"
        );
        return Ok(());
    }
    let checkpoint = args.first().map(String::as_str).unwrap_or("5m");
    let backend = match args.get(1).map(String::as_str).unwrap_or("cpu") {
        "cpu" => Backend::Cpu,
        "metal" => Backend::Metal { ordinal: 0 },
        "cuda" => Backend::Cuda { ordinal: 0 },
        value => return Err(format!("unknown backend {value:?}").into()),
    };
    let context: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(512);
    let horizon: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(96);
    let repeats: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(3);
    if repeats == 0 {
        return Err("repeats must be positive".into());
    }
    let scenario = args.get(5).map(String::as_str).unwrap_or("uni");
    if !["uni", "market", "multi"].contains(&scenario) {
        return Err(format!("unknown scenario {scenario:?}").into());
    }
    let loaded = Instant::now();
    let model = match checkpoint {
        "5m" => Forecaster::from_hub(Model::Timestral5M, backend)?,
        "28m" => Forecaster::from_hub(Model::Timestral28M, backend)?,
        "123m" => Forecaster::from_hub(Model::Timestral123M, backend)?,
        "209m" => Forecaster::from_hub(Model::Timestral209M, backend)?,
        directory => Forecaster::from_dir(Path::new(directory), backend)?,
    };
    let load_ms = loaded.elapsed().as_secs_f64() * 1000.0;
    let close: Vec<_> = (0..context)
        .map(|t| 100.0 + t as f32 * 0.01 + (t as f32 / 12.0).sin())
        .collect();
    let volume: Vec<_> = (0..context)
        .map(|t| 1000.0 + 100.0 * (t as f32 / 8.0).cos())
        .collect();
    let spread: Vec<_> = (0..context)
        .map(|t| {
            if t % 37 == 0 {
                f32::NAN
            } else {
                0.02 + (t % 5) as f32 * 0.01
            }
        })
        .collect();
    let phase: Vec<_> = (0..context + horizon)
        .map(|t| (t as f32 / 24.0).sin())
        .collect();
    let session: Vec<_> = (0..context + horizon)
        .map(|t| Some(if t % 24 < 8 { "closed" } else { "open" }))
        .collect();
    let mut request = ForecastRequest::new("close", &close, horizon)?;
    if scenario != "uni" {
        request = request
            .with_past_covariate("spread", Covariate::Numeric(&spread))?
            .with_known_covariate(
                "phase",
                Covariate::Numeric(&phase[..context]),
                Covariate::Numeric(&phase[context..]),
            )?
            .with_known_covariate(
                "session",
                Covariate::Categorical(&session[..context]),
                Covariate::Categorical(&session[context..]),
            )?;
    }
    request = match scenario {
        "multi" => request.with_target("volume", &volume)?,
        "market" => request.with_past_covariate("volume", Covariate::Numeric(&volume))?,
        _ => request,
    };
    let warmup = Instant::now();
    let mut forecast = model.forecast(&request)?;
    let warmup_ms = warmup.elapsed().as_secs_f64() * 1000.0;
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        forecast = model.forecast(&request)?;
        times.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    let targets: Vec<_> = forecast
        .targets()
        .map(|target| {
            json!({
                "name": target.name(),
                "p10": target.quantile(Quantile::P10).take(3).collect::<Vec<_>>(),
                "median": target.median().take(3).collect::<Vec<_>>(),
                "p90": target.quantile(Quantile::P90).take(3).collect::<Vec<_>>()
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "checkpoint": checkpoint, "backend": format!("{:?}", model.backend()),
            "settings": format!("{:?}", model.settings()), "source": format!("{:?}", model.source()),
            "context": context, "horizon": horizon, "scenario": scenario, "repeats": repeats,
            "load_ms": load_ms, "warmup_ms": warmup_ms, "run_ms": times,
            "p50_ms": times[times.len() / 2], "targets_first_three_steps": targets
        }))?
    );
    Ok(())
}
