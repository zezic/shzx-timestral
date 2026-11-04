//! Pinned reference outputs, exercised without a Python runtime.
//!
//! Set TIMESTRAL_5M_DIR, TIMESTRAL_28M_DIR, TIMESTRAL_123M_DIR and TIMESTRAL_209M_DIR
//! to checkpoint directories, then run
//! `cargo test --release --test parity -- --ignored --test-threads=1`.
//! To select one checkpoint, insert its test filter before `--`, e.g. `123m`.
//! TIMESTRAL_BACKEND selects `cpu` (default) or `metal` (requires that feature).
//!
//! Fixture cases are adapted from nfnt/datafusion-forecast's Apache-2.0 tests.
//! This version adds finite-value checks, backend selection and wrapper coverage;
//! see fixtures/README.md for attribution and exact revisions.

use std::{collections::HashMap, path::PathBuf};

use candle_core::{Device, Tensor};
use shzx_timestral::{
    Backend, Covariate, ForecastRequest, Forecaster, Nonnegative, Quantile, Settings, Symmetry,
    native,
};

const TOLERANCE: f32 = 1e-4;

struct Golden {
    name: &'static str,
    arrays: HashMap<String, Tensor>,
}

impl Golden {
    fn load(name: &'static str) -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(format!("{name}.safetensors"));
        Self {
            name,
            arrays: candle_core::safetensors::load(path, &Device::Cpu)
                .expect("load reference fixture"),
        }
    }

    fn tensor(&self, name: &str) -> &Tensor {
        self.arrays
            .get(name)
            .unwrap_or_else(|| panic!("missing fixture {name}"))
    }

    fn f32(&self, name: &str) -> Vec<f32> {
        self.tensor(name).flatten_all().unwrap().to_vec1().unwrap()
    }

    fn observed(&self, name: &str) -> Vec<bool> {
        self.tensor(name)
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap()
            .into_iter()
            .map(|x| x != 0)
            .collect()
    }

    fn ids(&self, name: &str) -> Vec<usize> {
        self.tensor(name)
            .to_vec1::<i64>()
            .unwrap()
            .into_iter()
            .map(|x| usize::try_from(x).expect("nonnegative fixture ID"))
            .collect()
    }

    fn dims(&self, name: &str) -> &[usize] {
        self.tensor(name).dims()
    }
}

fn checkpoint(variable: &str) -> PathBuf {
    std::env::var_os(variable).unwrap_or_else(|| {
        panic!("{variable} must point to the pinned checkpoint directory; see tests/fixtures/README.md")
    }).into()
}

fn backend() -> Backend {
    match std::env::var("TIMESTRAL_BACKEND")
        .as_deref()
        .unwrap_or("cpu")
    {
        "cpu" => Backend::Cpu,
        "metal" => Backend::Metal { ordinal: 0 },
        value => panic!("unsupported TIMESTRAL_BACKEND={value:?}; expected cpu or metal"),
    }
}

fn device() -> Device {
    match std::env::var("TIMESTRAL_BACKEND")
        .as_deref()
        .unwrap_or("cpu")
    {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0).expect("Metal device; build with --features metal"),
        value => panic!("unsupported TIMESTRAL_BACKEND={value:?}; expected cpu or metal"),
    }
}

fn assert_close(golden: &Golden, case: &str, actual: &[f32], expected: &[f32]) {
    let label = format!("{} {case}", golden.name);
    assert_eq!(actual.len(), expected.len(), "{label}: output length");
    assert!(!actual.is_empty(), "{label}: empty output");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite(),
            "{label}: nonfinite actual at {index}: {actual}"
        );
        assert!(
            expected.is_finite(),
            "{label}: nonfinite reference at {index}: {expected}"
        );
    }
    let magnitude = expected.iter().fold(1.0_f32, |max, x| max.max(x.abs()));
    let (index, error) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .unwrap();
    let relative = error / magnitude;
    eprintln!("{label}: max absolute error {error:.3e}, relative {relative:.3e} at {index}");
    assert!(
        relative < TOLERANCE,
        "{label}: error {relative} exceeds {TOLERANCE} at {index}: {} vs {}",
        actual[index],
        expected[index]
    );
}

fn native_suite(variable: &str, fixture: &'static str) {
    let golden = Golden::load(fixture);
    let forecaster = native::TimestralForecaster::from_dir(checkpoint(variable), &device())
        .expect("load native checkpoint");

    let groups = golden.ids("d1.group_ids");
    let quantiles = forecaster
        .model()
        .predict_block(
            &golden.f32("d1.inputs"),
            &golden.observed("d1.observed"),
            &groups,
            4,
            None,
            None,
        )
        .expect("context-only block");
    assert_close(
        &golden,
        "native d1",
        &quantiles,
        &golden.f32("d1.quantiles"),
    );

    let roles: Vec<_> = golden
        .ids("d2.roles")
        .into_iter()
        .map(|role| match role {
            0 => native::Role::Target,
            1 => native::Role::PastCovariate,
            2 => native::Role::KnownCovariate,
            _ => panic!("unexpected fixture role {role}"),
        })
        .collect();
    let groups = golden.ids("d2.group_ids");
    let future = golden.f32("d2.known_future");
    let future_mask = golden.observed("d2.known_future_mask");
    let quantiles = forecaster
        .model()
        .predict_block(
            &golden.f32("d2.inputs"),
            &golden.observed("d2.observed"),
            &groups,
            4,
            Some(&roles),
            Some((&future, &future_mask)),
        )
        .expect("covariate block");
    assert_close(
        &golden,
        "native d2",
        &quantiles,
        &golden.f32("d2.quantiles"),
    );

    let context = golden.f32("a.context");
    let input = native::ForecastInput::univariate(&context);
    let forecast = forecaster
        .forecast(&input, &native::ForecastOptions::new(96))
        .unwrap();
    assert_close(
        &golden,
        "native a",
        &forecast.quantiles,
        &golden.f32("a.quantiles"),
    );
    let options = native::ForecastOptions::new(96)
        .with_symmetric_averaging(false)
        .with_clip_nonnegative(false);
    let forecast = forecaster.forecast(&input, &options).unwrap();
    assert_close(
        &golden,
        "native a.raw",
        &forecast.quantiles,
        &golden.f32("a.raw"),
    );

    let context = golden.f32("b.context");
    let dims = golden.dims("b.context");
    let input = native::ForecastInput::new(&context, dims[0], dims[1]);
    let forecast = forecaster
        .forecast(&input, &native::ForecastOptions::new(1100))
        .unwrap();
    assert_close(
        &golden,
        "native b",
        &forecast.quantiles,
        &golden.f32("b.quantiles"),
    );

    let context = golden.f32("c.context");
    let past = golden.f32("c.past");
    let known = golden.f32("c.known");
    let input = native::ForecastInput::new(&context, 2, 1)
        .with_past_covariates(&past, golden.dims("c.past")[1])
        .with_known_covariates(&known, golden.dims("c.known")[1]);
    let forecast = forecaster
        .forecast(&input, &native::ForecastOptions::new(1100))
        .unwrap();
    assert_close(
        &golden,
        "native c",
        &forecast.quantiles,
        &golden.f32("c.quantiles"),
    );

    let forecaster = forecaster.with_max_context(1024);
    let context = golden.f32("e.context");
    let input = native::ForecastInput::new(&context, 3, 1);
    let forecast = forecaster
        .forecast(&input, &native::ForecastOptions::new(1100))
        .unwrap();
    assert_close(
        &golden,
        "native e",
        &forecast.quantiles,
        &golden.f32("e.quantiles"),
    );
}

fn wrapper_suite(variable: &str, fixture: &'static str) {
    let golden = Golden::load(fixture);
    let forecaster =
        Forecaster::from_dir(checkpoint(variable), backend()).expect("load wrapper checkpoint");

    let context = golden.f32("a.context");
    let request = ForecastRequest::new("close", &context, 96).unwrap();
    let forecast = forecaster.forecast(&request).unwrap();
    assert_close(
        &golden,
        "wrapper a",
        &forecast.as_native().quantiles,
        &golden.f32("a.quantiles"),
    );

    let forecaster = forecaster.with_settings(Settings {
        symmetry: Symmetry::Direct,
        nonnegative: Nonnegative::Unconstrained,
    });
    let forecast = forecaster.forecast(&request).unwrap();
    assert_close(
        &golden,
        "wrapper a.raw",
        &forecast.as_native().quantiles,
        &golden.f32("a.raw"),
    );
    let forecaster = forecaster.with_settings(Settings::default());

    for case in ["b", "c"] {
        wrapper_batched(&forecaster, &golden, case);
    }
    wrapper_trained_context(&forecaster);
    let forecaster = forecaster
        .with_max_context(1024)
        .expect("valid context cap");
    wrapper_batched(&forecaster, &golden, "e");
}

fn wrapper_trained_context(forecaster: &Forecaster) {
    // Exercise bounded attention at the trained limit, including head/chunk
    // remainders in the larger checkpoints.
    let config = forecaster.as_native().config();
    let context_steps = config.context_length;
    let horizon = config.patch_size;
    assert_eq!(forecaster.max_context(), context_steps);
    let context: Vec<_> = (0..context_steps)
        .map(|step| 100.0 + (step as f32 / 64.0).sin() + step as f32 * 0.0001)
        .collect();
    let request = ForecastRequest::new("close", &context, horizon).unwrap();
    let forecast = forecaster.forecast(&request).unwrap();
    assert_eq!(forecast.horizon(), horizon);
    assert_eq!(forecast.targets().len(), 1);
    assert_eq!(forecast.provenance().context_steps, context_steps);
    assert_eq!(forecast.provenance().max_context, context_steps);
    let target = forecast.target("close").expect("named full-context target");
    for step in 0..horizon {
        let quantiles = target.step(step).expect("full-context forecast step");
        assert_eq!(quantiles.len(), Quantile::ALL.len());
        assert!(quantiles.iter().all(|value| value.is_finite()));
        assert!(quantiles.windows(2).all(|pair| pair[0] <= pair[1]));
    }
    eprintln!(
        "wrapper trained context: {context_steps} steps, {horizon} finite monotone forecast steps"
    );
}

fn wrapper_batched(forecaster: &Forecaster, golden: &Golden, case: &str) {
    let context_name = format!("{case}.context");
    let context = golden.f32(&context_name);
    let dims = golden.dims(&context_name);
    let (batch, variates, time) = (dims[0], dims[1], dims[2]);
    let expected = golden.f32(&format!("{case}.quantiles"));
    let horizon = 1100;
    let quantiles = 21;
    let past = (case == "c").then(|| golden.f32("c.past"));
    let known = (case == "c").then(|| golden.f32("c.known"));
    for series in 0..batch {
        let offset = series * variates * time;
        let mut request =
            ForecastRequest::new("target_0", &context[offset..offset + time], horizon).unwrap();
        for variate in 1..variates {
            let start = offset + variate * time;
            request = request
                .with_target(format!("target_{variate}"), &context[start..start + time])
                .unwrap();
        }
        if let Some(past) = &past {
            let channels = golden.dims("c.past")[1];
            for channel in 0..channels {
                let start = (series * channels + channel) * time;
                request = request
                    .with_past_covariate(
                        format!("past_{channel}"),
                        Covariate::Numeric(&past[start..start + time]),
                    )
                    .unwrap();
            }
        }
        if let Some(known) = &known {
            let channels = golden.dims("c.known")[1];
            for channel in 0..channels {
                let start = (series * channels + channel) * (time + horizon);
                request = request
                    .with_known_covariate(
                        format!("known_{channel}"),
                        Covariate::Numeric(&known[start..start + time]),
                        Covariate::Numeric(&known[start + time..start + time + horizon]),
                    )
                    .unwrap();
            }
        }
        let forecast = forecaster.forecast(&request).unwrap();
        let start = series * variates * horizon * quantiles;
        let reference = &expected[start..start + variates * horizon * quantiles];
        assert_close(
            golden,
            &format!("wrapper {case} series {series}"),
            &forecast.as_native().quantiles,
            reference,
        );
        for (variate, name) in request.target_names().enumerate() {
            let actual: Vec<_> = forecast
                .target(name)
                .expect("named target")
                .quantile(Quantile::P50)
                .collect();
            let row_start = variate * horizon * quantiles;
            let expected_median: Vec<_> = reference[row_start..row_start + horizon * quantiles]
                .iter()
                .skip(10)
                .step_by(quantiles)
                .copied()
                .collect();
            assert_close(
                golden,
                &format!("wrapper {case} {name} median"),
                &actual,
                &expected_median,
            );
        }
    }
}

#[test]
#[ignore = "requires pinned Timestral-5M checkpoint in TIMESTRAL_5M_DIR"]
fn native_5m() {
    native_suite("TIMESTRAL_5M_DIR", "golden_5m");
}

#[test]
#[ignore = "requires pinned Timestral-28M checkpoint in TIMESTRAL_28M_DIR"]
fn native_28m() {
    native_suite("TIMESTRAL_28M_DIR", "golden_28m");
}

#[test]
#[ignore = "requires pinned Timestral-123M checkpoint in TIMESTRAL_123M_DIR"]
fn native_123m() {
    native_suite("TIMESTRAL_123M_DIR", "golden_123m");
}

#[test]
#[ignore = "requires pinned Timestral-209M checkpoint in TIMESTRAL_209M_DIR"]
fn native_209m() {
    native_suite("TIMESTRAL_209M_DIR", "golden_209m");
}

#[test]
#[ignore = "requires pinned Timestral-5M checkpoint in TIMESTRAL_5M_DIR"]
fn wrapper_5m() {
    wrapper_suite("TIMESTRAL_5M_DIR", "golden_5m");
}

#[test]
#[ignore = "requires pinned Timestral-28M checkpoint in TIMESTRAL_28M_DIR"]
fn wrapper_28m() {
    wrapper_suite("TIMESTRAL_28M_DIR", "golden_28m");
}

#[test]
#[ignore = "requires pinned Timestral-123M checkpoint in TIMESTRAL_123M_DIR"]
fn wrapper_123m() {
    wrapper_suite("TIMESTRAL_123M_DIR", "golden_123m");
}

#[test]
#[ignore = "requires pinned Timestral-209M checkpoint in TIMESTRAL_209M_DIR"]
fn wrapper_209m() {
    wrapper_suite("TIMESTRAL_209M_DIR", "golden_209m");
}
