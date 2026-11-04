//! Per-series categorical encoding following Timestral's reference forecaster.
//!
//! See <https://github.com/mistralai/timestral/blob/main/src/timestral/covariate_encoding.py>
//! and the forecaster's `_encoding_target` method.

use std::collections::{BTreeMap, BTreeSet};

/// Encode one categorical channel over context and optional known future.
///
/// Targets must align with `context`; masked values must already be NaN. One
/// target uses smoothed target means. Multiple targets use deterministic codes,
/// assigned in lexical label order across context and future; missing labels
/// remain NaN. Only the last `max_context` target steps contribute to fitting.
pub(crate) fn encode(
    context: &[Option<&str>],
    future: &[Option<&str>],
    targets: &[&[f32]],
    max_context: usize,
) -> Vec<f32> {
    assert!(!targets.is_empty(), "categorical encoding requires targets");
    assert!(max_context > 0, "encoding context must be positive");
    assert!(targets.iter().all(|target| target.len() == context.len()));

    let [target] = targets else {
        return category_codes(context, future);
    };

    // Forward filling starts before the retained window, so an earlier observed
    // target can still fill missing values at the start of that window.
    let window_start = context.len().saturating_sub(max_context);
    let mut previous = None;
    let encoding_targets: Vec<Option<f64>> = target
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if value.is_finite() {
                previous = Some(f64::from(value));
            }
            (index >= window_start).then_some(previous).flatten()
        })
        .collect();

    let mut categories: BTreeMap<Option<&str>, Statistics> = BTreeMap::new();
    let mut total = Statistics::default();
    for (&category, target) in context.iter().zip(&encoding_targets) {
        if let Some(value) = target {
            categories.entry(category).or_default().observe(*value);
            total.observe(*value);
        }
    }
    let mean = if total.count == 0 {
        0.0
    } else {
        total.sum / total.count as f64
    };
    let lookup = |category: Option<&str>| match categories.get(&category) {
        Some(statistics) => ((mean + statistics.sum) / (1.0 + statistics.count as f64)) as f32,
        None => mean as f32,
    };

    context
        .iter()
        .zip(encoding_targets)
        .map(|(&category, target)| {
            if target.is_none() && category.is_none() {
                f32::NAN
            } else {
                lookup(category)
            }
        })
        .chain(future.iter().map(|&category| lookup(category)))
        .collect()
}

#[derive(Default)]
struct Statistics {
    sum: f64,
    count: usize,
}

impl Statistics {
    fn observe(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
    }
}

fn category_codes(context: &[Option<&str>], future: &[Option<&str>]) -> Vec<f32> {
    let labels: BTreeSet<&str> = context
        .iter()
        .chain(future)
        .filter_map(|&label| label)
        .collect();
    let codes: BTreeMap<&str, usize> = labels
        .into_iter()
        .enumerate()
        .map(|(index, label)| (label, index))
        .collect();
    context
        .iter()
        .chain(future)
        .map(|label| label.map_or(f32::NAN, |label| codes[label] as f32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::encode;

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }

    #[test]
    fn smooths_category_means_and_uses_series_mean_for_unseen_future() {
        let result = encode(
            &[Some("a"), Some("a"), Some("b"), Some("b")],
            &[Some("a"), Some("b"), Some("new"), None],
            &[&[10.0, 20.0, 30.0, 40.0]],
            4,
        );
        assert_close(result[0], 55.0 / 3.0);
        assert_close(result[2], 95.0 / 3.0);
        assert_eq!(result[0], result[4]);
        assert_eq!(result[2], result[5]);
        assert_eq!(&result[6..], &[25.0, 25.0]);
    }

    #[test]
    fn missing_labels_form_a_category_and_leading_padding_stays_missing() {
        let result = encode(
            &[None, Some("a"), Some("a"), None, None],
            &[None, Some("new")],
            &[&[f32::NAN, f32::NAN, 4.0, f32::NAN, 8.0]],
            5,
        );
        let mean = 16.0 / 3.0;
        assert!(result[0].is_nan());
        assert_close(result[1], (mean + 4.0) / 2.0);
        assert_eq!(result[1], result[2]);
        assert_close(result[3], (mean + 12.0) / 3.0);
        assert_eq!(result[3], result[4]);
        assert_eq!(result[4], result[5]);
        assert_close(result[6], mean);
    }

    #[test]
    fn context_cap_excludes_old_targets_but_keeps_their_forward_fill() {
        let result = encode(
            &[Some("old"), Some("old"), Some("a"), Some("b")],
            &[Some("new")],
            &[&[1000.0, 10.0, f32::NAN, f32::NAN]],
            2,
        );
        assert_eq!(result, vec![10.0; 5]);
    }

    #[test]
    fn entirely_unobserved_target_encodes_finite_labels_as_zero() {
        let result = encode(
            &[None, Some("a")],
            &[None, Some("a"), Some("new")],
            &[&[f32::NAN, f32::NAN]],
            2,
        );
        assert!(result[0].is_nan());
        assert_eq!(&result[1..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn multivariate_labels_use_consistent_codes_without_target_encoding() {
        let result = encode(
            &[Some("z"), None, Some("a"), Some("z")],
            &[Some("b"), None, Some("a")],
            &[&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0]],
            4,
        );
        assert_eq!(result[0], 2.0);
        assert!(result[1].is_nan());
        assert_eq!(&result[2..5], &[0.0, 2.0, 1.0]);
        assert!(result[5].is_nan());
        assert_eq!(result[6], 0.0);
    }
}
