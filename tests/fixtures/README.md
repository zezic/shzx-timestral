# Timestral reference-output fixtures

These files are copied unchanged from
[nfnt/datafusion-forecast](https://github.com/nfnt/datafusion-forecast/tree/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/crates/candle-timestral/tests/fixtures),
commit `cd7bb7799fbe8f67dc61b400fb52185388e0c5db`. They contain synthetic inputs and
reference predictions, not model weights. The upstream project distributes its
tests and fixtures under [Apache 2.0](LICENSE-APACHE-2.0); that license is retained
verbatim alongside these files. Attribution: nfnt/datafusion-forecast contributors.

The test scenarios in `../parity.rs` are adapted from the same project's
`crates/candle-timestral/tests/golden.rs`. Our version adds finite-value checks,
selectable backends and coverage of this library's named request/result API.

| File | SHA-256 | Required Hugging Face checkpoint revision |
|---|---|---|
| `golden_5m.safetensors` | `9072f370d64ca7b60c9d9109b4927916989833b5c0f73bef126972370f6c1d57` | `mistralai/Timestral-5M@bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f` |
| `golden_28m.safetensors` | `bb01d18bbba9294ef99051db434db5d89fccd884a9fe838684b6776d811b520d` | `mistralai/Timestral-28M@49b26409ac87b5935539a5d8ffe7f397b720db47` |
| `golden_123m.safetensors` | `4257e6f7b2b91b98a3f1d9d3d8680fb7f78c04150a8a2021ce97b068fcbe2a69` | `mistralai/Timestral-123M@5a147287addda85f6acd8da3fb8858fe9af7e8f7` |
| `golden_209m.safetensors` | `447ab2ec7bad7cf3c3ca19dd2e99f340aca89f9bc169e91115b57fe2d8b93f31` | `mistralai/Timestral-209M@9874d252ab0aae16f7f7c999f98f98d6ad7d8365` |

The checkpoint revisions are pinned in the upstream
[Golden workflow](https://github.com/nfnt/datafusion-forecast/blob/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/.github/workflows/golden.yml).
The upstream
[fixture generator](https://github.com/nfnt/datafusion-forecast/blob/cd7bb7799fbe8f67dc61b400fb52185388e0c5db/crates/candle-timestral/scripts/make_golden.py)
pins reference implementation
`mistralai/timestral@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`.
We consume its existing results directly in Rust. No Python generator or runtime
is needed or executed here.

Each fixture covers univariate default and direct inference, multivariate targets
with missing observations, numeric past and known-future covariates, binary future
indicators, long horizons crossing decode blocks, a reduced context cap and
nonnegative clipping, plus raw context-only and covariate block predictions.

Each wrapper suite also runs a synthetic smoke test at the checkpoint's full
trained context, checking output shape, provenance and all 21 finite monotone
quantiles. This exercises bounded attention and head/chunk remainders. It is
separate from the recorded-reference comparisons and has no stored golden output.

Run all four checkpoints on CPU:

```sh
TIMESTRAL_5M_DIR=/absolute/path/to/Timestral-5M \
TIMESTRAL_28M_DIR=/absolute/path/to/Timestral-28M \
TIMESTRAL_123M_DIR=/absolute/path/to/Timestral-123M \
TIMESTRAL_209M_DIR=/absolute/path/to/Timestral-209M \
cargo test --release --test parity -- --ignored --test-threads=1
```

For one checkpoint, insert its test-name filter before `--`, for example:

```sh
TIMESTRAL_123M_DIR=/absolute/path/to/Timestral-123M \
cargo test --release --test parity 123m -- --ignored --test-threads=1
```

The opt-in [Reference models workflow](../../.github/workflows/reference.yml)
downloads each pinned model and runs its native and named-API tests on Linux CPU.
The regular CI workflow retains the smaller 5M reference check on every push.

For Apple GPU
execution add `--features metal` to Cargo and `TIMESTRAL_BACKEND=metal` to the
environment. Explicitly requesting an unavailable backend or omitting a requested
checkpoint variable fails rather than silently skipping verification.

Every compared element must be finite before checking the upstream tolerance:
maximum absolute error divided by `max(1, max(abs(reference)))` must be below
`1e-4`. The denominator is shared by a compared case, not a per-element relative
error bound. These checks establish numerical correspondence to the pinned
reference, not forecasting quality on real markets.

Model checkpoints are obtained separately and retain their own Modified MIT
license; that license is distinct from the fixture/source code license.
