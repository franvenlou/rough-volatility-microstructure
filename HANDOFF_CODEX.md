# Codex handoff

## Task and ground rules

Audit the whole repository for clarity, honest research claims, duplication, and
hygiene. Keep all authored material in English. Preserve methodology unless a
specific implementation bug is demonstrated. Record uncertain mathematics rather
than silently replacing it. Use real, incremental commits; do not rewrite history.

## Repository baseline and inventory

- Worktree: `/Users/franciscoventosolourido/Desktop/USC/Git/rough-volatility-microstructure`.
- Starting commit: `b01e314` on `main`; initially clean, tracking `origin/main`.
- Read all tracked text: README, five root Python scripts, seven `src` Python
  files, the notebook and its identical checkpoint. Tracked bytecode/Numba caches
  are generated binary artifacts, not independent source implementations.
- No tests, dependency manifest, `.gitignore`, or LICENSE file existed.
- The README ends inside a shell fence and includes an unsupported MIT badge.
- This file was created as an empty section skeleton before any other change.

## Running log and commits

1. `0d98a26` — Initial audit and handoff creation. No source changes yet.
   Independent work is assigned for core math/entropy, topology, and GMM/notebook;
   the coordinating agent owns ingestion, pipeline, documentation, hygiene, and
   all commits. Agents do not edit this handoff or the Git index.

2. `93f2d55` — Hygiene setup: add ignore rules and runtime dependencies,
   untrack existing bytecode/Numba caches/notebook checkpoint, add package markers.
   A Python 3.12.6 `.venv` now contains dependencies. The first download was
   blocked by sandbox networking; an approved retry encountered a TLS interruption;
   a second approved retry succeeded. No global environment was modified.
3. Source consolidation is present but not yet committed: math, entropy, topology,
   GMM, and notebook changes await coordinated test runs/review. Shared quote
   schema, offline toy-data generator, English pipeline, and bounded stream
   capture have been written. Mock HTTP engine migration and docs remain pending.

4. `383c0e8` — Core math consolidation: remove root Hurst/entropy scripts,
   retain q=1 default and optional q=2, correct OLS normalization, validate
   undefined inputs, and preserve imbalance/discrete entropy/MI under `src`.
   A new regression test caught constant 1e-8 return smoothing roundoff producing
   false finite H estimates. Constant inputs now remain bit-for-bit constant.
   The 16 core + 13 entropy + 8 ingestion checks pass in the shared environment.
   The ingestion files/tests are still unstaged for their own logical commit.

5. Ingestion and pipeline cleanup (current commit): move the custom-HTTP polling
   engine under `src/ingestion`, delete the root orchestrator, remove the dummy
   endpoint/no-op demo, and keep the distinct trade-buffer functionality.
   Callbacks now receive observed rows in arrival order and independent read-only
   copies, instead of uninitialized, wrapped, subsequently mutated buffer views.
   Callbacks still execute on the event loop and must be short. Request failures
   and cancellation are explicit; no backoff/reconnection system was added.
   The Binance adapter now bounds recv waits, uses aware local UTC timestamps,
   exposes CLI options, and rejects empty output. It still retains only best
   quotes in memory and writes once after capture.
   Shared snapshot validation, deterministic synthetic Parquet generation, and
   English threshold diagnostics make the offline pipeline executable.
   All eight ingestion/pipeline tests pass with no external market connection.

## Pending checklist

- [ ] A: README installation, runnable usage, status/limitations, license disclosure.
- [ ] B: Consolidate all five root Python scripts into canonical `src` modules.
- [ ] C: Remove instructional/trailer comments and translate all shipped text.
- [x] D: Replace threshold-triggered empirical/causal claims with diagnostics.
- [ ] E: Ignore/untrack generated files; declare actual dependencies; add real tests.
- [ ] Review every changed module; execute tests and offline end-to-end examples.
- [ ] Record each logical commit and final exact resume command here.

## Canonical implementation decisions

Planned locations (implementation and verification pending):

- `src/microstruct/core_math.py`: canonical Hurst, volume entropy, kernel smoother.
- `src/microstruct/entropy.py`: preserve distinct imbalance/discrete entropy/MI tools.
- `src/gmm_latent_space.py`: canonical rolling features and plotting; notebook imports it.
- `src/regime_gmm.py`: retain separate exploratory BIC/third-feature clustering.
- `src/topology/pmfg_filter.py`: retain PMFG; `network_filters.py` preserves distinct
  Ledoit-Wolf, MST, and face-insertion graph experiments from the root script.
- `src/topology/portfolio.py`: English filename for existing portfolio experiment.
- `src/ingestion/lob_streamer.py`: retain actual quote capture.
- `src/ingestion/polling_engine.py`: preserve custom-endpoint trade/quote buffers
  separately; the old mock URL is not a real data source.
- `src/ingestion/snapshots.py`: shared quote schema and return/volume features.
- `src/ingestion/synthetic_snapshot.py`: reproducible offline input generation.
- Run entrypoints from the repository root with `python -m src.<module>`.

### Completed core decisions

The `src` first-moment estimator is kept as the default because it is used by both
quote analysis paths. The root second-moment method survives as `moment_order=2`
in the same function, rather than being silently substituted. Shannon volume
entropy is one canonical function; discrete entropy counts observed labels and
uses that function for probabilities. Imbalance and MI are distinct functionality,
so they are preserved in `src/microstruct/entropy.py`. Arbitrary entropy bases now
use log(base); sparse labels no longer allocate arrays up to the largest label.
Volume imbalance uses explicit zero-depth handling instead of adding epsilon to
nonzero denominators. Numba remains on the Hurst and smoothing kernels; no
performance improvement or parity claim is made for other helpers.

## Mathematical concerns and defaults

- Root Hurst code uses second moments and divides the slope by two; existing
  `src` code uses first absolute moments. These are different estimators, not
  exact duplicates. Keep the first-moment default and expose any retained
  second-moment variant explicitly.
- `np.cov` uses sample normalization while `np.var` uses population normalization
  in the existing Hurst slope. This multiplies the slope by `m/(m-1)` for `m`
  lags. An algebraically consistent OLS slope is a demonstrable bug fix.
- `bandwidth=2`, `max_lag=20`, rolling window 50, rolling lag 5, return floor
  `1e-8`, entropy cutoff 0.4, GMM component count, and ridge `1e-4` have no
  calibration evidence in the repository. Preserve/expose and label as defaults.
- Thresholds on a single H estimate do not test rough volatility, Brownian
  dynamics, Black-Scholes validity, tradability, or tail-risk underestimation.
- The quote pipeline measures entropy of total best-quote volume across time,
  in nats. It does not measure directional toxicity or entropy across depth.
- Masking covariance or precision need not preserve positive definiteness;
  a fixed ridge does not guarantee it. No portfolio stability validation exists.
- Signed/zero graph weights must not determine whether an adjacency edge exists.
- The GMM third feature is mean absolute return in basis points, not log sigma.
- The root GMM's ESG feature and precision initialization are separate from the
  live quote pipeline; no demonstrated data/model integration exists.

## Validation evidence and limits

Math agent reports 28 checks passed in its system interpreter; the coordinating
agent will record shared-environment verification separately. Topology/GMM tests
are written and their first shared-environment runs are pending. Baseline Python is 3.14.4; NumPy, pandas, and pyarrow
are available globally, but most required packages are missing. Python 3.12 also
exists. A local isolated environment will be used where feasible. No real-market
validation or performance benchmark is planned or claimed.

Shared Python 3.12.6 / NumPy 2.5.3 / Numba 0.67.0 / llvmlite 0.49.0:
- 29 core/entropy tests pass, including the new exact constant-smoother regression.
- Seeded H=0.2 fBm ensemble (24 paths, 512 observations, max lag 20) means:
  q=1 0.207654933; q=2 0.205535062. Brownian H=0.5 (100,000 observations,
  max lag 50) estimates: 0.494148722 and 0.493245006. These are implementation
  sanity checks, not estimator calibration, confidence intervals, or market evidence.
- Initial ingestion tests found one failure (constant input artifact), fixed as
  above; all eight now pass, using a fake WebSocket connection only.
- Independent agents also report 11 GMM and 14 topology tests pass, both GMM
  CLIs and notebook cells execute, and topology demos execute. Full-suite
  coordinating run remains pending after final review.

## OPEN QUESTION FOR FRANCISCO

1. Should this repository reproduce a specific academic paper/protocol, or stand
   alone as a broader engineering showcase? No paper is present, and the crypto,
   GMM, ESG, and portfolio experiments do not establish such a reproduction.
2. Which license do you intend? The old README says MIT, but no license grant/file
   exists. Until you choose, documentation must report that absence honestly.
3. Which volatility proxy, sampling clock, smoothing/lag calibration, and
   out-of-sample validation should define future research claims?
4. Should the experimental ESG/third-feature and graph-masked portfolio branches
   remain part of the public scope once they are clearly labeled exploratory?

## Exact next step

From the repository root, run `git status --short` and
`cat HANDOFF_CODEX.md`. Next inspect/commit GMM and topology groups, complete README and tested
dependency snapshot, then execute the complete suite and documented offline commands. Do not claim test results before executing them.
