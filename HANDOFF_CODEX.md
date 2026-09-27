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

1. Initial audit and handoff creation: this commit. No source changes yet.
   Independent work is assigned for core math/entropy, topology, and GMM/notebook;
   the coordinating agent owns ingestion, pipeline, documentation, hygiene, and
   all commits. Agents do not edit this handoff or the Git index.

## Pending checklist

- [ ] A: README installation, runnable usage, status/limitations, license disclosure.
- [ ] B: Consolidate all five root Python scripts into canonical `src` modules.
- [ ] C: Remove instructional/trailer comments and translate all shipped text.
- [ ] D: Replace threshold-triggered empirical/causal claims with diagnostics.
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
- `src/ingestion/`: retain real quote capture and any useful mock-engine primitives.
- Run entrypoints from the repository root with `python -m src.<module>`.

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

No new tests have run yet. Baseline Python is 3.14.4; NumPy, pandas, and pyarrow
are available globally, but most required packages are missing. Python 3.12 also
exists. A local isolated environment will be used where feasible. No real-market
validation or performance benchmark is planned or claimed.

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
`cat HANDOFF_CODEX.md`. Then finish `.gitignore`, dependency declarations and
package markers; inspect the pending independent source changes before staging
only one logical group at a time. Do not claim test results before executing them.
