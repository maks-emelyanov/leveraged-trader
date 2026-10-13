# Historical grid comparison — 2026-10-12

**Decision: retain the current 49 combinations. The available evidence does not establish an optimal
grid.** The leading exploratory alternative has 35 combinations, but its entire advantage comes
from one asset in one year. No replacement met the acceptance criteria fixed before inspecting
results. Retaining 49 is a decision against an unsupported change, not evidence that 49 is optimal.

This archived comparison uses robust selection version 1. Its evaluator remains pinned to that
version; these results do not evaluate the completed-trade activity reward added in version 2.

The current grid is:

- Long-product RSI thresholds: **20, 25, 30, 35, 40, 45, 50**.
- Inverse-product RSI thresholds: **50, 55, 60, 65, 70, 75, 80**.
- Profit-target multiples: **1.02, 1.05, 1.10, 1.20, 1.30, 1.50, 2.00**.

The prior reduction from **1,240 = 31 RSI thresholds × 40 targets** to **49 = 7 × 7** was a deliberate
change of search space: RSI spacing increased from 1 to 5, and targets changed from 1.10–5.00 in
0.10 increments to the seven values above. It was not a historical test that eliminated 1,191
unsuccessful combinations. This study supplies the subsequent evidence check.

## Evidence and candidate grids

Eight grid designs were registered before evaluating their outcomes. Every family uses the same
robust selector, duration penalties, neighborhood smoothing, and existing buy-eligibility gates.
The legacy grid label refers to its parameter values; it does **not** restore legacy Sharpe-only
selection. Threshold counts and combination counts below apply separately to each direction.

| Family | RSI thresholds: long / inverse | Target multiples | Pairs |
| --- | --- | --- | ---: |
| Current | 20–50 / 50–80, step 5 | 1.02, 1.05, 1.10, 1.20, 1.30, 1.50, 2.00 | 49 |
| Legacy values | 20–50 / 50–80, step 1 | 1.10–5.00, step 0.10 | 1,240 |
| Denser RSI | 20–50 / 50–80, step 1 | Current targets | 217 |
| Coarser RSI | 20–50 / 50–80, step 10 | Current targets | 28 |
| Shorter targets | Current thresholds | 1.01, 1.02, 1.03, 1.05, 1.075, 1.10, 1.15, 1.20 | 56 |
| Medium targets | Current thresholds | 1.05, 1.10, 1.20, 1.30, 1.50, 2.00 | 42 |
| Wider RSI range | 15–55 / 45–85, step 5 | Current targets | 63 |
| Narrower RSI range | 25–45 / 55–75, step 5 | Current targets | 35 |

The cache has 547 current products. Requiring at least 1,260 prior aligned asset sessions leaves
**76 funds: 38 long and 38 inverse, representing 33 signal proxies**. There are **455 eligible
asset-years** in 2017–2022: 75 funds in 2017 and 76 in each subsequent year. All eight grids receive
exactly this sample, for 3,640 grid/asset/year records. Inverse products use 32 of the 33 proxies.

For each annual test, parameter selection sees only earlier data. Its own development blocks and
holdout are entirely within that earlier training prefix. A selected pair must pass the existing
robust acceptance rules and have full-training Sharpe at least 1.0 before the outer test can trade.
Rejected selections remain cash and remain in all comparison denominators. Every outer period
starts flat, with causal RSI warmed up from earlier signal history.
These tests freeze the policy for each outer period; they do not reconstruct the production
workflow's repeated parameter selection as new daily observations arrive. Cash in these tests
does not imply that a daily production replay would remain cash throughout the same years.

The primary outer-test utility is:

```text
Sharpe − 0.5 × log(1 + mean holding sessions / 21)
       − 0.25 × log(1 + maximum holding sessions / 63)
```

Zero-exposure outcomes contribute zero utility and zero return; cash earns no interest in the
simulation. Returns and holding times are also reported separately. Metrics are averaged across
asset/year cells within each signal proxy, then equally across proxies. These are research
comparisons, not the return of a constructed investable portfolio.

Base trading cost is 3 bps per side. A second run uses 10 bps per side with the **same selected
parameters**. Each family recomputes its own neighborhood scores: a restricted grid cannot inherit
the dense grid's scores. Raw simulations are cached across overlapping families; only each frozen
winner's inner-holdout result can affect acceptance.

## Results

“Accepted” counts policies that passed all training gates; “exposed” counts those that subsequently
held an asset. A policy can pass training yet encounter no entry signal during its outer test.

| Grid | Accepted / 455 | Exposed / 455 | Proxy-weighted utility | Utility at 10 bps/side |
| --- | ---: | ---: | ---: | ---: |
| Narrower RSI, 35 | 1 | 1 | 0.001972 | 0.001856 |
| Denser RSI, 217 | 2 | 2 | 0.001295 | 0.001189 |
| Coarser RSI, 28 | 0 | 0 | 0 | 0 |
| Medium targets, 42 | 1 | 0 | 0 | 0 |
| Current, 49 | 0 | 0 | 0 | 0 |
| Shorter targets, 56 | 0 | 0 | 0 | 0 |
| Wider RSI, 63 | 0 | 0 | 0 | 0 |
| Legacy values, 1,240 | 2 | 1 | −0.000949 | −0.000961 |

There are only **four exposed records in the entire comparison, all TQQQ**. All inverse policies
remain cash. Thus the apparent ranking has almost no breadth and cannot establish either the best
RSI range or the best target set across the universe.

The 35-point family selects RSI **45** and target **1.02** for TQQQ's 2017 outer test, based entirely on
pre-2017 observations. That test returns **15.35%**, completes **seven** round trips, and holds for
**2.57 sessions on average, seven at maximum**. Its weighted improvement over the current grid is
0.001972. The registered cluster bootstrap's multiplicity-adjusted interval is **[0, 0.009586]**:
the lower bound is not positive, and only **one** proxy group differs, versus the required ten.
There are no matched exposed outcomes against the current grid from which to establish better
holding duration. Short holds in this isolated example do not demonstrate a reliable reduction.

For context, the dense grid's exposed cases are TQQQ 2017 and 2018; the latter loses **22.67%** and
finishes with an open position aged 56 sessions. The legacy grid's exposed TQQQ 2018 case loses
**22.31%**, with an open age of 57 sessions. These small samples are not sufficient to rank either
family reliably.

The current grid's inactivity is a substantive finding: 432 of its 455 selections lack sufficient
development evidence, four fail the inner holdout, and 19 pass robust selection but fail the
full-training Sharpe gate. Those 19 robust passes select target 1.02 eleven times and 1.05 eight
times. This is a descriptive clue about the duration-aware objective; it is **not** out-of-sample
evidence for deleting all other targets. The gates were not relaxed after observing inactivity.

## Later confirmation and decision

The 35-point family was frozen as the exploratory leader before evaluating 2023-01-01 through
2026-10-08. Only it and the incumbent were evaluated over this continuous, 945-session period for
the same 76 funds. Validation-prefix input hashes, signal mappings, and cohort membership were
checked for equality between phases.

**Neither family activates any policy in confirmation.** Current49 has 75 insufficient-evidence
results and one failed inner holdout; narrow_rsi35 has 74 and two respectively. Their identical
cash outcome provides no positive confirmation of performance or holding time.

The registered replacement criteria were: a positive adjusted lower confidence bound; at least ten
discordant proxies with at least 60% improving; no direction-level utility deterioration;
nonnegative improvement under higher costs; no worsening of matched exposed mean or maximum
holding time; and nonnegative utility improvement in continuous confirmation. The candidate fails
the uncertainty, breadth, and matched-holding requirements. The arithmetic confirmation gate
passes on equal cash outcomes, but cannot rescue failed validation gates.

**No strategy defaults or trading state were changed by this grid-family study.** The 35-point
family is retained as a documented research candidate, not promoted as an optimum or a trade
recommendation. A useful next study would need enough independent trading episodes to distinguish
families; changing the selection gates would be a separate experiment with new acceptance rules.

## Limits and reproducibility

The universe is today's surviving, cached products, with today's signal mappings applied
historically. It includes 72 ProShares products and four from Nasdaq ETF definitions. Direxion
history and a reconstructed delisted universe are absent. Signal proxies remain correlated despite
equal weighting. Daily-bar fills, fractional sizing, and fixed transaction costs are research
assumptions. Open holding durations are censored ages, and annual resets cannot measure full
position lifetimes. The continuous confirmation cannot resolve this here because it makes no trades.

Some TQQQ/SQQQ recent history was inspected earlier in this work. The later period is therefore
retrospective confirmation, **not a pristine holdout**. Nested selection limits reuse of outer
outcomes, but does not erase prior inspection or make repeated historical experimentation unbiased.
See the [scikit-learn nested-selection example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
and [Bailey et al., The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

The source database was opened using SQLite `mode=ro` in a consistent snapshot for each phase.
The study did not initialize or migrate it, fetch market data, access a broker, or send orders.
RSI uses full prior proxy history; source observations after each phase cutoff cannot affect its
inputs. The bootstrap uses 2,000 signal-proxy cluster resamples, seed 20261012, and two-sided
Bonferroni tails for seven comparisons. Sparse, correlated clusters limit the inference.

Study artifacts in [research/grid-search-2026-10-12](../research/grid-search-2026-10-12/):

- `protocol.json`: fixed grid families, periods, ranking, and adoption rules.
- `validation_rows.csv`, `validation_summary.json`, `validation_coverage.csv`: all trials,
  aggregated results, and included/excluded assets with input hashes.
- `frozen_choice.json`: the leader fixed before confirmation; retained unchanged afterward.
- `confirmation_rows.csv`, `confirmation_summary.json`, `confirmation_coverage.csv`: later results.
- `decision.json`: final application of the registered adoption gates.
- `run_notes.json`: a summary-only implementation correction to match the registered bootstrap,
  made before inspecting or ranking results. Trial outcomes were not changed.

Run these commands from a repository checkout after completing the README setup. The saved
study artifacts live in the repository; they are not included in the installed package.
To reproduce against the same cached data, use an empty result directory and copy the protocol:

```bash
mkdir -p /tmp/grid-evidence-rerun
cp research/grid-search-2026-10-12/protocol.json /tmp/grid-evidence-rerun/protocol.json
NUMBA_NUM_THREADS=2 .venv/bin/python scripts/research-grid.py --study-dir /tmp/grid-evidence-rerun --phase validation
NUMBA_NUM_THREADS=2 .venv/bin/python scripts/research-grid.py --study-dir /tmp/grid-evidence-rerun --phase freeze
NUMBA_NUM_THREADS=2 .venv/bin/python scripts/research-grid.py --study-dir /tmp/grid-evidence-rerun --phase confirmation
NUMBA_NUM_THREADS=2 .venv/bin/python scripts/research-grid.py --study-dir /tmp/grid-evidence-rerun --phase decide
```

Results cannot overwrite an existing phase. The runner accepts `--db PATH` for a separately
preserved cache. New data revisions or mappings can change a new study's results; confirmation of
an existing study rejects changes to its validation inputs. The study shares the production
simulation kernel but explicitly selects version 1; production defaults to version 2. The optional
evaluator hook exists only to cache duplicate research simulations.

Verification: 120 focused tests and 170 subtests pass, Ruff and whitespace checks pass, and an
independent rerun through all four phases reproduces every trial, coverage record, summary,
frozen choice, and decision (numerical tolerance 1e-12).
