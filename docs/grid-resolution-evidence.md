# Grid resolution evidence

October 12, 2026. **No statistically supported optimal grid or performance degradation boundary
was found.** The current version 2 selector was tested with 45 grid designs, starting with the
requested 4,059 pairs per direction and continuing to the four-pair endpoint limit. The strongest
validation candidate, RSI step 4 and profit-multiple step 0.04, is approximately 4.9 times faster
but fails the evidence requirements and loses its positive utility in later confirmation.
It is an exploratory candidate, not a recommended replacement. Production defaults remain at
49 pairs; this does not establish that 49 is optimal.

The completed study contains **34,020 validation selections and 3,910 confirmation selections**.
The [machine-readable decision](../research/grid-resolution-2026-10-12/decision.json) and
[all-grid comparison](../research/grid-resolution-2026-10-12/comparison.csv) accompany this report.

## Search ranges and spacing

The finest grid uses long RSI **10–50, step 1**, inverse-product RSI **50–90, step 1**, and
profit multiples **1.02–2.00, step 0.01**. Each direction has 41 × 99 = **4,059 pairs**;
long and inverse policies are optimized separately.

Every coarser grid preserves those ranges and endpoints. Targets start at 1.02; if the spacing
does not land on 2.00, that endpoint is appended. The 45 designs comprise 12 joint spacings,
10 RSI-only changes, 11 target-only changes, 11 shifted joint grids, and the incumbent.
Shifted grids move the interior points by half a step, rounded down to whole RSI points or
target cents, while preserving endpoints. The incumbent changes ranges as well as spacing and
is a separate comparator.

All 12 joint spacings were completed, including the endpoint-only grid; the study did not stop
at the first apparent winner. The following results include all 756 eligible validation
asset-years, with inactive policies counted as cash.

| RSI step / target step | Pairs per direction | Exposed periods | Completed trades | Mean utility | Median search ms | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 0.01 | 4,059 | 10 | 34 | −0.077680 | 121.3 | 1.0× |
| 2 / 0.02 | 1,050 | 8 | 45 | −0.085797 | 45.2 | 2.7× |
| 3 / 0.03 | 510 | 6 | 46 | −0.000863 | 33.2 | 3.7× |
| 4 / 0.04 | 286 | 4 | 55 | +0.003551 | 24.6 | 4.9× |
| 5 / 0.05 | 189 | 1 | 13 | −0.000340 | 21.1 | 5.7× |
| 8 / 0.08 | 84 | 0 | 0 | 0 | 17.2 | 7.1× |
| 10 / 0.10 | 55 | 0 | 0 | 0 | 14.8 | 8.2× |
| 15 / 0.15 | 32 | 0 | 0 | 0 | 13.2 | 9.2× |
| 20 / 0.20 | 18 | 0 | 0 | 0 | 12.6 | 9.6× |
| 30 / 0.30 | 15 | 0 | 0 | 0 | 12.8 | 9.5× |
| 40 / 0.49 | 6 | 0 | 0 | 0 | 11.8 | 10.3× |
| 40 / 0.98 | 4 | 0 | 0 | 0 | 10.9 | 11.2× |

Utility is Sharpe minus `0.5 × log(1 + mean holding sessions / 21)` and
`0.25 × log(1 + maximum holding sessions / 63)`. Outcomes are averaged within each signal proxy,
then equally across proxies. These are comparisons of asset policies, not returns of an
investable portfolio. Completed trades are round trips. A still-open losing position remains
marked to market even if it has no completed trades. The selector retains its version 2
trade-activity reward; annual trade rates and activity-adjusted outer utility are also saved.

**Exposure falls from four periods at step 4 to one at step 5 and zero at step 8**
for the unshifted joint grids.
This is not a reliable universal performance threshold: shifted step 8 still trades, and going
to cash can improve the measured result when the fine benchmark loses money. No statistically
supported degradation boundary emerged before the range-preserving endpoint limit.

## Overfitting and outlier evidence

The apparent step-4 leader has long thresholds `10, 14, …, 50`, inverse thresholds
`50, 54, …, 90`, and targets `1.02, 1.06, …, 1.98, 2.00`: 11 × 26 = **286 pairs**, 92.95% fewer
than the finest grid. Four findings prevent treating its rank as an optimum:

- All four exposed validation periods are **TQQQ in 2014–2017**. Removing the QQQ signal group
  removes its entire trading record. There are no inverse exposures anywhere in the validation
  study, across all 45 grids.
- Shifting the step-4 grid reverses its utility from **+0.003551 to −0.006874**. At spacing 4,
  target-only utility is −0.002809 and RSI-only utility is −0.077244. The result depends on the
  parameter axis and grid alignment.
- Its improvement over the finest grid is **+0.081231 utility**. About 90% of the absolute
  proxy contributions to that comparison comes from avoiding one **LTL 2022** outcome, which
  loses 40.26%. Excluding that proxy for sensitivity reduces the improvement to **+0.005601**;
  symmetric 10% trimming of paired proxy differences reduces it to **+0.001054**.
- The finest grid supplies only **five exposed proxy groups and 34 completed trades**. The
  registered minimums are ten exposed groups, ten discordant groups, and 50 baseline trades.
  A narrow confidence interval around mostly cash outcomes cannot substitute for this breadth.

The step-4 versus fine utility interval, adjusted for all registered comparisons, is
**[−0.007044, +0.533103]**. Its breadth requirements fail. None of the 44 alternatives satisfies
all evidence requirements; absence of a significant loss is not proof of equivalence.

Outlier control operates on development evidence. Of 4,051 eligible fine-grid candidate profiles
across validation windows, the existing neighbor rule rejects **720 isolated peaks**, leaving
3,331 supported profiles. There are also 56 selected coarse-grid profiles without support in
the fine grid's immediate neighborhood; every one already fails its inner holdout or
full-history gate. Excluding these unsupported selections therefore changes no executed outcome.
Counts across windows can include the same parameter pair repeatedly.

Valid market gains and losses remain in the primary results. Trimming is symmetric and used
only as a sensitivity test; dropping inconvenient losses would bias the comparison. Among the
finest grid's ten exposed tests, four lose money and seven have negative holding-adjusted
utility. Their average full-training Sharpe is 1.076 versus 0.383 in the outer tests. Together
with alignment sensitivity and concentration, this is evidence against trusting the best
historical grid rank; it is not an estimated probability of overfitting.

## Later confirmation

The leader and confirmation grid list were frozen before evaluating 2023–October 8, 2026.
Ten grids cover the same **391 asset-years, 117 products, and 60 proxies**. The 2026 test has
193 sessions and its return is a partial-year return. Newly eligible products enter according
to the same predeclared history rule.

| Grid | Exposed periods | Completed trades | Mean utility | Mean period return |
| --- | ---: | ---: | ---: | ---: |
| Finest, 4,059 pairs | 12 | 47 | −0.011574 | +0.0860% |
| Frozen step-4 candidate, 286 pairs | 4 | 16 | −0.034216 | −0.1997% |
| Step 3, 510 pairs | 6 | 30 | +0.006787 | +0.5681% |
| Step 5, 189 pairs | 2 | 25 | +0.007585 | +0.2861% |
| Endpoint-only, 4 pairs | 0 | 0 | 0 | 0% |
| Incumbent, 49 pairs | 4 | 25 | +0.002821 | +0.3221% |

The frozen step-4 candidate loses its earlier positive utility and underperforms the fine
benchmark on both utility and mean period return. Its utility remains negative at 10 bps per
side: **−0.034506**. Later rankings do not replace the frozen candidate.

For the **same 79 products used in validation**, fine, step 4, and incumbent all remain cash
in every one of the 316 confirmation asset-years. Their later exposures come entirely from
newly eligible products. Thus the later period supplies neither a repeated TQQQ advantage nor
confirmation of a universal grid setting. Confirmation also fails the registered breadth
requirements, and no candidate is promoted.

## Efficiency measurement

The table reports medians of 15 uncached, warmed production selections per grid: three repeats
on each of TTT, SPXU, TBT, SRS, and SSO, with randomized grid order and four Numba threads.
These products were selected at deterministic history-length quantiles before timing. This
sample contains four inverse funds and one long fund.

A separate TQQQ/SQQQ timing check, specified before its measurements, gives **128.8 ms** for
the finest grid and **27.0 ms** for step 4, approximately **4.8× faster**. The incumbent takes
16.0 ms on that pair. Both timing files are preserved. Timings are local wall-clock measurements,
not production service guarantees; fewer pairs do not yield proportional speedups because
validation, price rounding, and other fixed work remain.

The historical sweep shares identical raw development simulations across grids, but recomputes
each grid's own neighborhood scores. Its per-grid cached timing is deliberately excluded from
speedup claims. The validation sweep took about five minutes; the later sweep about 92 seconds,
excluding summary bootstrap work and standalone timings.

## Protocol and limits

The source SQLite database was read using `mode=ro` in one consistent transaction. The retained
input archive contains 121 products passing the initial history screen: 72 long and 49 inverse.
Four of these products qualify for no annual test under the window-specific rules, leaving
117 products with actual test windows. Another 426 current
products lack the minimum 756 price sessions needed for any 630-session training prefix plus
126-session test. Validation has 79 products, 36 proxies, and 756 eligible annual windows
in 2013–2022; the cohort expands only when prior history permits it.

Every outer test starts flat. Selection sees only prior observations and keeps production
version 2, all development/holdout rules, full-history Sharpe ≥ 1, and at least one completed
round trip unchanged. Only the development winner sees the inner holdout. RSI uses causal
earlier signal history. Base costs are 3 bps per side; the 10 bps stress reuses frozen parameters.
Each coarser grid recomputes its own neighbors, so the study measures grid resolution together
with the existing selector's changing neighborhood distances.

Before results, practical deterioration was set at **0.05 utility or one percentage point of
period return**. Noninferiority requires simultaneous lower bounds above both negative margins;
degradation requires an upper bound below a negative margin, with adequate breadth and robust
cost, trimming, and leave-one-proxy-out results. The bootstrap uses 250,000 proxy-cluster
resamples, seed 20261012, and Bonferroni tails for 44 alternatives × four outcome metrics ×
three direction summaries. It preserves the full correction in confirmation.

The cached universe contains current survivors with current signal mappings. It does not
reconstruct delisted products. Proxies can remain correlated; cluster intervals are approximate,
especially with sparse exposure. Annual flat starts do not reproduce daily reoptimization or
positions carried between years. Open holding ages are censored. These periods have been
researched before, so confirmation is retrospective rather than a pristine holdout.

Separating selection from evaluation limits the optimism caused by reusing the same outcomes;
it cannot erase earlier historical inspection or repeated research. See the
[scikit-learn nested selection example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
and [Bailey and colleagues on backtest overfitting](https://www.davidhbailey.com/dhbpapers/overfitting.pdf).

## Artifacts and reproduction

[Study directory](../research/grid-resolution-2026-10-12/) contains the registered protocol,
input arrays and hashes, coverage, all trial rows, summaries, frozen choice, timings, compact
comparison, decision, and verification results. Input metadata and simulation source hashes
are checked before each phase. An altered frozen choice, validation file, protocol, or source
is rejected before confirmation.

Review extended the source guard to the transitive `pricing.py` and `backtest.py` dependencies;
these files were not hash-pinned in the original registration. The
[original provenance](../research/grid-resolution-2026-10-12/provenance.original.json) is preserved
byte-for-byte, and the [correction record](../research/grid-resolution-2026-10-12/provenance_correction.json)
documents the updated runner guard and manifest. The study protocol, simulation sources, frozen
inputs, trial results, and decisions remain unchanged.

Run these commands from a repository checkout after completing the README setup. The saved
study artifacts live in the repository; they are not included in the installed package.
To rerun the exact frozen inputs with the same source versions, copy only preparation artifacts
into an empty directory and run the phases:

```bash
mkdir -p /tmp/grid-resolution-rerun
cp research/grid-resolution-2026-10-12/{protocol.json,provenance.json,inputs.json,inputs.npz,coverage.csv} /tmp/grid-resolution-rerun/
NUMBA_NUM_THREADS=4 .venv/bin/python scripts/research-resolution.py --study-dir /tmp/grid-resolution-rerun --phase validation
NUMBA_NUM_THREADS=4 .venv/bin/python scripts/research-resolution.py --study-dir /tmp/grid-resolution-rerun --phase freeze
NUMBA_NUM_THREADS=4 .venv/bin/python scripts/research-resolution.py --study-dir /tmp/grid-resolution-rerun --phase confirmation
NUMBA_NUM_THREADS=4 .venv/bin/python scripts/research-resolution.py --study-dir /tmp/grid-resolution-rerun --phase benchmark
NUMBA_NUM_THREADS=4 .venv/bin/python scripts/research-resolution.py --study-dir /tmp/grid-resolution-rerun --phase summarize
```

For a new source snapshot, start in a new study directory with `--phase prepare --db PATH`.
Existing simulation phases and frozen choices cannot be overwritten. Summaries may be
recomputed deterministically; timing results naturally vary.

Verification: **90 tests and 98 subtests pass**, new source/test files pass Ruff, and whitespace
checks pass. Direct production selection matches the cached evaluator. Independent replay of
three predetermined asset-years across all 45 grids reproduces **135 rows and 9,045 compared
fields with zero discrepancies**, excluding only wall-clock timing. No trading state or
strategy defaults were changed.
