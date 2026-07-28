# UFC Outcome Prediction from Stylistic Matchups — Design

**Date:** 2026-07-29
**Status:** Approved, pending implementation plan
**Repo:** https://github.com/raphaelkwek/mma-predictions

## Problem

Predict UFC fight outcomes well enough to beat the betting market on moneylines, with
method of victory as a secondary target.

The working hypothesis is that **fights are stylistic**: outcomes depend less on each
fighter's absolute quality than on how their tendencies interact. A wrestler's takedown
rate matters only relative to this opponent's takedown defense. A volume striker's output
matters only if the opponent lets the fight stay at range. If that hypothesis holds,
features encoding the *interaction* between two style profiles should predict better than
features describing each fighter in isolation.

## Success criteria

Ranked, and deliberately strict:

1. **Primary.** Beat the de-vigged closing line on log-loss over a chronologically
   held-out test period.
2. **Secondary.** If (1) holds, positive ROI in a walk-forward betting simulation whose
   bootstrap confidence interval on ROI excludes zero.
3. **Thesis.** Matchup-interaction features measurably improve log-loss over the same
   model with per-fighter style features only.

Raw accuracy is reported but never optimized. Picking every favorite scores roughly 65%
and loses money after vig.

"The market is efficient and no edge was found" is an acceptable, publishable outcome.
The design must be capable of returning that answer rather than manufacturing a positive
one — hence the confidence intervals in criterion (2).

## Non-goals

- Live/in-play betting, or line-shopping across books.
- Promotions other than the UFC.
- Round-level or prop markets beyond method of victory, unless historical prop lines
  turn out to be available (audited in Phase 1, not assumed).
- Automated bet placement. The output is probabilities and analysis.

## Architecture

Five stages, each writing a persisted artifact so any stage can be re-run without
redoing its predecessor:

```
scrape → parse → canonical tables → as-of features → models → evaluation/backtest
                                                                     ↓
                                                               dashboard
```

Raw HTML is cached to disk before parsing. This is not premature optimization: parser
bugs are discovered late (a mis-parsed control-time field, a method string that only
appears twice in the corpus), and re-parsing a local cache takes seconds where
re-scraping takes hours and hammers someone else's server.

### Repository layout

```
data/            raw HTML cache, parsed parquet tables (gitignored)
src/             the package
  scrape/        ufcstats fetchers, rate limiting, cache
  parse/         HTML → tabular
  canonical/     schema definitions, entity resolution, odds join
  features/      as-of feature builders (baseline, style, interaction, archetype)
  models/        training, calibration, method-of-victory
  backtest/      walk-forward harness, staking, bootstrap CIs
notebooks/       thin exploration and results; call the package, never redefine it
tests/           parser fixtures, leakage assertions, backtest accounting
dashboard/       Streamlit app
docs/            design docs and specs
```

## Data layer

### Sources

- **Fight stats and fighter attributes:** scraped from ufcstats.com. Free, canonical,
  and the only public source with per-round positional strike breakdowns. Scraping is
  sequential, rate-limited, cached, and uses an identifying user-agent.
- **Historical closing moneylines:** bootstrapped from a public dataset (Kaggle or
  equivalent). Identifying the specific dataset is a Phase 1 task with acceptance
  criteria below, not an assumption of this design.

### Canonical schema

**`fighters`** — fighter_id, name, height, reach, weight, stance, date of birth.

**`fights`** — fight_id, event_id, date, weight class, scheduled rounds, title bout flag,
fighter_a_id, fighter_b_id, winner_id, method (KO/TKO · SUB · U-DEC · S-DEC · M-DEC · DQ ·
NC), end round, end time.

**`fight_stats`** — long format, one row per fight_id × fighter_id × round × stat:
knockdowns; significant strikes landed and attempted, split by **target** (head/body/leg)
and by **position** (distance/clinch/ground); total strikes; takedowns landed and
attempted; submission attempts; reversals; control time.

The position and target breakdowns are the raw material for style. They are what separate
a distance point-fighter from a clinch grinder from a top-control grappler, and no other
field in the dataset carries that information.

**`odds`** — fight_id, fighter_id, closing moneyline, source.

### Entity resolution and the coverage gate

Odds data is keyed by fighter name and event date; UFCStats is keyed by its own ids.
Names disagree systematically — diacritics (José/Jose), suffixes (Jr./Sr.), nicknames
embedded in the name field, and inconsistent ordering for Brazilian and Korean fighters.
The join therefore needs normalization plus fuzzy matching constrained by event date and
weight class, with ambiguous matches surfaced for manual review rather than silently
resolved.

**Phase 1 exit gate.** Produce a coverage report: the fraction of fights from 2010 onward
with a matched closing line, plus a breakdown of *unmatched* fights by year, card position,
and weight class.

- **≥80% coverage on 2010+, with unmatched fights showing no systematic bias** → proceed
  as designed.
- **Below that, or unmatched fights concentrated in prelims or a particular era** →
  stop and reconsider before building anything that depends on the odds. Biased coverage
  is worse than sparse coverage: if only main-card fights match, the backtest silently
  evaluates on the market's sharpest subset.

## Leakage discipline

This is the central engineering constraint. Most published UFC models report accuracy
inflated by lookahead, usually through career-total statistics that include the fight
being predicted.

Rules, enforced by tests rather than by care:

1. **As-of feature construction.** For a fight on date D, every feature aggregates only
   that fighter's bouts strictly before D. The feature builder takes the fight date as an
   explicit argument; there is no code path that can see future rows.
2. **Chronological splits only.** No random k-fold anywhere in the project. Train on
   ≤ T, validate on T..T', test after T'. The backtest walks forward.
3. **Archetype clusters are fit on training-period data and frozen** before any test-period
   prediction.
4. **Cold start is explicit.** Debut fighters have no history. They get physicals plus
   weight-class priors and an `is_debut` flag — never zero-imputed style stats, which a
   model reads as "throws no strikes and defends no takedowns."

Tests: an assertion that every source row feeding a computed feature carries a date earlier
than the fight's, and a golden-value test on a hand-verified fight.

## Features

Four blocks, layered so each can be disabled independently for the ablation.

**1. Baseline.** Elo rating updated chronologically; win/loss record; age at fight; days
since last fight; height, reach, weight class; stance; title bout; scheduled rounds;
debut flag.

**2. Style profile** — per fighter, as of the fight date. Strike volume, accuracy, strikes
absorbed, striking defense; distance/clinch/ground strike mix; head/body/leg mix; takedown
attempt rate, accuracy, defense; submission attempts per 15 minutes; control-time share;
knockdown rate; finish rate; average fight duration.

Each is computed two ways: **career to date** and a **recency-weighted recent window**.
Fighters change. A wrestler who reinvented himself as a striker three fights ago should
read as a striker.

**3. Matchup interactions** — the thesis, made explicit. Not two separate columns in the
hope that the model discovers their product, but the product itself: A's takedown rate ×
B's takedown defense; A's grappling threat vs B's ground defense; A's distance preference ×
B's pressure; southpaw/orthodox mismatch; reach advantage × distance-fighting tendency.

Gradient-boosted trees *can* represent interactions, but with roughly 8,000 fights they
learn them poorly. Supplying them directly is worth more than model capacity.

**4. Archetypes.** Cluster fighters on the standardized style vector into approximately
5–7 archetypes (pressure striker, counter-striker, wrestle-boxer, submission grappler,
scrambler, and similar). Features: each fighter's archetype, plus the historical win rate
of archetype *i* against archetype *j*, shrunk toward 0.50 where cells are thin. With 6
archetypes there are 21 unordered matchup cells against a dataset split by era and weight
class, so shrinkage is required, not optional.

### Corner symmetry

Fighter A/B assignment is arbitrary, and in the source data it correlates with being the
favorite. Left uncorrected, the model learns corner assignment instead of fighting ability.

Mitigations, all three: features are built as antisymmetric differences wherever possible;
training data includes both orderings with mirrored labels; inference averages
`P(A beats B)` with `1 − P(B beats A)`. A test asserts those two sum to 1.

## Models

In ascending order of ambition:

- **Benchmark — the market.** Closing odds converted to probabilities with the vig
  removed. Every model is scored against this. It is a strong opponent.
- **Elo-only logistic regression.** Sanity floor.
- **Regularized logistic regression** on the full feature set. Interpretable coefficients,
  naturally well-calibrated, and genuinely competitive at this data size.
- **LightGBM.** Usually the strongest on tabular data of this shape.

**Calibration is mandatory.** Isotonic regression fit on a held-out time slice, with
reliability diagrams reported. Betting decisions are made on probabilities, not rankings;
a model can rank fights correctly and still be systematically overconfident, which loses
money while appearing to be right.

**Method of victory** is two-stage: `P(winner)`, then `P(method | winner)` over three
classes (KO/TKO, submission, decision). Cleaner than a six-class joint model, and it lets
the style features do their most natural work — a grappler-versus-striker matchup should
move submission probability far more than it moves win probability.

Method is evaluated on log-loss and accuracy. It is evaluated on ROI only if Phase 1
turns up historical method or inside-the-distance lines.

## Evaluation

**Primary.** Log-loss on the chronological test period, against the de-vigged closing line
on the same fights. Brier score and reliability diagrams alongside.

**Betting simulation**, run only if the model beats the market on log-loss. Walk-forward:
train on everything before date D, bet card D, roll forward. Flat stakes and fractional
Kelly, an EV threshold, realistic vig. Reports ROI, maximum drawdown, and **bootstrap
confidence intervals on ROI**.

The confidence interval is load-bearing. Over a few thousand test fights, a strategy with
no true edge routinely posts +8% ROI through variance alone. Without a CI that result is
indistinguishable from skill, and the project would conclude the opposite of the truth.

**Thesis ablation.** Each rung adds one feature block:

| Model | Adds |
|---|---|
| Elo only | strength prior |
| + physicals & record | conventional baseline |
| + style profiles | individual tendencies |
| + matchup interactions | **the thesis** |
| + archetypes | style clusters |

If interactions do not improve log-loss over style profiles alone, style is a property of
*fighters* rather than *matchups*. That is a genuine and interesting negative result, and
the ladder is what makes it defensible.

## Testing

Concentrated where failures are silent rather than loud. A broken scraper announces
itself; a leaking feature looks like success.

- Parser tests against saved HTML fixtures, including known-awkward fights (no contest,
  DQ, overturned result, five-round main event).
- As-of leakage assertions and a golden-value feature test.
- Corner symmetry: `P(A beats B) + P(B beats A) = 1`.
- Elo update correctness against hand-computed values.
- Backtest accounting: American odds → implied probability → payout, for favorites and
  underdogs, including the vig-removal step.

## Stack

Python 3.12 (conda), pandas, scikit-learn, LightGBM, httpx, BeautifulSoup, parquet,
pytest, Streamlit, Plotly.

## Phases

Each phase ends in a reviewable artifact.

1. **Data and coverage audit.** Scrapers, parsers, canonical tables, odds join, coverage
   report. **Gate:** odds coverage acceptable per the criteria above.
2. **Baseline model.** Elo and physicals, as-of feature builder, walk-forward harness,
   market benchmark wired in, leakage tests passing.
3. **Style, interactions, archetypes.** The full ablation ladder.
4. **Method of victory.** Two-stage model.
5. **Betting simulation.** Staking, bootstrap CIs — the actual verdict on criteria 1 and 2.
6. **Dashboard.** Fighter profiles with style radars, archetype map, matchup explorer
   showing which interactions drove a prediction, backtest results, upcoming card.
7. **Later — learned embeddings.** Latent per-fighter style vectors, deferred until there
   is a validated pipeline to compare against. Excluded from initial scope because roughly
   8,000 fights with a long tail of fighters holding one to three UFC bouts makes
   cold-start and overfitting severe.

## Decisions deliberately deferred

- The specific odds dataset, pending the Phase 1 audit.
- Whether method-of-victory gets a betting evaluation, pending prop-line availability.
- Opponent-adjusted style statistics (correcting for strength of schedule). A real
  improvement — a 60% takedown defense against journeymen is not the same as against
  contenders — but it is an iteration on Phase 3, not a prerequisite.
