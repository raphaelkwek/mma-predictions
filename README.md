# mma-predictions

Predicting UFC fight outcomes from **stylistic matchups**, evaluated against the betting market.

## Thesis

Fights are stylistic. A wrestler's takedown rate matters less than how it meets *this* opponent's
takedown defense; a distance striker's volume matters less than whether the opponent closes range.
This project builds explicit style profiles for each fighter and models the **interaction** between
them, then asks whether that beats a strong baseline — and whether it beats the closing line.

## Goal

Beat the market on UFC moneylines, with method-of-victory (KO/TKO · submission · decision) as a
secondary target. The benchmark is not 50/50 — it is the de-vigged closing price. Log-loss against
the market, not raw accuracy, is the scoreboard.

## Status

Design phase. See `docs/superpowers/specs/` for the design document.

## Layout

```
data/          downloaded source files and parsed parquet tables (gitignored)
src/mma/       the package: data, parsers, features, models, backtest
notebooks/     thin exploration and results notebooks (call the package, never redefine it)
tests/         parser fixtures, leakage assertions, backtest accounting
dashboard/     Streamlit app
docs/          design docs and specs
```

## Disclaimer

Research and educational project. Nothing here is betting advice.
