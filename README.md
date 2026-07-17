# German Electricity Demand — Forecasting the Average Daily Peak

Case study forecasting German electricity demand (OPSD, country code `DE`,
January 2015 – October 2020). The series modelled throughout is the **weekly
mean of daily peak load**: for every calendar day the maximum hourly load in
GW is taken, and the seven daily maxima of each week are averaged. This lens
is peak-oriented — it tracks how hard the grid is stressed on a typical day —
while staying smoother than a single weekly maximum. The final 104 weeks are
held out and forecast by five model families, all judged on the same window.

## Aim

1. Prepare the hourly OPSD load series and derive daily-peak and weekly
   average-daily-peak series; explore components and test stationarity.
2. Benchmark forecasts: mean, naive, seasonal naive, drift (104-week horizon).
3. SARIMA with a full AIC grid search over p ∈ [0,6], d ∈ [0,2], q ∈ [0,6]
   with seasonal block (1,1,1)[52]; residual diagnostics; 95% intervals.
4. SARIMAX with Berlin temperature (weekly mean, HDD base 15.0 °C, CDD base
   21.0 °C) as exogenous regressors — a conditional forecast.
5. Feature-based ML: RandomForest (primary) and ExtraTrees (secondary) on
   lag/rolling/calendar/holiday/temperature features, forecast recursively.
6. LSTM on the hourly series with a layer-design sweep (1 vs 2 stacked
   72-unit layers at two learning rates), then a genuine two-year free-run
   rollout aggregated back through the daily-max-then-weekly-mean lens.
7. Consolidated MAE / RMSE / MASE / Bias table, MASE ratios against the
   seasonal naive, regime diagnostics and a pre-2020 vs 2020 error split.

## Data

- Load: [Open Power System Data, 60-min time series (2020-10-06)](https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv),
  column `DE_load_actual_entsoe_transparency`. Auto-downloaded by the
  notebook if `time_series_60min_singleindex.csv` is not present.
- Temperature: [Open-Meteo archive API](https://archive-api.open-meteo.com/v1/archive),
  daily mean 2-m temperature for Berlin, cached to
  `berlin_daily_temperature.csv`. If the API is unreachable a clearly
  labelled synthetic curve substitutes so the notebook still runs.

Preprocessing: slice from 2015-01-01 on the UTC index first, convert to
Europe/Berlin, interpolate short gaps only, convert MW → GW. Weeks are kept
only when all 7 daily peaks are present.

## How to run

Open `Ganesh.ipynb` in Google Colab (or Jupyter) and Run All. The
first cell installs any missing packages; everything else is self-contained.
Keep `FAST_RUN = False` for a submission run — the full 147-order SARIMA
screening takes roughly 30–45 minutes on a multi-core runtime and prints
elapsed/ETA banners as it goes. The two-year LSTM rollout (~17,000 compiled
graph steps) takes a few minutes.

```
pip install -r requirements.txt
```

## Models

| Family | Model | Notes |
|---|---|---|
| Benchmarks | Mean, Naive, Seasonal naive, Drift | seasonal naive is the yardstick |
| Statistical | SARIMA (AIC grid, (1,1,1)[52] seasonal) | 95% intervals, residual diagnostics |
| Statistical + exog | SARIMAX + temp/HDD/CDD | conditional forecast (observed weather) |
| Feature ML | RandomForest, ExtraTrees | recursive 104-week rollout, importances |
| Neural | LSTM (depth sweep, look-back 168 h) | free-run rollout, train-only scaling |

## Evaluation

MAE, RMSE, MASE and Bias on the identical 104-week test window. The MASE
denominator is the in-sample seasonal-naive MAE, so MASE < 1 beats a
same-week-last-year rule. Regime diagnostics cover festive ISO weeks
(52, 1, 2), the five warmest and five coldest test weeks, and a pre-2020
versus 2020 split that isolates the COVID demand break.

## Leakage protections

- Target lags and rolling means computed from history strictly before the
  target week; lagged temperature shifted by one week.
- Recursive rollouts append predictions, never held-out actuals.
- LSTM scaling statistics fitted on training hours only.
- Same-week temperature in SARIMAX/ML is deliberate and labelled: those are
  conditional (explanatory) forecasts, not operational ones.

## Outputs

Created at run time under `outputs/`:

- `figures/model_comparison_master.png` — all models vs actual.
- `forecasts/weekly_forecasts.csv` — each model's 104-week path.
- `metrics/model_scores.csv` — consolidated metric table.

## Files

```
Ganesh.ipynb   the analysis notebook (Rounds 1–9)
README.md             this file
requirements.txt      package list
outputs/              figures, forecasts, metrics (run time)
```
