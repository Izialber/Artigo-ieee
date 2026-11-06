# Day-ahead load forecasting under rapid behind-the-meter solar growth

Code and manuscript for the article *"Day-Ahead Load Forecasting Under Rapid
Behind-the-Meter Solar Growth: An Open Benchmark Against a System Operator's
Official Forecast in Brazil"* (submitted to IEEE Access).

Everything in the paper is reproduced from public data of the Brazilian
National System Operator (ONS):

| Data | Source | Period |
|---|---|---|
| Hourly load per subsystem (`curva-carga-ho`) | ONS open-data portal (S3 bucket `ons-aws-prod-opendata`) | 2012-2026 |
| Verified global load and estimated behind-the-meter generation (MMGD) | ONS load API, `cargaverificada` | 2020-2026 |
| Official day-ahead programmed load | ONS load API, `cargaprogramada` | 2021-2026 |

## Reproducing the results

```bash
pip install -r requirements.txt
./run_experiments.sh          # download data, tune, run all experiments, build tables/figures
cd paper && latexmk -pdf main.tex
```

The full run takes roughly 3 hours on a 4-core machine. Predictions are cached in
`results/preds/`, so an interrupted run resumes where it stopped.

## Layout

| Path | Content |
|---|---|
| `src/data.py` | download and cleaning of the ONS hourly load curve |
| `src/ons_api.py` | official forecast and verified global load from the ONS load API |
| `src/features.py` | Brazilian holiday calendar and day-ahead features (issuance at D-1 12:00) |
| `src/models.py` | SN7, SN-DT, ARX, GBM (LightGBM), MLP and ensemble |
| `src/backtest.py` | rolling-origin backtesting engine |
| `src/tune.py` | hyper-parameter selection on the 2023 validation year |
| `src/experiments.py` | all experiments (E1-E6) |
| `src/prob.py` | empirical-quantile and adaptive conformal prediction intervals |
| `src/metrics.py` | accuracy metrics, Diebold-Mariano test, Holm correction |
| `src/analysis.py` | tables (`paper/tables/`) and figures (`paper/figures/`) |
| `paper/` | LaTeX manuscript (IEEE Access template) |

## License

Code: MIT. ONS data are subject to the ONS open-data terms of use.
