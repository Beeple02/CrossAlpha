# CrossAlpha

CrossAlpha is a full v1 research-to-production scaffold for a long-only, cross-sectional stock recommendation system across five horizons:

- `1d`
- `1w`
- `2w`
- `1m`
- `2m`

The system is built around point-in-time data handling, survivorship-bias-aware universe construction, purged walk-forward validation, and a recommendation layer that emits one of:

- `BUY`
- `NO_BUY`
- `NOT_ENOUGH_DATA`

## What This Repository Builds

- Modular data ingestion adapters for free data sources
- Local artifact storage with Parquet outputs
- Historical S&P 500 universe reconstruction with reliability flags
- Feature engineering for price, volume, volatility, relative strength, sector, earnings, fundamentals, and regime context
- Label generation for all five horizons from day one
- Separate baseline and main model workflows per horizon
- Purged walk-forward validation with embargo
- Daily recommendation generation
- Long-only backtesting net of friction

## Repository Layout

```text
CrossAlpha/
  configs/
  data/
  reports/
  scripts/
  src/crossalpha/
  tests/
```

## Install

Create a virtual environment and install the project:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

For Google Colab, use [requirements-colab.txt](C:\Users\lucia\Documents\Codex\2026-05-03-files-mentioned-by-the-user-crossalpha\requirements-colab.txt) and see [COLAB_SETUP.md](C:\Users\lucia\Documents\Codex\2026-05-03-files-mentioned-by-the-user-crossalpha\COLAB_SETUP.md).

Before running live ingestion against Wikipedia and the SEC, set `CROSSALPHA_USER_AGENT` to something descriptive with contact info, for example `CrossAlpha Research your_email@example.com`.

## End-to-End Commands

The main CLI is `crossalpha`.

```powershell
crossalpha ingest --config configs/base.toml
crossalpha features --config configs/base.toml
crossalpha labels --config configs/base.toml
crossalpha validate --config configs/base.toml
crossalpha train --config configs/base.toml
crossalpha recommend --config configs/base.toml
crossalpha backtest --config configs/base.toml
```

Run the full pipeline:

```powershell
crossalpha run-all --config configs/base.toml
```

`configs/data_sources.toml` documents the default adapter choices and is the intended place to swap vendors later.
`configs/colab.toml` is the ready-to-use Colab configuration that writes artifacts to Google Drive.

## Main Artifacts

- `data/raw/`: vendor-normalized raw pulls
- `data/processed/`: cleaned prices, universe, features, labels, and recommendation-ready tables
- `data/models/`: serialized model artifacts per horizon
- `reports/output/`: validation summaries, feature metadata, backtest metrics, and recommendations

## Example Recommendation Schema

Each row in the final recommendation output contains:

```text
date
ticker
horizon
score
rank
decision
confidence
regime_state
not_enough_data_reason
```

## Assumptions

- Historical S&P 500 membership is reconstructed from the Wikipedia constituent table plus the published change log on the same page.
- Point-in-time fundamentals are sourced from SEC company facts using filing dates when available.
- When free-source history is incomplete, the system marks observations as unreliable and emits `NOT_ENOUGH_DATA` rather than silently widening assumptions.
- Daily recommendations are backtested with daily rebalancing and next-day mark-to-market returns, even for longer-horizon models.

## Module Summary

- `crossalpha.data`: ingestion adapters, storage, quality rules, and universe construction
- `crossalpha.features`: feature catalog and feature engineering
- `crossalpha.labels`: forward-return and ranking labels
- `crossalpha.models`: baseline and main model wrappers, training, persistence
- `crossalpha.validation`: purged walk-forward splits and metrics
- `crossalpha.engine`: recommendation scoring and decision rules
- `crossalpha.backtest`: daily long-only strategy simulation with friction
- `crossalpha.reports`: report writers for metrics and outputs

## Known External Setup Needs

- Internet access is required to run the free-source ingestion steps.
- `pyarrow`, `scikit-learn`, `lightgbm`, `yfinance`, and `lxml` must be installed.
- On Python 3.10, `tomli` is also required.
- SEC access works best with a descriptive `User-Agent` environment variable for request headers.
- Wikipedia access is also more reliable when `CROSSALPHA_USER_AGENT` is set explicitly.

## Exact Next Step

After installing dependencies, run:

```powershell
crossalpha run-all --config configs/base.toml
```
