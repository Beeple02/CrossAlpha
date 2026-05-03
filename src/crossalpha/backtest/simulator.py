"""Long-only recommendation backtester."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from crossalpha.config import ProjectConfig
from crossalpha.data.storage import DataCatalog
from crossalpha.engine.recommender import apply_recommendation_logic
from crossalpha.utils.io import read_parquet, write_json, write_parquet
from crossalpha.utils.math import max_drawdown


LOGGER = logging.getLogger(__name__)


def run_backtest(cfg: ProjectConfig) -> dict[str, object]:
    catalog = DataCatalog(cfg)
    LOGGER.info("Running backtest from validation predictions.")

    predictions = read_parquet(catalog.processed("validation_predictions"))
    features = read_parquet(catalog.processed("feature_store"))
    prices = read_parquet(catalog.processed("prices_cleaned"))
    benchmark = read_parquet(catalog.processed("benchmark"))

    scored = predictions.merge(
        features[[
            "date",
            "ticker",
            "sector",
            "industry",
            "insufficient_price_history",
            "insufficient_post_listing_history",
            "stale_price",
            "missing_gap_fail",
            "feature_missing_fail",
            "fundamentals_missing_fail",
            "universe_history_fail",
            "vix_close",
            "spx_above_200d",
        ]],
        on=["date", "ticker"],
        how="left",
    )
    recommendations = apply_recommendation_logic(scored, cfg)
    write_parquet(recommendations, catalog.processed("historical_recommendations"))

    price_returns = _next_day_returns(prices)
    benchmark_returns = _benchmark_returns(benchmark, cfg)

    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []

    for horizon in cfg.horizons:
        horizon_recs = recommendations[recommendations["horizon"] == horizon].copy()
        horizon_daily = _simulate_horizon(horizon_recs, price_returns, benchmark_returns, cfg)
        if horizon_daily.empty:
            continue

        horizon_daily["horizon"] = horizon
        daily_frames.append(horizon_daily)
        summary_rows.append(_summarize_horizon(horizon, horizon_daily, horizon_recs))

        by_regime = horizon_daily.groupby("regime_state", observed=True).agg(
            mean_net_return=("net_return", "mean"),
            mean_turnover=("turnover", "mean"),
            days=("date", "count"),
        ).reset_index()
        by_regime["horizon"] = horizon
        regime_rows.extend(by_regime.to_dict(orient="records"))

    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    summary = {"rows": summary_rows}

    if not daily.empty:
        write_parquet(daily, catalog.report("backtest_daily.parquet"))
    write_json(summary, catalog.report("backtest_summary.json"))
    write_json({"rows": regime_rows}, catalog.report("backtest_by_regime.json"))
    LOGGER.info("Backtest complete.")
    return summary


def _next_day_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy().sort_values(["ticker", "date"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices["next_day_return"] = prices.groupby("ticker")["adj_close"].pct_change().shift(-1)
    return prices[["date", "ticker", "next_day_return"]]


def _benchmark_returns(benchmark: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    frame = benchmark[benchmark["ticker"] == cfg.data.benchmark_symbol].copy().sort_values("date")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["benchmark_return"] = frame["adj_close"].pct_change().shift(-1)
    return frame[["date", "benchmark_return"]]


def _simulate_horizon(
    recommendations: pd.DataFrame,
    price_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    cfg: ProjectConfig,
) -> pd.DataFrame:
    frame = recommendations.merge(price_returns, on=["date", "ticker"], how="left")
    frame = frame.merge(benchmark_returns, on="date", how="left")
    frame = frame.sort_values(["date", "rank"], na_position="last")

    prev_weights = pd.Series(dtype=float)
    portfolio_value = cfg.backtest.initial_capital
    rows: list[dict[str, object]] = []

    for date, day in frame.groupby("date", sort=True):
        day = day.copy()
        buy_names = day[day["decision"] == "BUY"]["ticker"].tolist()
        if buy_names:
            weight = 1.0 / len(buy_names)
            target_weights = pd.Series(weight, index=buy_names, dtype=float)
        else:
            target_weights = pd.Series(dtype=float)

        aligned_index = sorted(set(prev_weights.index).union(target_weights.index))
        prev_aligned = prev_weights.reindex(aligned_index).fillna(0.0)
        target_aligned = target_weights.reindex(aligned_index).fillna(0.0)
        turnover = float((target_aligned - prev_aligned).abs().sum())

        day_returns = day.set_index("ticker")["next_day_return"].reindex(aligned_index).fillna(0.0)
        gross_return = float((target_aligned * day_returns).sum())
        cost = turnover * (cfg.backtest.transaction_cost_bps + cfg.backtest.slippage_bps) / 10_000.0
        net_return = gross_return - cost
        portfolio_value *= 1.0 + net_return

        post_return_weights = target_aligned * (1.0 + day_returns)
        if post_return_weights.sum() > 0:
            prev_weights = post_return_weights / post_return_weights.sum()
        else:
            prev_weights = pd.Series(dtype=float)

        rows.append({
            "date": date,
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
            "cost": cost,
            "benchmark_return": float(day["benchmark_return"].iloc[0]) if day["benchmark_return"].notna().any() else np.nan,
            "regime_state": day["regime_state"].iloc[0],
            "buy_count": len(buy_names),
            "portfolio_value": portfolio_value,
        })

    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily
    daily["cumulative_return"] = (1.0 + daily["net_return"]).cumprod()
    daily["benchmark_cumulative_return"] = (1.0 + daily["benchmark_return"].fillna(0.0)).cumprod()
    daily["relative_return"] = daily["net_return"] - daily["benchmark_return"].fillna(0.0)
    return daily


def _summarize_horizon(horizon: str, daily: pd.DataFrame, recommendations: pd.DataFrame) -> dict[str, object]:
    buy_rows = recommendations[recommendations["decision"] == "BUY"]
    annualized_vol = float(daily["net_return"].std(ddof=0) * np.sqrt(252))
    sharpe = float(daily["net_return"].mean() / daily["net_return"].std(ddof=0) * np.sqrt(252)) if daily["net_return"].std(ddof=0) > 0 else np.nan
    return {
        "horizon": horizon,
        "days": int(len(daily)),
        "total_return": float(daily["cumulative_return"].iloc[-1] - 1.0),
        "benchmark_total_return": float(daily["benchmark_cumulative_return"].iloc[-1] - 1.0),
        "benchmark_relative_return": float((daily["cumulative_return"].iloc[-1] - 1.0) - (daily["benchmark_cumulative_return"].iloc[-1] - 1.0)),
        "precision_buy": float(buy_rows["target_class"].mean()) if not buy_rows.empty and "target_class" in buy_rows else np.nan,
        "hit_rate": float((buy_rows["forward_return"] > buy_rows["daily_median_forward_return"]).mean()) if not buy_rows.empty and "forward_return" in buy_rows else np.nan,
        "average_forward_return_buy": float(buy_rows["forward_return"].mean()) if not buy_rows.empty and "forward_return" in buy_rows else np.nan,
        "turnover_mean": float(daily["turnover"].mean()),
        "max_drawdown": max_drawdown(daily["cumulative_return"]),
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
    }
