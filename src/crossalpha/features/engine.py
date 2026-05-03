"""Feature engineering pipeline."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from crossalpha.config import ProjectConfig
from crossalpha.data.storage import DataCatalog
from crossalpha.features.catalog import feature_catalog_records
from crossalpha.utils.io import read_parquet, write_json, write_parquet
from crossalpha.utils.math import rolling_slope, zscore_series


LOGGER = logging.getLogger(__name__)


def build_feature_store(cfg: ProjectConfig) -> pd.DataFrame:
    catalog = DataCatalog(cfg)
    LOGGER.info("Building feature store.")

    prices = read_parquet(catalog.processed("prices_cleaned"))
    quality = read_parquet(catalog.processed("price_quality"))
    universe = read_parquet(catalog.processed("daily_universe"))
    benchmark = read_parquet(catalog.processed("benchmark"))
    vix = read_parquet(catalog.processed("vix"))
    risk_free = read_parquet(catalog.processed("risk_free"))
    earnings = read_parquet(catalog.processed("earnings"))
    fundamentals = read_parquet(catalog.processed("fundamentals"))

    feature_frame = _build_feature_frame(
        prices=prices,
        quality=quality,
        universe=universe,
        benchmark=benchmark,
        vix=vix,
        risk_free=risk_free,
        earnings=earnings,
        fundamentals=fundamentals,
        cfg=cfg,
    )

    sector_columns = sorted(column for column in feature_frame.columns if column.startswith("sector_"))
    write_parquet(feature_frame, catalog.processed("feature_store"))
    write_json(feature_catalog_records(sector_columns), catalog.report("feature_catalog.json"))

    LOGGER.info("Feature store complete with %s rows and %s columns.", len(feature_frame), len(feature_frame.columns))
    return feature_frame


def _build_feature_frame(
    prices: pd.DataFrame,
    quality: pd.DataFrame,
    universe: pd.DataFrame,
    benchmark: pd.DataFrame,
    vix: pd.DataFrame,
    risk_free: pd.DataFrame,
    earnings: pd.DataFrame,
    fundamentals: pd.DataFrame,
    cfg: ProjectConfig,
) -> pd.DataFrame:
    base = prices.merge(quality, on=["date", "ticker"], how="left").merge(
        universe[universe["is_member"]].copy(),
        on=["date", "ticker"],
        how="inner",
    )
    base = base.sort_values(["ticker", "date"]).reset_index(drop=True)

    macro = _prepare_macro_frame(benchmark, vix, risk_free, cfg)
    base = base.merge(macro, on="date", how="left")
    feature_frame = _compute_security_features(base)
    feature_frame = _add_sector_context(feature_frame)
    feature_frame = _add_earnings_features(feature_frame, earnings)
    feature_frame = _add_fundamental_features(feature_frame, fundamentals, cfg)
    feature_frame = _finalize_features(feature_frame, cfg)
    return feature_frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def _prepare_macro_frame(
    benchmark: pd.DataFrame,
    vix: pd.DataFrame,
    risk_free: pd.DataFrame,
    cfg: ProjectConfig,
) -> pd.DataFrame:
    bench = benchmark[benchmark["ticker"] == cfg.data.benchmark_symbol].copy().sort_values("date")
    bench["date"] = pd.to_datetime(bench["date"])
    bench["spx_ret_1d"] = bench["adj_close"].pct_change()
    bench["spx_ret_5d"] = bench["adj_close"].pct_change(5)
    bench["spx_ret_21d"] = bench["adj_close"].pct_change(21)
    bench["spx_ret_63d"] = bench["adj_close"].pct_change(63)
    bench["spx_sma_200"] = bench["adj_close"].rolling(200).mean()
    bench["spx_above_200d"] = (bench["adj_close"] > bench["spx_sma_200"]).astype(float)
    bench = bench.rename(columns={"adj_close": "spx_close"})

    vix = vix[vix["ticker"] == cfg.data.vix_symbol].copy().sort_values("date")
    vix["date"] = pd.to_datetime(vix["date"])
    vix["vix_close"] = vix["adj_close"]
    vix["vix_change_5d"] = vix["adj_close"].pct_change(5)

    risk_free = risk_free.copy().sort_values("date")
    risk_free["date"] = pd.to_datetime(risk_free["date"])
    risk_free["risk_free_rate"] = pd.to_numeric(risk_free["value"], errors="coerce") / 100.0
    risk_free["risk_free_rate"] = risk_free["risk_free_rate"].ffill()

    macro = bench[["date", "spx_close", "spx_ret_1d", "spx_ret_5d", "spx_ret_21d", "spx_ret_63d", "spx_above_200d"]].merge(
        vix[["date", "vix_close", "vix_change_5d"]],
        on="date",
        how="left",
    ).merge(
        risk_free[["date", "risk_free_rate"]],
        on="date",
        how="left",
    )
    return macro.sort_values("date").reset_index(drop=True)


def _compute_security_features(base: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker, frame in base.groupby("ticker", sort=False):
        frame = frame.sort_values("date").copy()
        frame["ret_1d"] = frame["adj_close"].pct_change()
        frame["ret_5d"] = frame["adj_close"].pct_change(5)
        frame["ret_21d"] = frame["adj_close"].pct_change(21)
        frame["ret_63d"] = frame["adj_close"].pct_change(63)
        frame["ret_126d"] = frame["adj_close"].pct_change(126)
        frame["ret_252d"] = frame["adj_close"].pct_change(252)
        frame["momentum_12_1"] = frame["ret_252d"] - frame["ret_21d"]

        sma20 = frame["adj_close"].rolling(20).mean()
        sma50 = frame["adj_close"].rolling(50).mean()
        frame["dist_sma_20"] = frame["adj_close"] / sma20 - 1.0
        frame["dist_sma_50"] = frame["adj_close"] / sma50 - 1.0
        frame["rsi_14"] = _rsi(frame["adj_close"], 14)

        ret_5d_mean_63 = frame["ret_5d"].rolling(63).mean()
        ret_5d_std_63 = frame["ret_5d"].rolling(63).std()
        frame["ret_5d_z_63d"] = (frame["ret_5d"] - ret_5d_mean_63) / ret_5d_std_63

        frame["realized_vol_21d"] = frame["ret_1d"].rolling(21).std() * math.sqrt(252)
        frame["realized_vol_63d"] = frame["ret_1d"].rolling(63).std() * math.sqrt(252)
        frame["vol_ratio_21_63"] = frame["realized_vol_21d"] / frame["realized_vol_63d"]

        prev_close = frame["adj_close"].shift(1)
        true_range = pd.concat([
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        frame["atr_14_close"] = true_range.rolling(14).mean() / frame["close"]
        frame["intraday_range"] = (frame["high"] - frame["low"]) / frame["close"]

        vol5 = frame["volume"].rolling(5).mean()
        vol20 = frame["volume"].rolling(20).mean()
        vol63 = frame["volume"].rolling(63).mean()
        frame["volume_ratio_5_20"] = vol5 / vol20
        frame["volume_ratio_5_63"] = vol5 / vol63
        frame["dollar_volume_20d"] = (frame["adj_close"] * frame["volume"]).rolling(20).mean()
        frame["volume_trend_21d"] = rolling_slope(np.log1p(frame["volume"]), 21)

        frame["rel_strength_5d"] = frame["ret_5d"] - frame["spx_ret_5d"]
        frame["rel_strength_21d"] = frame["ret_21d"] - frame["spx_ret_21d"]
        frame["rel_strength_63d"] = frame["ret_63d"] - frame["spx_ret_63d"]

        cov = frame["ret_1d"].rolling(63).cov(frame["spx_ret_1d"])
        var = frame["spx_ret_1d"].rolling(63).var()
        frame["beta_63d"] = cov / var
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _add_sector_context(feature_frame: pd.DataFrame) -> pd.DataFrame:
    sector_means = feature_frame.groupby(["date", "sector"], observed=True).agg({
        "ret_1d": "mean",
        "ret_21d": "mean",
        "realized_vol_21d": "mean",
    }).rename(columns={
        "ret_1d": "sector_mean_ret_1d",
        "ret_21d": "sector_mean_ret_21d",
        "realized_vol_21d": "sector_mean_vol_21d",
    }).reset_index()

    feature_frame = feature_frame.merge(sector_means, on=["date", "sector"], how="left")
    feature_frame["sector_rel_ret_1d"] = feature_frame["ret_1d"] - feature_frame["sector_mean_ret_1d"]
    feature_frame["sector_rel_ret_21d"] = feature_frame["ret_21d"] - feature_frame["sector_mean_ret_21d"]
    feature_frame["sector_vol_gap_21d"] = feature_frame["realized_vol_21d"] - feature_frame["sector_mean_vol_21d"]

    dummies = pd.get_dummies(feature_frame["sector"], prefix="sector", dtype=float)
    feature_frame = pd.concat([feature_frame, dummies], axis=1)
    return feature_frame


def _add_earnings_features(feature_frame: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    earnings = earnings.copy()
    if earnings.empty:
        for column in (
            "days_to_next_earnings",
            "days_since_last_earnings",
            "last_earnings_surprise_pct",
            "last_post_earnings_reaction_1d",
        ):
            feature_frame[column] = np.nan
        return feature_frame

    earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"]).dt.normalize()
    output_frames: list[pd.DataFrame] = []

    for ticker, frame in feature_frame.groupby("ticker", sort=False):
        frame = frame.sort_values("date").copy()
        ticker_events = earnings[earnings["ticker"] == ticker].sort_values("earnings_date")
        if ticker_events.empty:
            frame["days_to_next_earnings"] = np.nan
            frame["days_since_last_earnings"] = np.nan
            frame["last_earnings_surprise_pct"] = np.nan
            frame["last_post_earnings_reaction_1d"] = np.nan
            output_frames.append(frame)
            continue

        event_dates = ticker_events["earnings_date"].to_numpy(dtype="datetime64[ns]")
        price_dates = frame["date"].to_numpy(dtype="datetime64[ns]")
        next_idx = np.searchsorted(event_dates, price_dates, side="left")
        prev_idx = np.searchsorted(event_dates, price_dates, side="right") - 1

        next_dates = np.where(next_idx < len(event_dates), event_dates[np.clip(next_idx, 0, len(event_dates) - 1)], np.datetime64("NaT"))
        prev_dates = np.where(prev_idx >= 0, event_dates[np.clip(prev_idx, 0, len(event_dates) - 1)], np.datetime64("NaT"))
        frame["days_to_next_earnings"] = (pd.to_datetime(next_dates) - frame["date"]).dt.days
        frame["days_since_last_earnings"] = (frame["date"] - pd.to_datetime(prev_dates)).dt.days

        event_reactions = _compute_post_earnings_reaction(frame[["date", "ret_1d"]], ticker_events)
        ticker_events = ticker_events.merge(event_reactions, on="earnings_date", how="left")

        surprise = ticker_events["surprise_pct"].to_numpy()
        post_reaction = ticker_events["post_earnings_reaction_1d"].to_numpy()
        frame["last_earnings_surprise_pct"] = np.where(prev_idx >= 0, surprise[np.clip(prev_idx, 0, len(surprise) - 1)], np.nan)
        frame["last_post_earnings_reaction_1d"] = np.where(prev_idx >= 0, post_reaction[np.clip(prev_idx, 0, len(post_reaction) - 1)], np.nan)
        output_frames.append(frame)

    return pd.concat(output_frames, ignore_index=True)


def _compute_post_earnings_reaction(
    price_returns: pd.DataFrame,
    ticker_events: pd.DataFrame,
) -> pd.DataFrame:
    price_returns = price_returns.sort_values("date").reset_index(drop=True)
    event_dates = ticker_events["earnings_date"].to_numpy(dtype="datetime64[ns]")
    price_dates = price_returns["date"].to_numpy(dtype="datetime64[ns]")
    ret_1d = price_returns["ret_1d"].to_numpy()
    positions = np.searchsorted(price_dates, event_dates, side="left")
    reactions = np.where(positions < len(ret_1d), ret_1d[np.clip(positions, 0, len(ret_1d) - 1)], np.nan)
    return pd.DataFrame({"earnings_date": ticker_events["earnings_date"].to_numpy(), "post_earnings_reaction_1d": reactions})


def _add_fundamental_features(
    feature_frame: pd.DataFrame,
    fundamentals: pd.DataFrame,
    cfg: ProjectConfig,
) -> pd.DataFrame:
    if fundamentals.empty:
        for column in (
            "pe_ttm",
            "pb_ratio",
            "ev_ebitda",
            "roe",
            "debt_to_equity",
            "revenue_growth_yoy",
            "assets_growth_yoy",
            "fundamentals_age_days",
            "fundamentals_fresh",
        ):
            feature_frame[column] = np.nan if column != "fundamentals_fresh" else False
        return feature_frame

    fundamentals = _prepare_fundamentals(fundamentals)
    output_frames: list[pd.DataFrame] = []
    for ticker, frame in feature_frame.groupby("ticker", sort=False):
        frame = frame.sort_values("date").copy()
        ticker_fundamentals = fundamentals[fundamentals["ticker"] == ticker].sort_values("filed_at").copy()
        if ticker_fundamentals.empty:
            for column in fundamentals.columns:
                if column != "ticker" and column not in frame.columns:
                    frame[column] = np.nan
            frame["filed_at"] = pd.NaT
            output_frames.append(frame)
            continue

        merged_ticker = pd.merge_asof(
            frame,
            ticker_fundamentals.drop(columns=["ticker"]),
            left_on="date",
            right_on="filed_at",
            direction="backward",
            allow_exact_matches=True,
        )
        merged_ticker["ticker"] = ticker
        output_frames.append(merged_ticker)

    merged = pd.concat(output_frames, ignore_index=True)

    merged["market_cap"] = merged["adj_close"] * merged["common_shares_outstanding"]
    merged["enterprise_value"] = merged["market_cap"] + merged["total_liabilities"] - merged["cash_and_equivalents"]
    merged["pe_ttm"] = merged["market_cap"] / merged["net_income_ttm"]
    merged["pb_ratio"] = merged["market_cap"] / merged["stockholders_equity"]
    merged["ev_ebitda"] = merged["enterprise_value"] / merged["ebitda_ttm"]
    merged["roe"] = merged["net_income_ttm"] / merged["stockholders_equity"]
    merged["debt_to_equity"] = merged["total_liabilities"] / merged["stockholders_equity"]
    merged["fundamentals_age_days"] = (merged["date"] - merged["filed_at"]).dt.days
    merged["fundamentals_fresh"] = merged["filed_at"].notna() & (merged["fundamentals_age_days"] <= cfg.quality.stale_fundamental_days)
    return merged


def _prepare_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    fundamentals = fundamentals.copy()
    fundamentals["filed_at"] = pd.to_datetime(fundamentals["filed_at"])
    fundamentals["period_end"] = pd.to_datetime(fundamentals["period_end"])
    frames: list[pd.DataFrame] = []

    for ticker, frame in fundamentals.groupby("ticker", sort=False):
        frame = frame.sort_values(["filed_at", "period_end"]).copy()
        quarter_mask = frame["form"].astype(str).str.contains("10-Q", na=False)
        annual_mask = frame["form"].astype(str).str.contains("10-K", na=False)

        frame["revenue_quarterly"] = frame["revenue"].where(quarter_mask)
        frame["net_income_quarterly"] = frame["net_income"].where(quarter_mask)
        frame["operating_income_quarterly"] = frame["operating_income"].where(quarter_mask)
        frame["da_quarterly"] = frame["depreciation_amortization"].where(quarter_mask)

        frame["revenue_ttm"] = frame["revenue_quarterly"].rolling(4, min_periods=4).sum()
        frame["net_income_ttm"] = frame["net_income_quarterly"].rolling(4, min_periods=4).sum()
        frame["operating_income_ttm"] = frame["operating_income_quarterly"].rolling(4, min_periods=4).sum()
        frame["depreciation_ttm"] = frame["da_quarterly"].rolling(4, min_periods=1).sum()

        frame.loc[annual_mask & frame["revenue_ttm"].isna(), "revenue_ttm"] = frame.loc[annual_mask, "revenue"]
        frame.loc[annual_mask & frame["net_income_ttm"].isna(), "net_income_ttm"] = frame.loc[annual_mask, "net_income"]
        frame.loc[annual_mask & frame["operating_income_ttm"].isna(), "operating_income_ttm"] = frame.loc[annual_mask, "operating_income"]

        frame["ebitda_ttm"] = frame["operating_income_ttm"] + frame["depreciation_ttm"]
        frame["revenue_growth_yoy"] = frame["revenue_quarterly"] / frame["revenue_quarterly"].shift(4) - 1.0
        frame["assets_growth_yoy"] = frame["total_assets"] / frame["total_assets"].shift(4) - 1.0
        frames.append(frame)

    columns = [
        "ticker",
        "filed_at",
        "period_end",
        "revenue_ttm",
        "net_income_ttm",
        "operating_income_ttm",
        "depreciation_ttm",
        "ebitda_ttm",
        "revenue_growth_yoy",
        "assets_growth_yoy",
        "total_assets",
        "total_liabilities",
        "stockholders_equity",
        "cash_and_equivalents",
        "common_shares_outstanding",
    ]
    return pd.concat(frames, ignore_index=True)[columns]


def _finalize_features(feature_frame: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    continuous_feature_columns = [
        "ret_1d",
        "ret_5d",
        "ret_21d",
        "ret_63d",
        "ret_126d",
        "ret_252d",
        "momentum_12_1",
        "dist_sma_20",
        "dist_sma_50",
        "rsi_14",
        "ret_5d_z_63d",
        "realized_vol_21d",
        "realized_vol_63d",
        "vol_ratio_21_63",
        "atr_14_close",
        "intraday_range",
        "volume_ratio_5_20",
        "volume_ratio_5_63",
        "dollar_volume_20d",
        "volume_trend_21d",
        "rel_strength_5d",
        "rel_strength_21d",
        "rel_strength_63d",
        "sector_rel_ret_1d",
        "sector_rel_ret_21d",
        "sector_vol_gap_21d",
        "beta_63d",
        "vix_close",
        "vix_change_5d",
        "spx_ret_21d",
        "days_to_next_earnings",
        "days_since_last_earnings",
        "last_earnings_surprise_pct",
        "last_post_earnings_reaction_1d",
        "pe_ttm",
        "pb_ratio",
        "ev_ebitda",
        "roe",
        "debt_to_equity",
        "revenue_growth_yoy",
        "assets_growth_yoy",
        "fundamentals_age_days",
        "risk_free_rate",
    ]
    binary_feature_columns = ["spx_above_200d"]
    sector_columns = sorted(column for column in feature_frame.columns if column.startswith("sector_"))
    feature_columns = continuous_feature_columns + binary_feature_columns + sector_columns

    feature_frame["missing_feature_ratio"] = feature_frame[feature_columns].isna().mean(axis=1)
    feature_frame["feature_missing_fail"] = feature_frame["missing_feature_ratio"] > cfg.quality.max_missing_feature_ratio
    feature_frame["fundamentals_missing_fail"] = ~feature_frame["fundamentals_fresh"].fillna(False)
    feature_frame["universe_history_fail"] = ~feature_frame["universe_reliable"].fillna(False)

    feature_frame[continuous_feature_columns] = feature_frame.groupby("date")[continuous_feature_columns].transform(
        lambda series: series.fillna(series.median())
    )
    feature_frame[continuous_feature_columns] = feature_frame.groupby("date")[continuous_feature_columns].transform(zscore_series)
    feature_frame[binary_feature_columns] = feature_frame[binary_feature_columns].fillna(0.0)

    feature_frame["trainable"] = ~(
        feature_frame["insufficient_price_history"].fillna(True)
        | feature_frame["insufficient_post_listing_history"].fillna(True)
        | feature_frame["stale_price"].fillna(True)
        | feature_frame["missing_gap_fail"].fillna(True)
        | feature_frame["feature_missing_fail"].fillna(True)
        | feature_frame["fundamentals_missing_fail"].fillna(True)
        | feature_frame["universe_history_fail"].fillna(True)
    )
    return feature_frame


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
