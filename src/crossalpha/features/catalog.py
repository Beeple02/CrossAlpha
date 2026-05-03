"""Feature catalog metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class FeatureDefinition:
    name: str
    formula: str
    valid_at: str
    missing_handling: str
    horizon_sensitive: list[str]


BASE_FEATURE_DEFINITIONS: list[FeatureDefinition] = [
    FeatureDefinition("ret_1d", "(adj_close_t / adj_close_t-1) - 1", "market_close_t", "cross-sectional median", ["1d"]),
    FeatureDefinition("ret_5d", "(adj_close_t / adj_close_t-5) - 1", "market_close_t", "cross-sectional median", ["1d", "1w"]),
    FeatureDefinition("ret_21d", "(adj_close_t / adj_close_t-21) - 1", "market_close_t", "cross-sectional median", ["1w", "2w"]),
    FeatureDefinition("ret_63d", "(adj_close_t / adj_close_t-63) - 1", "market_close_t", "cross-sectional median", ["1m", "2m"]),
    FeatureDefinition("ret_126d", "(adj_close_t / adj_close_t-126) - 1", "market_close_t", "cross-sectional median", ["2m"]),
    FeatureDefinition("ret_252d", "(adj_close_t / adj_close_t-252) - 1", "market_close_t", "cross-sectional median", ["2m"]),
    FeatureDefinition("momentum_12_1", "ret_252d - ret_21d", "market_close_t", "cross-sectional median", ["1m", "2m"]),
    FeatureDefinition("dist_sma_20", "(adj_close - sma20) / sma20", "market_close_t", "cross-sectional median", ["1d", "1w"]),
    FeatureDefinition("dist_sma_50", "(adj_close - sma50) / sma50", "market_close_t", "cross-sectional median", ["1w", "2w"]),
    FeatureDefinition("rsi_14", "standard RSI(14)", "market_close_t", "cross-sectional median", ["1d", "1w"]),
    FeatureDefinition("ret_5d_z_63d", "(ret_5d - mean_63d(ret_5d)) / std_63d(ret_5d)", "market_close_t", "cross-sectional median", ["1d"]),
    FeatureDefinition("realized_vol_21d", "std(ret_1d, 21) * sqrt(252)", "market_close_t", "cross-sectional median", ["1d", "1w", "2w", "1m", "2m"]),
    FeatureDefinition("realized_vol_63d", "std(ret_1d, 63) * sqrt(252)", "market_close_t", "cross-sectional median", ["1d", "1w", "2w", "1m", "2m"]),
    FeatureDefinition("vol_ratio_21_63", "realized_vol_21d / realized_vol_63d", "market_close_t", "cross-sectional median", ["1d", "1w", "2w", "1m", "2m"]),
    FeatureDefinition("atr_14_close", "ATR(14) / close", "market_close_t", "cross-sectional median", ["1d", "1w"]),
    FeatureDefinition("intraday_range", "(high - low) / close", "market_close_t", "cross-sectional median", ["1d"]),
    FeatureDefinition("volume_ratio_5_20", "mean(volume, 5) / mean(volume, 20)", "market_close_t", "cross-sectional median", ["1d", "1w"]),
    FeatureDefinition("volume_ratio_5_63", "mean(volume, 5) / mean(volume, 63)", "market_close_t", "cross-sectional median", ["1w", "2w"]),
    FeatureDefinition("dollar_volume_20d", "mean(adj_close * volume, 20)", "market_close_t", "cross-sectional median", ["1d", "1w", "2w", "1m", "2m"]),
    FeatureDefinition("volume_trend_21d", "slope(log(volume), 21)", "market_close_t", "cross-sectional median", ["1w", "2w"]),
    FeatureDefinition("rel_strength_5d", "ret_5d - spx_ret_5d", "market_close_t", "cross-sectional median", ["1d", "1w"]),
    FeatureDefinition("rel_strength_21d", "ret_21d - spx_ret_21d", "market_close_t", "cross-sectional median", ["1w", "2w"]),
    FeatureDefinition("rel_strength_63d", "ret_63d - spx_ret_63d", "market_close_t", "cross-sectional median", ["1m", "2m"]),
    FeatureDefinition("sector_rel_ret_1d", "ret_1d - sector_mean(ret_1d)", "market_close_t", "cross-sectional median", ["1d", "1w"]),
    FeatureDefinition("sector_rel_ret_21d", "ret_21d - sector_mean(ret_21d)", "market_close_t", "cross-sectional median", ["1w", "2w", "1m"]),
    FeatureDefinition("sector_vol_gap_21d", "realized_vol_21d - sector_mean(realized_vol_21d)", "market_close_t", "cross-sectional median", ["1d", "1w", "2w", "1m", "2m"]),
    FeatureDefinition("beta_63d", "cov(ret_1d, spx_ret_1d, 63) / var(spx_ret_1d, 63)", "market_close_t", "cross-sectional median", ["1m", "2m"]),
    FeatureDefinition("vix_close", "current VIX close", "market_close_t", "cross-sectional median", ["1d", "1w", "2w", "1m", "2m"]),
    FeatureDefinition("vix_change_5d", "(vix_t / vix_t-5) - 1", "market_close_t", "cross-sectional median", ["1d", "1w", "2w", "1m", "2m"]),
    FeatureDefinition("spx_ret_21d", "(spx_t / spx_t-21) - 1", "market_close_t", "cross-sectional median", ["1d", "1w", "2w", "1m", "2m"]),
    FeatureDefinition("spx_above_200d", "1 if spx > sma200 else 0", "market_close_t", "leave binary", ["1d", "1w", "2w", "1m", "2m"]),
    FeatureDefinition("days_to_next_earnings", "next_earnings_date - date", "market_close_t", "cross-sectional median", ["1d", "1w", "2w"]),
    FeatureDefinition("days_since_last_earnings", "date - previous_earnings_date", "market_close_t", "cross-sectional median", ["1d", "1w", "2w", "1m"]),
    FeatureDefinition("last_earnings_surprise_pct", "previous known earnings surprise", "market_close_t", "cross-sectional median", ["1w", "2w", "1m"]),
    FeatureDefinition("last_post_earnings_reaction_1d", "first post-event 1d return after previous earnings", "market_close_t", "cross-sectional median", ["1d", "1w", "2w"]),
    FeatureDefinition("pe_ttm", "market_cap / net_income_ttm", "market_close_t", "cross-sectional median", ["1m", "2m"]),
    FeatureDefinition("pb_ratio", "market_cap / stockholders_equity", "market_close_t", "cross-sectional median", ["1m", "2m"]),
    FeatureDefinition("ev_ebitda", "enterprise_value / ebitda_ttm", "market_close_t", "cross-sectional median", ["1m", "2m"]),
    FeatureDefinition("roe", "net_income_ttm / stockholders_equity", "market_close_t", "cross-sectional median", ["1m", "2m"]),
    FeatureDefinition("debt_to_equity", "total_liabilities / stockholders_equity", "market_close_t", "cross-sectional median", ["1m", "2m"]),
    FeatureDefinition("revenue_growth_yoy", "quarterly revenue / lag_4_quarters - 1", "market_close_t", "cross-sectional median", ["2m"]),
    FeatureDefinition("assets_growth_yoy", "assets / lag_4_quarters - 1", "market_close_t", "cross-sectional median", ["2m"]),
    FeatureDefinition("fundamentals_age_days", "date - latest filing date", "market_close_t", "cross-sectional median", ["1d", "1w", "2w", "1m", "2m"]),
    FeatureDefinition("risk_free_rate", "latest daily FRED DGS3MO", "market_close_t", "cross-sectional median", ["1m", "2m"]),
]


def feature_catalog_records(dynamic_sector_columns: list[str] | None = None) -> list[dict[str, object]]:
    records = [asdict(feature) for feature in BASE_FEATURE_DEFINITIONS]
    for sector_column in dynamic_sector_columns or []:
        records.append(asdict(FeatureDefinition(
            name=sector_column,
            formula="one-hot sector membership",
            valid_at="market_close_t",
            missing_handling="binary indicator",
            horizon_sensitive=["1d", "1w", "2w", "1m", "2m"],
        )))
    return records
