"""Data ingestion pipeline."""

from __future__ import annotations

import logging

import pandas as pd

from crossalpha.config import ProjectConfig
from crossalpha.data.adapters.fred import FredAdapter
from crossalpha.data.adapters.sec import SecCompanyFactsAdapter
from crossalpha.data.adapters.wikipedia import WikipediaSp500Adapter
from crossalpha.data.adapters.yahoo import YahooFinanceAdapter
from crossalpha.data.quality import clean_prices_and_flags
from crossalpha.data.storage import DataCatalog, dedupe_sort
from crossalpha.data.universe import build_daily_universe
from crossalpha.utils.io import read_parquet, write_parquet


LOGGER = logging.getLogger(__name__)


def run_data_ingestion(cfg: ProjectConfig) -> None:
    catalog = DataCatalog(cfg)
    yahoo = YahooFinanceAdapter()
    wiki = WikipediaSp500Adapter()
    sec = SecCompanyFactsAdapter()
    fred = FredAdapter()

    LOGGER.info("Starting data ingestion.")

    membership = wiki.fetch_membership_history(cfg.data.start_date, cfg.data.end_date)
    wiki_metadata = wiki.fetch_sector_metadata()
    write_parquet(membership, catalog.raw("sp500_membership"))
    write_parquet(wiki_metadata, catalog.raw("wikipedia_metadata"))

    tickers = sorted(membership["ticker"].dropna().unique().tolist())

    price_history = _incremental_prices(
        yahoo=yahoo,
        catalog=catalog,
        artifact_name="prices",
        symbols=tickers,
        cfg=cfg,
    )
    available_tickers = _tickers_with_price_history(price_history)
    unavailable_count = len(set(tickers) - set(available_tickers))
    LOGGER.info(
        "Usable Yahoo price history found for %s/%s universe tickers.",
        len(available_tickers),
        len(tickers),
    )
    if unavailable_count > 0:
        LOGGER.warning(
            "Skipping metadata, earnings, and fundamentals fetch for %s tickers without usable price history.",
            unavailable_count,
        )
    if not available_tickers:
        raise RuntimeError("No usable Yahoo price history was downloaded for the selected universe.")

    benchmark = _incremental_prices(
        yahoo=yahoo,
        catalog=catalog,
        artifact_name="benchmark",
        symbols=[cfg.data.benchmark_symbol],
        cfg=cfg,
    )
    vix = _incremental_prices(
        yahoo=yahoo,
        catalog=catalog,
        artifact_name="vix",
        symbols=[cfg.data.vix_symbol],
        cfg=cfg,
    )
    recent_tickers = _tickers_with_recent_price_history(price_history, cfg.data.end_date)
    vendor_tickers = _vendor_enrichment_tickers(recent_tickers, wiki_metadata)
    LOGGER.info(
        "Recent vendor-eligible tickers near %s: %s.",
        cfg.data.end_date,
        len(recent_tickers),
    )
    LOGGER.info(
        "Current-constituent vendor enrichment tickers: %s.",
        len(vendor_tickers),
    )
    metadata = yahoo.fetch_metadata(vendor_tickers)
    earnings = yahoo.fetch_earnings(vendor_tickers)
    fundamentals = sec.fetch_fundamentals(vendor_tickers)
    risk_free = fred.fetch_series(cfg.data.risk_free_series)

    metadata = metadata.merge(
        wiki_metadata.rename(columns={"sector": "wiki_sector", "industry": "wiki_industry"}),
        on="ticker",
        how="outer",
    )
    metadata["sector"] = metadata["sector"].fillna(metadata["wiki_sector"]).fillna("Unknown")
    metadata["industry"] = metadata["industry"].fillna(metadata["wiki_industry"]).fillna("Unknown")
    metadata = metadata.drop(columns=["wiki_sector", "wiki_industry"], errors="ignore")

    write_parquet(dedupe_sort(price_history, ["ticker", "date"]), catalog.raw("prices"))
    write_parquet(dedupe_sort(benchmark, ["ticker", "date"]), catalog.raw("benchmark"))
    write_parquet(dedupe_sort(vix, ["ticker", "date"]), catalog.raw("vix"))
    write_parquet(dedupe_sort(metadata, ["ticker"]), catalog.raw("metadata"))
    write_parquet(dedupe_sort(earnings, ["ticker", "earnings_date"]), catalog.raw("earnings"))
    write_parquet(dedupe_sort(fundamentals, ["ticker", "filed_at", "period_end", "form"]), catalog.raw("fundamentals"))
    write_parquet(dedupe_sort(risk_free, ["series_id", "date"]), catalog.raw("risk_free"))

    benchmark_dates = pd.DatetimeIndex(pd.to_datetime(benchmark["date"]).sort_values().unique())
    cleaned_prices, quality_flags = clean_prices_and_flags(price_history, benchmark_dates, cfg)
    daily_universe = build_daily_universe(membership, metadata, cfg)

    write_parquet(dedupe_sort(cleaned_prices, ["ticker", "date"]), catalog.processed("prices_cleaned"))
    write_parquet(dedupe_sort(quality_flags, ["ticker", "date"]), catalog.processed("price_quality"))
    write_parquet(dedupe_sort(daily_universe, ["ticker", "date"]), catalog.processed("daily_universe"))
    write_parquet(dedupe_sort(benchmark, ["ticker", "date"]), catalog.processed("benchmark"))
    write_parquet(dedupe_sort(vix, ["ticker", "date"]), catalog.processed("vix"))
    write_parquet(dedupe_sort(risk_free, ["series_id", "date"]), catalog.processed("risk_free"))
    write_parquet(dedupe_sort(earnings, ["ticker", "earnings_date"]), catalog.processed("earnings"))
    write_parquet(dedupe_sort(fundamentals, ["ticker", "filed_at", "period_end", "form"]), catalog.processed("fundamentals"))
    write_parquet(dedupe_sort(metadata, ["ticker"]), catalog.processed("metadata"))

    LOGGER.info("Completed data ingestion.")


def _incremental_prices(
    yahoo: YahooFinanceAdapter,
    catalog: DataCatalog,
    artifact_name: str,
    symbols: list[str],
    cfg: ProjectConfig,
) -> pd.DataFrame:
    artifact_path = catalog.raw(artifact_name)
    if artifact_path.exists() and not cfg.data.refresh:
        existing = read_parquet(artifact_path)
        last_date = pd.to_datetime(existing["date"]).max()
        start_date = (last_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        LOGGER.info("Refreshing %s incrementally from %s.", artifact_name, start_date)
        incoming = yahoo.fetch_prices(symbols, start_date, cfg.data.end_date)
        combined = pd.concat([existing, incoming], ignore_index=True)
        return dedupe_sort(combined, ["ticker", "date"])

    LOGGER.info("Downloading full %s history.", artifact_name)
    return yahoo.fetch_prices(symbols, cfg.data.start_date, cfg.data.end_date)


def _tickers_with_price_history(price_history: pd.DataFrame) -> list[str]:
    if price_history.empty:
        return []
    price_columns = ["open", "high", "low", "close", "adj_close"]
    usable = price_history[price_history[price_columns].notna().any(axis=1)].copy()
    if usable.empty:
        return []
    return sorted(usable["ticker"].dropna().astype(str).unique().tolist())


def _tickers_with_recent_price_history(
    price_history: pd.DataFrame,
    end_date: str,
    max_staleness_days: int = 120,
) -> list[str]:
    available_tickers = _tickers_with_price_history(price_history)
    if not available_tickers:
        return []

    frame = price_history.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    last_seen = frame.groupby("ticker")["date"].max()
    cutoff = pd.Timestamp(end_date) - pd.Timedelta(days=max_staleness_days)
    recent = last_seen[last_seen >= cutoff]
    return sorted(recent.index.astype(str).tolist())


def _vendor_enrichment_tickers(
    recent_tickers: list[str],
    wiki_metadata: pd.DataFrame,
) -> list[str]:
    if not recent_tickers or wiki_metadata.empty or "ticker" not in wiki_metadata.columns:
        return []
    current_constituents = set(wiki_metadata["ticker"].dropna().astype(str).tolist())
    return sorted(ticker for ticker in recent_tickers if ticker in current_constituents)
