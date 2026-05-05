"""Yahoo Finance adapter for prices, metadata, and earnings."""

from __future__ import annotations

from itertools import islice
import logging
import time

import pandas as pd

from crossalpha.data.adapters.base import EarningsAdapter, MetadataAdapter, PriceAdapter


LOGGER = logging.getLogger(__name__)


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    iterator = iter(items)
    batches: list[list[str]] = []
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return batches
        batches.append(batch)


class YahooFinanceAdapter(PriceAdapter, MetadataAdapter, EarningsAdapter):
    def __init__(self, pause_seconds: float = 0.25, batch_size: int = 50) -> None:
        self.pause_seconds = pause_seconds
        self.batch_size = batch_size

    def fetch_prices(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        import yfinance as yf

        frames: list[pd.DataFrame] = []
        for batch in _batched(symbols, self.batch_size):
            LOGGER.info("Downloading price history for %s tickers from Yahoo Finance.", len(batch))
            panel = yf.download(
                tickers=batch,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            frames.append(self._normalize_price_panel(panel, batch))
            time.sleep(self.pause_seconds)

        if not frames:
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"])

        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    def fetch_metadata(self, tickers: list[str]) -> pd.DataFrame:
        import yfinance as yf

        rows: list[dict[str, object]] = []
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).get_info()
            except Exception as exc:  # pragma: no cover - network adapter
                LOGGER.warning("Metadata fetch failed for %s: %s", ticker, exc)
                info = {}
            rows.append({
                "ticker": ticker,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "exchange": info.get("exchange"),
                "quote_type": info.get("quoteType"),
                "long_name": info.get("longName"),
                "source": "yfinance",
            })
            time.sleep(self.pause_seconds)
        return pd.DataFrame(rows)

    def fetch_earnings(self, tickers: list[str]) -> pd.DataFrame:
        import yfinance as yf

        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            try:
                earnings = yf.Ticker(ticker).get_earnings_dates(limit=24)
            except Exception as exc:  # pragma: no cover - network adapter
                LOGGER.warning("Earnings fetch failed for %s: %s", ticker, exc)
                earnings = None

            if earnings is None or len(earnings) == 0:
                continue

            frame = earnings.reset_index().rename(columns={
                "Earnings Date": "earnings_date",
                "EPS Estimate": "eps_estimate",
                "Reported EPS": "eps_actual",
                "Surprise(%)": "surprise_pct",
            })
            frame["ticker"] = ticker
            frame["source"] = "yfinance"
            frames.append(frame[["ticker", "earnings_date", "eps_estimate", "eps_actual", "surprise_pct", "source"]])
            time.sleep(self.pause_seconds)

        if not frames:
            return pd.DataFrame(columns=["ticker", "earnings_date", "eps_estimate", "eps_actual", "surprise_pct", "source"])
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _normalize_price_panel(panel: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
        if panel.empty:
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"])

        rows: list[pd.DataFrame] = []
        if isinstance(panel.columns, pd.MultiIndex):
            for symbol in symbols:
                if symbol not in panel.columns.get_level_values(0):
                    continue
                frame = panel[symbol].reset_index()
                frame["ticker"] = symbol
                rows.append(frame)
        else:
            frame = panel.reset_index()
            frame["ticker"] = symbols[0]
            rows.append(frame)

        if not rows:
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"])

        df = pd.concat(rows, ignore_index=True).rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })
        df["source"] = "yfinance"
        keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
        df = df[keep]
        price_columns = ["open", "high", "low", "close", "adj_close"]
        df = df[df[price_columns].notna().any(axis=1)].copy()
        return df
