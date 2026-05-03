"""Abstract adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PriceAdapter(ABC):
    @abstractmethod
    def fetch_prices(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError


class MetadataAdapter(ABC):
    @abstractmethod
    def fetch_metadata(self, tickers: list[str]) -> pd.DataFrame:
        raise NotImplementedError


class EarningsAdapter(ABC):
    @abstractmethod
    def fetch_earnings(self, tickers: list[str]) -> pd.DataFrame:
        raise NotImplementedError


class FundamentalsAdapter(ABC):
    @abstractmethod
    def fetch_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        raise NotImplementedError


class UniverseAdapter(ABC):
    @abstractmethod
    def fetch_membership_history(self, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError
