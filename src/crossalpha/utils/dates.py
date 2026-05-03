"""Date and trading-calendar helpers."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def to_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def trading_days_between(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(to_timestamp(start), to_timestamp(end))


def add_trading_days(date: str | pd.Timestamp, trading_days: int) -> pd.Timestamp:
    return to_timestamp(date) + pd.offsets.BDay(trading_days)


def ensure_datetime_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])
    return df
