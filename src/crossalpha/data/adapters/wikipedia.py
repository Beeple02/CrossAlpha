"""Wikipedia adapter for S&P 500 membership reconstruction."""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import requests

from crossalpha.data.adapters.base import UniverseAdapter
from crossalpha.utils.dates import add_trading_days, trading_days_between
from crossalpha.utils.http import web_request_headers


WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join(str(part).strip() for part in column if str(part).strip())
            for column in df.columns.to_flat_index()
        ]
    else:
        df.columns = [str(column).strip() for column in df.columns]
    return df


class WikipediaSp500Adapter(UniverseAdapter):
    def fetch_membership_history(self, start_date: str, end_date: str) -> pd.DataFrame:
        current, changes = self.fetch_tables()
        return self._reconstruct_history(current, changes, start_date, end_date)

    @staticmethod
    def fetch_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
        response = requests.get(
            WIKIPEDIA_URL,
            headers=web_request_headers(host="en.wikipedia.org"),
            timeout=30,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network adapter
            raise RuntimeError(
                "Wikipedia membership fetch failed. Set CROSSALPHA_USER_AGENT and retry."
            ) from exc
        tables = pd.read_html(StringIO(response.text))
        current = _flatten_columns(tables[0]).rename(columns={"Symbol": "ticker"})
        changes = next(
            _flatten_columns(table)
            for table in tables[1:]
            if _is_changes_table(_flatten_columns(table))
        )
        return current, changes

    def fetch_sector_metadata(self) -> pd.DataFrame:
        current, _ = self.fetch_tables()
        current = current.rename(columns={
            "GICS Sector": "sector",
            "GICS Sub-Industry": "industry",
        })
        metadata = current[["ticker", "sector", "industry"]].copy()
        metadata["source"] = "wikipedia"
        metadata["ticker"] = metadata["ticker"].str.replace(".", "-", regex=False)
        return metadata.drop_duplicates(subset=["ticker"])

    def _reconstruct_history(
        self,
        current: pd.DataFrame,
        changes: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        current_members = {ticker.replace(".", "-") for ticker in current["ticker"].dropna().astype(str)}
        changes = self._normalize_changes(changes)
        dates = trading_days_between(start_date, end_date)
        segment_end = dates.max()
        frames: list[pd.DataFrame] = []
        earliest_change = changes["date"].min() if not changes.empty else dates.min()

        for change_date, day_changes in changes.groupby("date", sort=False):
            segment_start = add_trading_days(change_date, 1)
            mask = (dates >= segment_start) & (dates <= segment_end)
            segment_dates = dates[mask]
            if len(segment_dates) > 0 and current_members:
                reliability = 0.9 if segment_start >= earliest_change else 0.5
                frames.append(self._materialize_segment(segment_dates, current_members, reliability))

            for _, row in day_changes.iterrows():
                added = row.get("added_ticker")
                removed = row.get("removed_ticker")
                if pd.notna(added):
                    current_members.discard(str(added))
                if pd.notna(removed):
                    current_members.add(str(removed))
            segment_end = pd.Timestamp(change_date)

        remaining_dates = dates[dates <= segment_end]
        if len(remaining_dates) > 0 and current_members:
            reliability = 0.5 if not changes.empty else 0.25
            frames.append(self._materialize_segment(remaining_dates, current_members, reliability))

        if not frames:
            return pd.DataFrame(columns=["date", "ticker", "is_member", "membership_reliability", "source"])

        history = pd.concat(frames, ignore_index=True)
        return history.drop_duplicates(subset=["date", "ticker"], keep="last").sort_values(["date", "ticker"]).reset_index(drop=True)

    @staticmethod
    def _normalize_changes(changes: pd.DataFrame) -> pd.DataFrame:
        date_col = _match_first_column(changes.columns, include=("date",), exclude=("added", "removed"))
        added_col = _match_first_column(changes.columns, include=("added", "ticker"))
        removed_col = _match_first_column(changes.columns, include=("removed", "ticker"))
        if date_col is None or added_col is None or removed_col is None:
            return pd.DataFrame(columns=["date", "added_ticker", "removed_ticker"])

        normalized = changes[[date_col, added_col, removed_col]].rename(columns={
            date_col: "date",
            added_col: "added_ticker",
            removed_col: "removed_ticker",
        })
        normalized["date"] = pd.to_datetime(normalized["date"])
        for column in ("added_ticker", "removed_ticker"):
            normalized[column] = normalized[column].astype(str).str.replace(".", "-", regex=False)
            normalized.loc[normalized[column].isin({"nan", "None", ""}), column] = np.nan
        return normalized.sort_values("date", ascending=False).reset_index(drop=True)

    @staticmethod
    def _materialize_segment(
        dates: pd.DatetimeIndex,
        members: set[str],
        reliability: float,
    ) -> pd.DataFrame:
        members_array = np.array(sorted(members), dtype=object)
        return pd.DataFrame({
            "date": np.repeat(dates.to_numpy(), len(members_array)),
            "ticker": np.tile(members_array, len(dates)),
            "is_member": True,
            "membership_reliability": reliability,
            "source": "wikipedia_sp500",
        })


def _is_changes_table(df: pd.DataFrame) -> bool:
    return (
        _match_first_column(df.columns, include=("date",), exclude=("added", "removed")) is not None
        and _match_first_column(df.columns, include=("added", "ticker")) is not None
        and _match_first_column(df.columns, include=("removed", "ticker")) is not None
    )


def _match_first_column(
    columns: pd.Index | list[str],
    include: tuple[str, ...],
    exclude: tuple[str, ...] = (),
) -> str | None:
    for column in columns:
        normalized = str(column).strip().lower()
        if all(token in normalized for token in include) and not any(token in normalized for token in exclude):
            return str(column)
    return None
