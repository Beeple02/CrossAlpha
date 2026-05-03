"""FRED adapter for the risk-free rate."""

from __future__ import annotations

from io import StringIO
import urllib.request

import pandas as pd


class FredAdapter:
    BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    def fetch_series(self, series_id: str) -> pd.DataFrame:
        url = self.BASE_URL.format(series_id=series_id)
        with urllib.request.urlopen(url, timeout=30) as response:  # nosec: B310
            payload = response.read().decode("utf-8")
        df = pd.read_csv(StringIO(payload))
        df = df.rename(columns={"DATE": "date", series_id: "value"})
        df["date"] = pd.to_datetime(df["date"])
        df["series_id"] = series_id
        df["source"] = "fred"
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df
