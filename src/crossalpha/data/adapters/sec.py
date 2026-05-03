"""SEC company-facts adapter for point-in-time fundamentals."""

from __future__ import annotations

import json
import logging
import time
import urllib.request

import pandas as pd

from crossalpha.data.adapters.base import FundamentalsAdapter
from crossalpha.utils.http import sec_request_headers


LOGGER = logging.getLogger(__name__)

FORM_ALLOWLIST = {"10-Q", "10-K", "10-Q/A", "10-K/A", "8-K"}
METRIC_MAP: dict[str, list[tuple[str, list[str], set[str]]]] = {
    "revenue": [("us-gaap", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"], {"USD"})],
    "net_income": [("us-gaap", ["NetIncomeLoss"], {"USD"})],
    "total_assets": [("us-gaap", ["Assets"], {"USD"})],
    "total_liabilities": [("us-gaap", ["Liabilities"], {"USD"})],
    "stockholders_equity": [("us-gaap", ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "StockholdersEquity"], {"USD"})],
    "cash_and_equivalents": [("us-gaap", ["CashAndCashEquivalentsAtCarryingValue"], {"USD"})],
    "common_shares_outstanding": [("dei", ["EntityCommonStockSharesOutstanding"], {"shares"})],
    "eps_diluted": [("us-gaap", ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"], {"USD/shares"})],
    "operating_income": [("us-gaap", ["OperatingIncomeLoss"], {"USD"})],
    "depreciation_amortization": [("us-gaap", ["DepreciationDepletionAndAmortization"], {"USD"})],
}


def _read_json(url: str) -> dict:
    request = urllib.request.Request(url=url, headers=sec_request_headers())
    with urllib.request.urlopen(request, timeout=30) as response:  # nosec: B310
        return json.loads(response.read().decode("utf-8"))


class SecCompanyFactsAdapter(FundamentalsAdapter):
    def __init__(self, pause_seconds: float = 0.15) -> None:
        self.pause_seconds = pause_seconds

    def fetch_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        ticker_map = self._fetch_ticker_cik_map()
        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            cik = ticker_map.get(ticker.upper())
            if cik is None:
                LOGGER.warning("No SEC CIK mapping found for %s.", ticker)
                continue
            try:
                facts = _read_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
            except Exception as exc:  # pragma: no cover - network adapter
                LOGGER.warning("SEC company-facts fetch failed for %s: %s", ticker, exc)
                continue
            frame = self._normalize_company_facts(ticker, facts)
            if not frame.empty:
                frames.append(frame)
            time.sleep(self.pause_seconds)

        if not frames:
            return pd.DataFrame(columns=["ticker", "filed_at", "period_end", "form", "source"])
        return pd.concat(frames, ignore_index=True).sort_values(["ticker", "filed_at", "period_end"]).reset_index(drop=True)

    @staticmethod
    def _fetch_ticker_cik_map() -> dict[str, int]:
        payload = _read_json("https://www.sec.gov/files/company_tickers.json")
        mapping: dict[str, int] = {}
        for value in payload.values():
            ticker = str(value["ticker"]).upper()
            mapping[ticker] = int(value["cik_str"])
        return mapping

    def _normalize_company_facts(self, ticker: str, facts: dict) -> pd.DataFrame:
        metric_frames: list[pd.DataFrame] = []
        for metric_name, definitions in METRIC_MAP.items():
            metric_frame = self._extract_metric_frames(facts, metric_name, definitions)
            if not metric_frame.empty:
                metric_frames.append(metric_frame)

        if not metric_frames:
            return pd.DataFrame()

        merged = metric_frames[0]
        for frame in metric_frames[1:]:
            merged = merged.merge(frame, on=["filed_at", "period_end", "form"], how="outer")

        merged["ticker"] = ticker
        merged["source"] = "sec_companyfacts"
        merged["filed_at"] = pd.to_datetime(merged["filed_at"])
        merged["period_end"] = pd.to_datetime(merged["period_end"])
        keep_columns = [
            "ticker",
            "filed_at",
            "period_end",
            "form",
            "revenue",
            "net_income",
            "total_assets",
            "total_liabilities",
            "stockholders_equity",
            "cash_and_equivalents",
            "common_shares_outstanding",
            "eps_diluted",
            "operating_income",
            "depreciation_amortization",
            "source",
        ]
        for column in keep_columns:
            if column not in merged.columns:
                merged[column] = pd.NA
        return merged[keep_columns].drop_duplicates(subset=["ticker", "filed_at", "period_end", "form"], keep="last")

    def _extract_metric_frames(
        self,
        facts: dict,
        metric_name: str,
        definitions: list[tuple[str, list[str], set[str]]],
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for taxonomy, concept_names, allowed_units in definitions:
            taxonomy_payload = facts.get("facts", {}).get(taxonomy, {})
            for concept_name in concept_names:
                concept_payload = taxonomy_payload.get(concept_name)
                if concept_payload is None:
                    continue
                unit_key = self._select_unit(concept_payload.get("units", {}), allowed_units)
                if unit_key is None:
                    continue
                for item in concept_payload["units"][unit_key]:
                    if item.get("form") not in FORM_ALLOWLIST:
                        continue
                    if item.get("filed") is None or item.get("end") is None:
                        continue
                    rows.append({
                        "filed_at": item["filed"],
                        "period_end": item["end"],
                        "form": item.get("form"),
                        metric_name: item.get("val"),
                    })
                if rows:
                    frame = pd.DataFrame(rows)
                    return frame.drop_duplicates(subset=["filed_at", "period_end", "form"], keep="last")
        return pd.DataFrame(columns=["filed_at", "period_end", "form", metric_name])

    @staticmethod
    def _select_unit(units_payload: dict[str, list[dict]], allowed_units: set[str]) -> str | None:
        exact_match = next((unit for unit in units_payload if unit in allowed_units), None)
        if exact_match is not None:
            return exact_match
        for unit in units_payload:
            if any(token in unit for token in allowed_units):
                return unit
        return None
