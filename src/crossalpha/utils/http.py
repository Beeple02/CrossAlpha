"""HTTP helpers for public data adapters."""

from __future__ import annotations

import os


DEFAULT_USER_AGENT = "CrossAlpha/0.1 research project (set CROSSALPHA_USER_AGENT with contact info)"


def default_user_agent() -> str:
    return (
        os.getenv("CROSSALPHA_USER_AGENT")
        or os.getenv("USER_AGENT")
        or DEFAULT_USER_AGENT
    )


def web_request_headers(host: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": default_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if host:
        headers["Host"] = host
    return headers


def sec_request_headers() -> dict[str, str]:
    return {
        "User-Agent": default_user_agent(),
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }
