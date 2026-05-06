from __future__ import annotations

from crossalpha.utils.http import sec_request_headers, web_request_headers


def test_http_headers_use_crossalpha_user_agent(monkeypatch):
    monkeypatch.setenv("CROSSALPHA_USER_AGENT", "CrossAlpha Research test@example.com")
    headers = web_request_headers(host="en.wikipedia.org")
    assert headers["User-Agent"] == "CrossAlpha Research test@example.com"
    assert headers["Host"] == "en.wikipedia.org"


def test_sec_headers_are_json_friendly(monkeypatch):
    monkeypatch.setenv("CROSSALPHA_USER_AGENT", "CrossAlpha Research test@example.com")
    headers = sec_request_headers("www.sec.gov")
    assert headers["User-Agent"] == "CrossAlpha Research test@example.com"
    assert headers["Accept"] == "application/json"
    assert headers["Host"] == "www.sec.gov"
