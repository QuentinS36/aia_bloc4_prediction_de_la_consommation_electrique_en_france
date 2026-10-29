import base64

import pytest

from dday.etl import extract

GOOD = base64.b64encode(b"mon-id:mon-secret").decode()


class FakeResponse:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body or {}, text

    def json(self):
        return self._body


def _patch_post(monkeypatch, response, sent):
    def post(url, headers=None, timeout=None, **kw):
        sent["headers"] = headers
        return response
    monkeypatch.setattr(extract.requests, "post", post)


def test_token_ok_and_key_cleaned(monkeypatch):
    sent = {}
    _patch_post(monkeypatch, FakeResponse(200, {"access_token": "T"}), sent)
    monkeypatch.setenv("RTE_SECRET_KEY", f'  "Basic {GOOD}"  ')
    assert extract._rte_token() == "T"
    assert sent["headers"]["Authorization"] == f"Basic {GOOD}"


@pytest.mark.parametrize("bad", ["pas du base64 !!", base64.b64encode(b"sansdeuxpoints").decode(), f"{GOOD}\n{GOOD}"])
def test_invalid_key_is_explained_before_any_request(monkeypatch, bad):
    monkeypatch.setenv("RTE_SECRET_KEY", bad)
    with pytest.raises(ValueError, match="RTE_SECRET_KEY"):
        extract._rte_token()


def test_missing_key(monkeypatch):
    monkeypatch.setenv("RTE_SECRET_KEY", "")
    with pytest.raises(RuntimeError, match="absent ou vide"):
        extract._rte_token()


def test_http_error_shows_server_message_not_the_key(monkeypatch):
    _patch_post(monkeypatch, FakeResponse(400, text='{"error":"invalid_client"}'), {})
    monkeypatch.setenv("RTE_SECRET_KEY", GOOD)
    with pytest.raises(RuntimeError) as e:
        extract._rte_token()
    assert "400" in str(e.value) and "invalid_client" in str(e.value) and GOOD not in str(e.value)


def test_consumption_error_mentions_subscription(monkeypatch):
    _patch_post(monkeypatch, FakeResponse(200, {"access_token": "T"}), {})
    monkeypatch.setattr(extract.requests, "get",
                        lambda *a, **k: FakeResponse(403, text='{"error":"forbidden"}'))
    monkeypatch.setenv("RTE_SECRET_KEY", GOOD)
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    with pytest.raises(RuntimeError, match="abonnée"):
        extract.fetch_rte_consumption(now - dt.timedelta(hours=2), now)
