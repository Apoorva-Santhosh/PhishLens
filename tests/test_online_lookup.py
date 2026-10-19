"""All DNS, WHOIS and worker transports are mocked; tests never contact domains."""

import json
import socket
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import dns.message
import dns.rrset
import pytest

from phishlens.features import url_online as online
from phishlens.features.url_online import (
    ONLINE_NAMES,
    LookupFailure,
    OnlineLookup,
    OnlineResult,
    OnlineSettings,
    UnsafeTarget,
    collect_features,
    query_dns,
    query_whois,
    run_bounded,
    safe_domain,
)


def complete(domain: str = "customer.com") -> OnlineResult:
    """Provide synthetic cache data without imitating real lookup quality."""
    return OnlineResult(
        domain,
        dict(zip(ONLINE_NAMES, [1000.0, 1.0, 2.0, 1.0, 2.0, 60.0, 300.0], strict=True)),
        "ok",
    )


@pytest.fixture
def settings(tmp_path: Path) -> OnlineSettings:
    return OnlineSettings(
        cache_path=tmp_path / "cache.sqlite3",
        cache_ttl=10,
        min_interval=0,
        backoff_base=5,
        backoff_max=20,
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1",
        "http://10.0.0.1",
        "http://169.254.169.254",
        "http://192.0.2.1",
        "http://[::1]",
        "http://[fe80::1]",
        "http://[fc00::1]",
        "http://8.8.8.8",
        "http://2130706433",
        "http://127.1",
        "http://0x7f000001",
        "http://0177.0.0.1",
        "http://localhost",
        "http://printer.local",
        "http://host.internal",
        "http://router.lan",
        "http://host.test",
        "http://host.invalid",
        "http://host.example",
        "http://example.com",
        "http://hidden.onion",
        "http://com",
        "http://host.unknownsuffix",
        "http://bad_host.com",
    ],
)
def test_unsafe_inputs_do_not_reach_network(
    url: str, settings: OnlineSettings, monkeypatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Unsafe target reached transport")

    monkeypatch.setattr(online, "run_bounded", forbidden)
    with pytest.raises(ValueError):
        OnlineLookup(settings).lookup(url)


def test_registered_domain_and_idna() -> None:
    assert safe_domain("https://login.shop.customer.co.uk:443/a?next=internal") == "customer.co.uk"
    assert safe_domain("https://tenant.github.io/page") == "github.io"
    assert safe_domain("https://b\u00fccher.de/") == "xn--bcher-kva.de"


def test_dns_exact_question_and_no_alias_follow(settings: OnlineSettings, monkeypatch) -> None:
    calls = []

    def response(query, where, timeout):
        calls.append((query.question[0].name.to_text(), where, timeout))
        message = dns.message.make_response(query)
        message.answer.append(dns.rrset.from_text("customer.com.", 60, "IN", "A", "8.8.8.8"))
        return message, False

    monkeypatch.setattr(online.dns.query, "udp_with_fallback", response)
    assert query_dns("customer.com", "A", settings) == (["8.8.8.8"], 60)
    assert calls == [("customer.com.", settings.dns_server, settings.dns_timeout)]

    def alias(query, *args, **kwargs):
        message = dns.message.make_response(query)
        message.answer.append(
            dns.rrset.from_text("customer.com.", 60, "IN", "CNAME", "private.local.")
        )
        return message, False

    monkeypatch.setattr(online.dns.query, "udp_with_fallback", alias)
    with pytest.raises(LookupFailure, match="alias_not_followed"):
        query_dns("customer.com", "A", settings)


def test_whois_numeric_server_and_no_referral(monkeypatch) -> None:
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.recv.side_effect = [
        b"Domain Name: CUSTOMER.COM\r\nCreation Date: 2020-01-01T00:00:00Z\r\n"
        b"Registrar: Example Registrar\r\nRegistrar WHOIS Server: whois.private.local\r\n",
        b"",
    ]
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **kw: [(socket.AF_INET, 1, 6, "", ("8.8.8.8", 43))]
    )
    transport = MagicMock(return_value=connection)
    monkeypatch.setattr(socket, "create_connection", transport)
    entry = query_whois("customer.com", 0.5)
    transport.assert_called_once_with(("8.8.8.8", 43), timeout=0.5)
    connection.sendall.assert_called_once_with(b"customer.com\r\n")
    assert entry["creation_date"].year == 2020
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **kw: [(socket.AF_INET, 1, 6, "", ("10.0.0.1", 43))]
    )
    with pytest.raises(LookupFailure, match="server_unavailable"):
        query_whois("customer.com", 0.5)
    assert transport.call_count == 1


def test_features_and_partial_failure(settings: OnlineSettings, monkeypatch) -> None:
    calls = []
    records = {
        "A": (["8.8.8.8"], 60),
        "AAAA": ([], 0),
        "MX": (["10 mail.customer.com."], 120),
        "NS": (["ns.customer.com."], 300),
    }

    def dns(domain, kind, limits):
        calls.append((domain, kind))
        return records[kind]

    monkeypatch.setattr(online, "query_dns", dns)
    monkeypatch.setattr(
        online,
        "query_whois",
        lambda d, t: {
            "creation_date": [datetime(2021, 1, 1), datetime(2020, 1, 1, tzinfo=UTC)],
            "registrar": "Registrar",
        },
    )
    result = collect_features("customer.com", settings)
    assert result.complete and result.features["domain_age_days"] > 1500
    assert result.features["registrar_present"] == 1
    assert result.features["dns_ttl_min"] == 60
    assert result.features["dns_ttl_mean"] == 160
    assert calls == [("customer.com", kind) for kind in ("A", "AAAA", "MX", "NS")]
    monkeypatch.setattr(online, "query_whois", lambda d, t: {"creation_date": None})
    partial = collect_features("customer.com", settings)
    assert not partial.complete and partial.features["dns_a_count"] == 1
    assert partial.features["domain_age_days"] is None
    assert partial.reason == "whois_creation_missing"


@pytest.mark.parametrize(
    "address,kind",
    [
        ("10.0.0.1", "A"),
        ("127.0.0.1", "A"),
        ("169.254.169.254", "A"),
        ("192.0.2.1", "A"),
        ("::1", "AAAA"),
        ("fc00::1", "AAAA"),
        ("224.0.0.1", "A"),
        ("ff02::1", "AAAA"),
    ],
)
def test_private_dns_answers_block_whois(address, kind, settings, monkeypatch) -> None:
    monkeypatch.setattr(
        online, "query_dns", lambda d, k, s: ([address], 60) if k == kind else ([], 0)
    )
    whois = MagicMock(side_effect=AssertionError("WHOIS must not run"))
    monkeypatch.setattr(online, "query_whois", whois)
    with pytest.raises(UnsafeTarget):
        collect_features("customer.com", settings)
    whois.assert_not_called()


@pytest.mark.parametrize("requires_kill", [False, True])
def test_hard_timeout_terminates_worker(settings, monkeypatch, requires_kill) -> None:
    receiver, sender = MagicMock(), MagicMock()
    receiver.poll.return_value = False
    process = MagicMock()
    process.pid = 123
    process.is_alive.side_effect = [True, requires_kill]
    context = MagicMock()
    context.Pipe.return_value = (receiver, sender)
    context.Process.return_value = process
    monkeypatch.setattr(online.multiprocessing, "get_context", lambda method: context)
    result = run_bounded("customer.com", settings)
    assert result.reason == "lookup_timeout"
    receiver.poll.assert_called_once_with(settings.total_timeout)
    process.terminate.assert_called_once()
    if requires_kill:
        process.kill.assert_called_once()
        assert process.join.call_count == 2
    else:
        process.kill.assert_not_called()
    process.close.assert_called_once()
    receiver.close.assert_called_once()


def test_cache_hits_expiry_and_domain_key(settings, monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(online.time, "time", lambda: clock[0])
    runner = MagicMock(side_effect=lambda d, s: complete(d))
    monkeypatch.setattr(online, "run_bounded", runner)
    lookup = OnlineLookup(settings)
    assert lookup.lookup("https://one.customer.com/path").complete
    hit = lookup.lookup("https://two.customer.com/other?x=1")
    assert hit.cached and hit.domain == "customer.com" and runner.call_count == 1
    clock[0] = 111.0
    assert lookup.lookup("customer.com").complete
    assert runner.call_count == 2
    exported = settings.cache_path.with_suffix(".csv")
    lookup.export(exported)
    assert exported.exists() and "https://" not in exported.read_text()


def test_negative_cache_exponential_backoff(settings, monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(online.time, "time", lambda: clock[0])
    runner = MagicMock(side_effect=lambda d, s: OnlineResult(d, reason="lookup_timeout"))
    monkeypatch.setattr(online, "run_bounded", runner)
    lookup = OnlineLookup(settings)
    assert not lookup.lookup("customer.com").complete
    clock[0] = 104.0
    assert lookup.lookup("customer.com").cached and runner.call_count == 1
    clock[0] = 106.0
    lookup.lookup("customer.com")
    with sqlite3.connect(settings.cache_path) as db:
        assert db.execute("SELECT expires_at,failures FROM snapshots").fetchone() == (116.0, 2)
    clock[0] = 117.0
    lookup.lookup("customer.com")
    with sqlite3.connect(settings.cache_path) as db:
        assert db.execute("SELECT expires_at,failures FROM snapshots").fetchone() == (137.0, 3)


def test_global_rate_lease_and_busy(settings, monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(online.time, "time", lambda: clock[0])
    runner = MagicMock(side_effect=lambda d, s: complete(d))
    monkeypatch.setattr(online, "run_bounded", runner)
    lookup = OnlineLookup(replace(settings, min_interval=50))
    assert lookup.lookup("customer.com").complete
    second = OnlineLookup(lookup.settings)
    assert second.lookup("otherdomain.com").reason == "rate_limited"
    assert runner.call_count == 1
    clock[0] = 151.0
    assert second.lookup("otherdomain.com").complete
    online._PROCESS_GATE.acquire()
    try:
        assert lookup.lookup("third-domain.com").reason == "lookup_busy"
    finally:
        online._PROCESS_GATE.release()
    with sqlite3.connect(settings.cache_path) as db:
        db.execute("UPDATE limiter SET owner='external', lease_until=200, next_start=0 WHERE id=1")
    assert lookup.lookup("third-domain.com").reason == "rate_limited"
    with sqlite3.connect(settings.cache_path) as db:
        assert db.execute("SELECT owner FROM limiter WHERE id=1").fetchone()[0] == "external"
    clock[0] = 201.0
    assert lookup.lookup("third-domain.com").complete


def test_unsafe_answer_is_cached_and_rejected(settings, monkeypatch) -> None:
    runner = MagicMock(
        side_effect=lambda d, s: OnlineResult(d, status="unsafe", reason="nonpublic_address")
    )
    monkeypatch.setattr(online, "run_bounded", runner)
    lookup = OnlineLookup(settings)
    for _ in range(2):
        with pytest.raises(UnsafeTarget):
            lookup.lookup("customer.com")
    assert runner.call_count == 1


def test_cache_failure_skips_network(settings, monkeypatch) -> None:
    settings.cache_path.write_bytes(b"not a sqlite database")
    runner = MagicMock(side_effect=AssertionError("Must not look up when cache is unavailable"))
    monkeypatch.setattr(online, "run_bounded", runner)
    result = OnlineLookup(settings).lookup("customer.com")
    assert result.reason == "cache_unavailable" and not result.complete
    runner.assert_not_called()


def test_corrupt_cached_features_degrade(settings, monkeypatch) -> None:
    monkeypatch.setattr(online, "run_bounded", lambda d, s: complete(d))
    lookup = OnlineLookup(settings)
    lookup.lookup("customer.com")
    with sqlite3.connect(settings.cache_path) as db:
        malformed = dict(complete().features)
        malformed["domain_age_days"] = "not numeric"
        db.execute("UPDATE snapshots SET features=?", (json.dumps(malformed),))
    result = lookup.lookup("customer.com")
    assert not result.complete


def test_infrastructure_settings_reject_private_resolver(settings) -> None:
    with pytest.raises(ValueError, match="public"):
        replace(settings, dns_server="127.0.0.1")
    with pytest.raises(ValueError, match="positive"):
        replace(settings, total_timeout=0)
    with pytest.raises(ValueError, match="public"):
        replace(settings, dns_server="224.0.0.1")


def test_transport_boundaries_reject_subdomain_queries(settings, monkeypatch) -> None:
    network = MagicMock(side_effect=AssertionError("No subdomain question allowed"))
    monkeypatch.setattr(online.dns.query, "udp_with_fallback", network)
    monkeypatch.setattr(socket, "getaddrinfo", network)
    with pytest.raises(UnsafeTarget):
        query_dns("sub.customer.com", "A", settings)
    with pytest.raises(UnsafeTarget):
        query_whois("sub.customer.com", 1)
    network.assert_not_called()
