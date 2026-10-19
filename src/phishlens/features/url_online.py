"""Registered-domain WHOIS/DNS only: no HTTP, redirect, CNAME or WHOIS referral fetching."""

import ipaddress
import json
import math
import multiprocessing
import re
import socket
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import dns.message
import dns.query
import dns.rcode
import dns.rdatatype
import tldextract
from whois.parser import WhoisEntry

from phishlens.config import (
    ONLINE_BACKOFF_BASE,
    ONLINE_BACKOFF_MAX,
    ONLINE_CACHE_PATH,
    ONLINE_CACHE_TTL,
    ONLINE_DNS_SERVER,
    ONLINE_DNS_TIMEOUT,
    ONLINE_MIN_INTERVAL,
    ONLINE_TOTAL_TIMEOUT,
    ONLINE_WHOIS_TIMEOUT,
)
from phishlens.features.url_offline import parse_url

ONLINE_DESCRIPTIONS = {
    "domain_age_days": "Young domains can be riskier; old domains can be compromised.",
    "registrar_present": "Registrar metadata availability is a weak registry-dependent signal.",
    "dns_a_count": "A-record count describes the registered domain's IPv4 hosting.",
    "dns_mx_count": "MX-record count describes configured mail infrastructure.",
    "dns_ns_count": "NS-record count describes delegated name servers.",
    "dns_ttl_min": "The shortest observed record TTL describes DNS cache duration.",
    "dns_ttl_mean": "Mean TTL describes the registered domain's DNS configuration.",
}
ONLINE_NAMES = tuple(ONLINE_DESCRIPTIONS)
_SUFFIX = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
_PROCESS_GATE = threading.BoundedSemaphore(1)
_RESERVED = {
    "localhost",
    "local",
    "internal",
    "intranet",
    "home",
    "lan",
    "test",
    "invalid",
    "example",
    "onion",
    "arpa",
}


class UnsafeTarget(ValueError):
    """Signal a target that must be rejected rather than looked up or silently accepted."""


class LookupFailure(Exception):
    """Use short reason codes instead of exposing transport exceptions or WHOIS text."""


@dataclass(frozen=True)
class OnlineSettings:
    """Keep operational limits explicit, injectable in tests and bounded in production."""

    cache_path: Path = ONLINE_CACHE_PATH
    cache_ttl: float = ONLINE_CACHE_TTL
    dns_timeout: float = ONLINE_DNS_TIMEOUT
    whois_timeout: float = ONLINE_WHOIS_TIMEOUT
    total_timeout: float = ONLINE_TOTAL_TIMEOUT
    min_interval: float = ONLINE_MIN_INTERVAL
    backoff_base: float = ONLINE_BACKOFF_BASE
    backoff_max: float = ONLINE_BACKOFF_MAX
    dns_server: str = ONLINE_DNS_SERVER

    def __post_init__(self) -> None:
        if (
            any(
                not math.isfinite(v) or v <= 0
                for v in (
                    self.cache_ttl,
                    self.dns_timeout,
                    self.whois_timeout,
                    self.total_timeout,
                    self.backoff_base,
                    self.backoff_max,
                )
            )
            or not math.isfinite(self.min_interval)
            or self.min_interval < 0
        ):
            raise ValueError("Online timeouts/TTLs must be positive; interval must be nonnegative")
        if not is_public_address(self.dns_server):
            raise ValueError("Online DNS resolver must be a public IP address")


@dataclass(frozen=True)
class OnlineResult:
    """Preserve partial data as missing values while making fallback decisions unambiguous."""

    domain: str
    features: dict[str, float | None] = field(default_factory=lambda: dict.fromkeys(ONLINE_NAMES))
    status: str = "failed"
    reason: str | None = None
    cached: bool = False
    observed_at: float | None = None

    @property
    def complete(self) -> bool:
        """Only successful finite snapshots may reach the online classifier."""
        return (
            self.status == "ok"
            and isinstance(self.features, dict)
            and set(self.features) == set(ONLINE_NAMES)
            and all(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
                for value in self.features.values()
            )
        )


def is_public_address(address: str) -> bool:
    """Require public unicast: is_global alone also accepts some multicast addresses."""
    parsed = ipaddress.ip_address(address)
    return parsed.is_global and not (
        parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_private
    )


def safe_domain(url: str) -> str:
    """Reject IP literals/local names and use the ICANN registrable domain, not a tenant suffix."""
    host = parse_url(url).hostname.lower().rstrip(".").encode("idna").decode("ascii")
    if (
        not host
        or host.split(".")[-1] in _RESERVED
        or host in {"example.com", "example.net", "example.org"}
    ):
        raise UnsafeTarget("Online mode requires a public registered domain")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise UnsafeTarget("IP literals are not registered domains; use offline mode")
    if (
        not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host)
        or any(
            not part or len(part) > 63 or part.startswith("-") or part.endswith("-")
            for part in host.split(".")
        )
        or len(host) > 253
    ):
        raise UnsafeTarget("Invalid public hostname")
    # Also reject numeric IP aliases such as 2130706433 or 127.1 before any lookup.
    if all(re.fullmatch(r"(?:0x[0-9a-f]+|[0-9]+)", part) for part in host.split(".")):
        raise UnsafeTarget("Numeric host aliases are not registered domains")
    extracted = _SUFFIX(host)
    if not extracted.suffix or not extracted.domain:
        raise UnsafeTarget("Hostname has no recognized public registration suffix")
    return extracted.top_domain_under_public_suffix


def query_dns(domain: str, record_type: str, settings: OnlineSettings) -> tuple[list[str], int]:
    """Send exactly one name question; never query a CNAME target or fetch a URL."""
    if safe_domain(domain) != domain:
        raise UnsafeTarget("DNS queries must target the registered domain only")
    query = dns.message.make_query(domain + ".", record_type)
    response, _ = dns.query.udp_with_fallback(
        query, settings.dns_server, timeout=settings.dns_timeout
    )
    if response.rcode() != dns.rcode.NOERROR:
        raise LookupFailure("dns_unavailable")
    records = []
    ttls = []
    expected_type = dns.rdatatype.from_text(record_type)
    for rrset in response.answer:
        if rrset.rdtype in {dns.rdatatype.CNAME, dns.rdatatype.DNAME}:
            raise LookupFailure("dns_alias_not_followed")
        if rrset.rdtype == expected_type:
            if rrset.name.to_text().lower().rstrip(".") != domain:
                raise LookupFailure("dns_unexpected_name")
            records.extend(record.to_text() for record in rrset)
            ttls.append(rrset.ttl)
    return records, min(ttls) if ttls else 0


def query_whois(domain: str, timeout: float) -> dict[str, Any]:
    """Query one registry over port 43, with no referrals and no raw response persistence."""
    if safe_domain(domain) != domain:
        raise UnsafeTarget("WHOIS queries must target the registered domain only")
    tld = domain.rsplit(".", 1)[1]
    server = {"com": "whois.verisign-grs.com", "net": "whois.verisign-grs.com"}.get(
        tld, f"{tld}.whois-servers.net"
    )
    # Infrastructure name resolution is not a lookup of a URL/subdomain. Validate all
    # server addresses and connect to a numeric IP to avoid a second resolution/rebinding.
    addresses = socket.getaddrinfo(server, 43, type=socket.SOCK_STREAM)
    if not addresses or any(not is_public_address(a[4][0]) for a in addresses):
        raise LookupFailure("whois_server_unavailable")
    deadline = time.monotonic() + timeout
    response = bytearray()
    with socket.create_connection((addresses[0][4][0], 43), timeout=timeout) as connection:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("WHOIS deadline")
        connection.settimeout(remaining)
        connection.sendall((domain + "\r\n").encode("ascii"))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("WHOIS deadline")
            connection.settimeout(remaining)
            chunk = connection.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
            if len(response) > 262_144:
                raise LookupFailure("whois_response_too_large")
    return dict(WhoisEntry.load(domain, response.decode("utf-8", errors="replace")))


def collect_features(domain: str, settings: OnlineSettings) -> OnlineResult:
    """Screen apex addresses before WHOIS; partial/missing metadata causes offline fallback."""
    if safe_domain(domain) != domain:
        raise UnsafeTarget("Lookups must target the registered domain only")
    features: dict[str, float | None] = dict.fromkeys(ONLINE_NAMES)
    try:
        a, a_ttl = query_dns(domain, "A", settings)
        aaaa, _ = query_dns(domain, "AAAA", settings)
        if any(not is_public_address(address) for address in a + aaaa):
            raise UnsafeTarget("Registered domain resolves to a nonpublic address")
        if not a and not aaaa:
            raise LookupFailure("no_public_address")
        mx, mx_ttl = query_dns(domain, "MX", settings)
        ns, ns_ttl = query_dns(domain, "NS", settings)
        ttls = [ttl for records, ttl in ((a, a_ttl), (mx, mx_ttl), (ns, ns_ttl)) if records]
        features.update(
            {
                "dns_a_count": float(len(a)),
                "dns_mx_count": float(len(mx)),
                "dns_ns_count": float(len(ns)),
                "dns_ttl_min": float(min(ttls, default=0)),
                "dns_ttl_mean": float(sum(ttls) / len(ttls)) if ttls else 0.0,
            }
        )
    except UnsafeTarget:
        raise
    except Exception as exc:
        reason = str(exc) if isinstance(exc, LookupFailure) else "dns_failed"
        return OnlineResult(domain, features, reason=reason)
    try:
        entry = query_whois(domain, settings.whois_timeout)
        dates = entry.get("creation_date")
        dates = dates if isinstance(dates, list) else [dates]
        usable = []
        for value in dates:
            if isinstance(value, datetime):
                usable.append(
                    value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
                )
        if not usable:
            raise LookupFailure("whois_creation_missing")
        age = (datetime.now(UTC) - min(usable)).total_seconds() / 86400
        if age < 0:
            raise LookupFailure("whois_creation_invalid")
        features["domain_age_days"] = age
        registrar = entry.get("registrar")
        registrars = registrar if isinstance(registrar, list) else [registrar]
        features["registrar_present"] = float(
            any(isinstance(value, str) and bool(value.strip()) for value in registrars)
        )
    except Exception as exc:
        reason = str(exc) if isinstance(exc, LookupFailure) else "whois_failed"
        return OnlineResult(domain, features, reason=reason)
    return OnlineResult(domain, features, status="ok")


def _worker(domain: str, settings: OnlineSettings, sender: Connection) -> None:
    try:
        result = collect_features(domain, settings)
    except UnsafeTarget:
        result = OnlineResult(domain, status="unsafe", reason="nonpublic_address")
    except Exception:
        result = OnlineResult(domain, reason="lookup_failed")
    try:
        sender.send(result)
    finally:
        sender.close()


def run_bounded(domain: str, settings: OnlineSettings) -> OnlineResult:
    """Kill timed-out workers rather than abandoning uninterruptible WHOIS threads."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(domain, settings, sender), daemon=True)
    try:
        process.start()
        sender.close()
        if receiver.poll(settings.total_timeout):
            return receiver.recv()
        return OnlineResult(domain, reason="lookup_timeout")
    except (EOFError, OSError):
        return OnlineResult(domain, reason="lookup_failed")
    finally:
        receiver.close()
        sender.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.5)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.5)
            process.close()


class OnlineLookup:
    """Serialize misses globally with a SQLite lease; cache hits never consume a lookup slot."""

    def __init__(self, settings: OnlineSettings | None = None) -> None:
        """Use a file-backed cache shared by precomputation and future API processes."""
        self.settings = settings or OnlineSettings()

    def _connect(self) -> sqlite3.Connection:
        self.settings.cache_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.settings.cache_path, timeout=0.1)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("""CREATE TABLE IF NOT EXISTS snapshots (
                domain TEXT PRIMARY KEY, features TEXT NOT NULL, status TEXT NOT NULL,
                reason TEXT, observed_at REAL NOT NULL, expires_at REAL NOT NULL,
                failures INTEGER NOT NULL)""")
            connection.execute("""CREATE TABLE IF NOT EXISTS limiter (
                id INTEGER PRIMARY KEY CHECK (id=1), owner TEXT,
                lease_until REAL, next_start REAL)""")
            connection.commit()
        except sqlite3.Error:
            connection.close()
            raise
        return connection

    def lookup(self, url: str) -> OnlineResult:
        """Reject unsafe inputs; return partial snapshots on failure with bounded retry backoff."""
        domain = safe_domain(url)
        settings = self.settings
        connection = None
        acquired = False
        owner = uuid.uuid4().hex
        claimed = False
        try:
            connection = self._connect()
            now = time.time()
            row = connection.execute("SELECT * FROM snapshots WHERE domain=?", (domain,)).fetchone()
            if row and row["expires_at"] > now:
                if row["status"] == "unsafe":
                    raise UnsafeTarget("Cached domain resolves to a nonpublic address")
                cached = OnlineResult(
                    domain,
                    json.loads(row["features"]),
                    row["status"],
                    row["reason"],
                    cached=True,
                    observed_at=row["observed_at"],
                )
                if cached.status == "ok" and not cached.complete:
                    return OnlineResult(domain, reason="cache_corrupt", cached=True)
                return cached
            acquired = _PROCESS_GATE.acquire(blocking=False)
            if not acquired:
                return OnlineResult(domain, reason="lookup_busy")
            connection.execute("BEGIN IMMEDIATE")
            lease = connection.execute("SELECT * FROM limiter WHERE id=1").fetchone()
            if lease and (lease["lease_until"] > now or lease["next_start"] > now):
                connection.rollback()
                return OnlineResult(domain, reason="rate_limited")
            connection.execute(
                "INSERT OR REPLACE INTO limiter VALUES (1, ?, ?, ?)",
                (owner, now + settings.total_timeout + 2, now + settings.min_interval),
            )
            connection.commit()
            claimed = True
            try:
                result = run_bounded(domain, settings)
            except Exception:
                result = OnlineResult(domain, reason="lookup_failed")
            observed = time.time()
            failures = 0 if result.complete else min((row["failures"] if row else 0) + 1, 20)
            ttl = (
                settings.cache_ttl
                if result.complete
                else min(settings.backoff_max, settings.backoff_base * (2 ** (failures - 1)))
            )
            connection.execute(
                "INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    domain,
                    json.dumps(result.features, allow_nan=False),
                    result.status,
                    result.reason,
                    observed,
                    observed + ttl,
                    failures,
                ),
            )
            connection.commit()
            if result.status == "unsafe":
                raise UnsafeTarget("Registered domain resolves to a nonpublic address")
            return OnlineResult(
                domain, result.features, result.status, result.reason, observed_at=observed
            )
        except (sqlite3.Error, OSError, ValueError) as exc:
            if isinstance(exc, UnsafeTarget):
                raise
            return OnlineResult(domain, reason="cache_unavailable")
        finally:
            if connection is not None:
                try:
                    if claimed:
                        connection.execute(
                            "UPDATE limiter SET lease_until=0 WHERE id=1 AND owner=?", (owner,)
                        )
                        connection.commit()
                except sqlite3.Error:
                    pass  # A crashed writer's lease expires; no unbounded lock or network retry.
                connection.close()
            if acquired:
                _PROCESS_GATE.release()

    def export(self, destination: Path) -> None:
        """Atomically export cached snapshots as a training file without any network access."""
        import csv
        import os

        connection = self._connect()
        try:
            rows = connection.execute("SELECT * FROM snapshots ORDER BY domain").fetchall()
        finally:
            connection.close()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["domain", "status", "reason", "observed_at", *ONLINE_NAMES]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "domain": row["domain"],
                        "status": row["status"],
                        "reason": row["reason"],
                        "observed_at": row["observed_at"],
                        **json.loads(row["features"]),
                    }
                )
        os.replace(temporary, destination)

    def completed_domains(self) -> set[str]:
        """Resume from the authoritative cache even if the exported CSV was removed."""
        connection = self._connect()
        try:
            rows = connection.execute("SELECT * FROM snapshots WHERE status='ok'").fetchall()
        finally:
            connection.close()
        completed = set()
        for row in rows:
            try:
                result = OnlineResult(row["domain"], json.loads(row["features"]), "ok")
            except ValueError:
                continue
            if result.complete:
                completed.add(row["domain"])
        return completed


def main() -> None:
    """Inspect a bounded domain lookup; never contact the supplied URL's web server."""
    import argparse
    from dataclasses import asdict

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--cache", type=Path, default=ONLINE_CACHE_PATH)
    args = parser.parse_args()
    print(
        json.dumps(
            asdict(OnlineLookup(OnlineSettings(cache_path=args.cache)).lookup(args.url)), indent=2
        )
    )


if __name__ == "__main__":
    main()
