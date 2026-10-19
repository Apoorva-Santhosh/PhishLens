"""Lexical signals are inexpensive clues, never proof of phishing."""

import ipaddress
import math
from collections import Counter
from urllib.parse import SplitResult, urlsplit

import tldextract

from phishlens.config import MAX_URL_LENGTH

# Empty URLs disable suffix-list downloads, including on a cold machine.
_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
)
SHORTENERS = frozenset({"bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "is.gd"})
RISK_TLDS = frozenset({"zip", "mov", "xyz", "top", "click", "work"})
KEYWORDS = ("login", "verify", "secure", "account", "update")
FEATURE_DESCRIPTIONS = {
    "url_length": "Long URLs can conceal misleading destinations.",
    "host_length": "Long hosts can disguise the actual domain.",
    "path_length": "Long paths can hide credential collection endpoints.",
    "dot_count": "Extra dots can imitate trusted domain hierarchies.",
    "subdomain_count": "Many subdomains can obscure the registered domain.",
    "digit_count": "Many digits may indicate generated identifiers.",
    "digit_ratio": "The share of digits captures generated-looking URLs.",
    "hyphen_count": "Hyphens are sometimes used in brand impersonation.",
    "at_count": "An at sign can disguise the actual host with user information.",
    "question_count": "Query delimiters indicate parameterized URLs.",
    "equals_count": "Assignments measure query complexity.",
    "ampersand_count": "Many parameters can conceal a destination.",
    "percent_count": "Encoding can obscure readable URL content.",
    "ip_host": "An IP host bypasses recognizable domain names.",
    "uses_https": "HTTPS encrypts traffic but does not establish trust.",
    "punycode_host": "Punycode may encode lookalike Unicode characters.",
    "non_ascii_host": "Unicode host characters may visually imitate a brand.",
    "shortener_host": "Shorteners conceal the ultimate destination.",
    "suspicious_keyword_count": "Account-action language can accompany credential theft.",
    "host_entropy": "High host entropy can indicate generated names.",
    "path_entropy": "High path entropy can indicate obfuscated identifiers.",
    "risky_tld": "A small illustrative TLD list is a weak, biased heuristic.",
    "has_port": "Explicit ports can indicate unusual hosting.",
}
FEATURE_NAMES = tuple(FEATURE_DESCRIPTIONS)


def parse_url(url: str) -> SplitResult:
    """Accept scheme-less dataset URLs while rejecting malformed hosts."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a nonempty string")
    url = url.strip()
    if len(url) > MAX_URL_LENGTH or any(c.isspace() or ord(c) < 32 for c in url):
        raise ValueError(f"URL must have no whitespace and at most {MAX_URL_LENGTH} characters")
    parsed = urlsplit(url if "://" in url else "//" + url)
    if parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported")
    if not parsed.hostname:
        raise ValueError("URL must contain a host")
    try:
        parsed.port
        parsed.hostname.encode("idna")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("URL has an invalid host or port") from exc
    return parsed


def registered_domain(url: str) -> str:
    """Group related hosts, including private suffix tenants, without DNS."""
    host = parse_url(url).hostname.lower().rstrip(".")
    host = host.encode("idna").decode("ascii")
    extracted = _EXTRACT(host)
    return extracted.top_domain_under_public_suffix or host


def entropy(text: str) -> float:
    """Measure character unpredictability; empty strings have zero entropy."""
    if not text:
        return 0.0
    return -sum((n / len(text)) * math.log2(n / len(text)) for n in Counter(text).values())


def extract_features(url: str) -> dict[str, float]:
    """Return a fixed-order numerical mapping for training and inference."""
    url = url.strip()
    parsed = parse_url(url)
    host = parsed.hostname.lower().rstrip(".")
    ascii_host = host.encode("idna").decode("ascii")
    extracted = _EXTRACT(ascii_host)
    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False
    digits = sum(c.isdigit() for c in url)
    values = [
        len(url),
        len(host),
        len(parsed.path),
        url.count("."),
        len(extracted.subdomain.split(".")) if extracted.subdomain and not is_ip else 0,
        digits,
        digits / len(url),
        *(url.count(c) for c in "-@?=&%"),
        is_ip,
        parsed.scheme.lower() == "https",
        any(part.startswith("xn--") for part in ascii_host.split(".")),
        not host.isascii(),
        ascii_host in SHORTENERS,
        sum(url.lower().count(word) for word in KEYWORDS),
        entropy(host),
        entropy(parsed.path),
        extracted.suffix in RISK_TLDS,
        parsed.port is not None,
    ]
    return dict(zip(FEATURE_NAMES, map(float, values), strict=True))
