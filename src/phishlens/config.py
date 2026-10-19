"""Shared defaults keep training reproducible across entry points."""

from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "urls.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MAX_URL_LENGTH = 8192
MAX_EMAIL_LENGTH = 100_000
EMAIL_DATA_PATH = ROOT / "data" / "emails.csv"
EMAIL_MAX_FEATURES = 30_000
SUSPICIOUS_CUTOFF = 0.35
PHISHING_CUTOFF = 0.70
ONLINE_CACHE_PATH = ROOT / "data" / "online_cache.sqlite3"
ONLINE_SNAPSHOT_PATH = ROOT / "data" / "url_online_features.csv"
ONLINE_CACHE_TTL = 86_400.0
ONLINE_DNS_TIMEOUT = 1.0
ONLINE_WHOIS_TIMEOUT = 3.0
ONLINE_TOTAL_TIMEOUT = 8.0
ONLINE_MIN_INTERVAL = 2.0
ONLINE_BACKOFF_BASE = 30.0
ONLINE_BACKOFF_MAX = 3_600.0
ONLINE_DNS_SERVER = "1.1.1.1"
ONLINE_MAX_DOMAINS = 100
