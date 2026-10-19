# PhishLens local data — Stages 1–4

Downloaded data and generated CSVs are Git-ignored. This README is tracked.
Training reads local files only: it never fetches URLs, follows redirects, or uses API keys.
Stage 4's separate collection command performs bounded DNS/WHOIS domain lookups.
Do not publish private emails or URL tokens. Check the source licenses before redistribution.

## Files retained

| File | Purpose |
| --- | --- |
| `phishing_site_urls.csv` | Current raw Kaggle URL source for reproducible preparation |
| `urls.csv` | Converted, two-column URL training input |
| `urls.provenance.json` | URL hashes, label conversion and observed source bias |
| `CEAS_08.csv`, `Enron.csv`, `Ling.csv` | Original email source corpora |
| `Nazario.csv`, `Nigerian_Fraud.csv`, `SpamAssasin.csv` | Original email source corpora |
| `emails.csv` | Prepared subject/body, label and source for email training |
| `emails.provenance.json` | Source hashes, exclusions, duplicates and prepared class counts |
| `online_cache.sqlite3` | Resumable registered-domain WHOIS/DNS cache and shared rate lease |
| `url_online_features.csv` | Frozen feature snapshot used by the separate online trainer |
| `url_online_features.collection.json` | Collection counts and sampling limits |

`phishing_email.csv` was removed: it is an unused, pre-cleaned aggregate with
82,486 rows, the same total as the six source corpora. It was not byte-equivalent
to the source text. The source corpora preserve text and provenance needed by
our reproducible preparation flow; we do not combine the aggregate with them.
The current Kaggle URL CSV and six email source corpora are retained. The duplicate
ZIP, superseded PhiUSIIL source/prepared CSVs and redundant inventory/provenance
were removed during cleanup. Historical model artifacts and source hashes remain
in `artifacts/phiusiil_baseline/`; reproducing that old run requires downloading
PhiUSIIL again. The conversion below is historical documentation.

## Current URL source and conversion

The replacement source is [Kaggle Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls),
downloaded from its public dataset endpoint without credentials on 2026-10-17.
The original `URL,Label` file contains 549,346 rows. `good` maps to 0 and `bad`
maps to 1. We preserve real URL paths and never fabricate legitimate URLs from
popular domains or label URLs found in legitimate emails as safe.

Run `python -m phishlens.models.prepare_url` or `make prepare-url` to rebuild
`urls.csv`. Preparation records 2,175 invalid rows, 3 conflicting-label rows,
41,530 duplicates, and 13,561 rows excluded by reserved sanity-check domains.
492,077 rows remain: 379,222 good and 112,855 bad. Approximately 87.31% of good
URLs and 91.64% of bad URLs have deeper paths; legitimate examples no longer
consist solely of `www` homepages. Domain-aware splitting is unchanged.

All example.com, python.org, wikipedia.org, mozilla.org and github.com groups are
reserved before splitting. Eight fixed benign-pattern probes are diagnostic
checks, not training examples or proof that pages are currently safe. Historical
good/bad labels may be noisy or stale, most URLs lack an explicit scheme, and
this dataset does not establish contemporary phishing-only performance. Read
the source's license/terms before redistributing it.

The historical PhiUSIIL model and reports are preserved under
`artifacts/phiusiil_baseline/`. `artifacts/url_bias_audit.json` compares the baseline
and candidate on held-out legitimate deep links and the reserved probes.

## Historical PhiUSIIL source and conversion (files no longer retained)

[PhiUSIIL, UCI](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset),
Arvind Prasad and Shalini Chandra (2024), CC BY 4.0. The original has 235,795 rows.
Its labels are **1 legitimate, 0 phishing**, opposite to PhishLens. Use only raw
`URL` and `label`; exclude all supplied website and precomputed features.

```python
import pandas as pd

frame = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv", usecols=["URL", "label"])
frame = frame.rename(columns={"URL": "url"})
frame["label"] = 1 - frame["label"]
frame.to_csv("data/urls.csv", index=False)
```

The URL loader requires `url,label`, with **0 legitimate, 1 phishing**. It validates
HTTP(S)/scheme-less URLs, trims outer whitespace, removes exact duplicates and
rejects conflicting labels. It splits by registered domain to prevent domain leakage.
The historical baseline run removed 425 duplicates, leaving 235,370 rows.

All legitimate examples in this source are HTTPS `www` homepages. This creates
severe shortcuts and false positives on normal deep links; see the model card.

Alternative free raw-URL sources: [Kaggle Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)
(`URL` -> `url`, `good` -> 0, `bad` -> 1), or the
[OpenPhish free feed](https://openphish.com/phishing_feeds.html) (label 1), paired
with a dated [Tranco list](https://tranco-list.eu/) (homepage URLs, provisional label 0).
Kaggle may require a free browser sign-in; application code needs no credentials.
Feeds/popular-homepage pairs have substantial source bias and are not ideal evaluation sets.

## Email sources and preparation

The supplied files match the corpus names in
[Kaggle Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
and [Phishing Email Curated Datasets, Zenodo](https://zenodo.org/records/8339691).
The actual download location/date was not supplied; file hashes identify this local copy.
Consult each source's terms and cite the corpus authors before redistribution.

The convention is **0 legitimate, 1 spam/fraud/phishing**. This positive class is
not exclusively confirmed phishing: Enron, Ling, CEAS and SpamAssassin include
spam, while Nazario and Nigerian Fraud supply phishing/fraud examples. Treat the
model as a suspicious-email proxy and do not claim phishing-only performance.

Preparation combines subject and body, retaining `source`. Sender, recipient,
dates and supplied URL indicators are excluded. Cleaning strips HTML and hidden
script/style text, replaces URLs/emails with tokens, and lowercases. SHA-256 keys
of the resulting token sequences detect duplicates even across source files.
All rows with conflicting labels for one normalized text are excluded and counted.
Empty/unusable messages and texts over 100,000 characters are excluded, not truncated.
The prediction cleaner rejects these inputs with clear errors.

Current preparation: 82,486 source rows; 21 oversized and 2 unusable messages excluded;
5,637 duplicate normalized rows removed; 0 conflicting rows; 76,826 retained
(39,112 legitimate, 37,714 spam/fraud/phishing). Exact token-sequence deduplication
prevents identical-document leakage; near-duplicate campaigns and source bias remain.

Prepared/custom CSV contract: `text,label`, optionally `source`. The generic loader
rejects unknown labels, empty/oversized inputs and normalized-label conflicts.
The pipeline fits unigrams/bigrams and IDF only on training documents. Calibration,
validation and test partitions are independent stratified document splits. They are
not source-held-out or temporal evaluations. Corpus age and English-heavy content
limit generalization to current attacks and other languages.

## Reproduce

From the repository root, with the environment activated:

```sh
python -m phishlens.models.prepare_email
python -m phishlens.models.prepare_url
python -m phishlens.models.train_url
python -m phishlens.models.train_email
python -m pytest
python -m ruff check .
```

`make prepare-email` prepares the email CSV. `make train` trains both models;
`make train-url` and `make train-email` train one. Source files are never modified.
Reports, model artifacts and readable metadata are written under `artifacts/`.
Only load trusted joblib files.

## Online feature snapshot (Stage 4)

```sh
python -m phishlens.models.precompute_online --max-domains 100
python -m phishlens.models.train_url_online
```

The collection step, not training, performs network lookups. It samples unique
ICANN registered domains from the URL CSV, interleaving shuffled classes with seed
42, and visits at most the specified domain limit. No URL is opened. Four apex DNS
questions (A, AAAA for safety, MX, NS) screen public addresses; aliases are not
followed. Registry WHOIS queries only the registered domain, never referral servers.
The stored features are domain age, registrar presence, A/MX/NS counts and minimum/
mean RRset TTL. No WHOIS contact text or full URLs are persisted in the cache.

Keep the DB to resume. An atomic CSV export is written after each domain; deleted
exports can be rebuilt from cached rows without recollecting completed domains.
`--refresh` honors runtime TTL/backoff instead of bypassing rate limits. Snapshot
columns: `domain,status,reason,observed_at` and the seven ordered online features.
Missing fields remain blank; only complete `status=ok` rows enter training. Frozen
snapshots can be used for reproducible training after runtime cache TTL expires.

The local pilot selected 100 domains: 30 complete, 69 failed and one unsafe.
7682 URL rows were skipped as unsuitable for online-domain lookup. Only 46 source
URLs matched complete snapshots. Test support is seven URLs with one positive;
the pilot detected none at threshold 0.5. This is a functioning acquisition/training
pilot, not a validated classifier. Many historical domains are gone, privacy-redacted
or use unsupported registry formats, producing severe lookup-success selection bias.
Current metadata can also describe a different owner from the historical label.
