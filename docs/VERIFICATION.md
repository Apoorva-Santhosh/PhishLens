# Stages 1–6 verification

Verified through Stage 6 locally on 2026-10-19. Stage 7 remains pending.

| Stage | Verified deliverables |
| --- | --- |
| 1 — Offline URLs | Validated/deduplicated CSV; 23 ordered lexical features; registered-domain partitions; LR/forest/XGBoost comparison; independent calibration; selected random forest on replacement data; preserved PhiUSIIL/XGBoost baseline; EDA notebook |
| 2 — URL explanations | TreeExplainer and scaled LinearExplainer paths; correct calibrated-wrapper handling; fixed phishing-class orientation; deterministic top features; trusted artifact loading and feature-schema validation |
| 3 — Emails | Local preparation and provenance; HTML/address cleaning; 100,000-character bound; unigram/bigram TF-IDF + logistic pipeline; isolated calibration/evaluation; registered email artifact; exact term contributions and cleaned-text highlight spans |
| 4 — Online URLs | Registered-domain-only DNS/WHOIS; public-address screening; SQLite TTL/negative cache; global lease/concurrency/rate limits; hard worker deadlines; resumable bounded collection; file-only extended-model training; 30-feature SHAP; predictor/scoring with explicit offline fallback |
| 5 — API | FastAPI lifespan model loading; health/version/experimental metadata; typed URL/email endpoints; cleaned-text highlights; byte/character limits; per-client rate limiting; private error handling; structured request logs; TestClient coverage |
| 6 — Dashboard | Reference-inspired navy/cyan Streamlit layout; URL/email forms; HTTPX API boundary; radial risk chart; signed feature chart and reasons; escaped overlapping email highlights; availability, loading, error and fallback states; responsive layout |

Checks completed:

- `python -m pytest -q --tb=short`: **135 passed** (all test DNS/WHOIS calls mocked).
- `python -m ruff check .`: clean.
- `python -m ruff format --check .`: clean.
- URL and email artifacts load and match their current training CSV hashes.
- Historical XGBoost explanations were verified on 64 held-out PhiUSIIL URLs.
  The replacement random forest's eight reserved probes have deterministic
  explanations; its false-positive audit covers 67,110 labeled legitimate deep links.
- Email normalized-token partitions are disjoint. On 64 held-out emails, repeated
  explanations were identical, probabilities finite, and highlight offsets valid.
  Maximum reconstruction error against base-model log-odds: approximately `1.16e-14`.
- Named email artifacts coexist with URL artifacts; registry regression tests pass.
- The unused pre-cleaned `phishing_email.csv` is deleted. Current Kaggle URL and six email
  source CSVs remain for reproducible preparation; superseded PhiUSIIL CSVs and
  the duplicate ZIP archive were removed. Dataset README matches retained files.

Local evidence: `artifacts/url_report.md`, `artifacts/email_report.md`, readable model
metadata, provenance JSON files, and `artifacts/stage3_verification.json`. Data and
model artifacts are Git-ignored and must be regenerated in a fresh checkout.

Stage 4 evidence: the public-domain probe and full online CLI prediction succeeded
without fetching the URL. A separate online artifact was trained and registered
from the frozen 100-domain acquisition. Successful snapshots cover only 30 domains/
46 source URLs. Its seven-row test set misses the sole positive, so this is an
experimental integration pilot, not a validated online detector. See
`artifacts/url_online_report.md`, `stage4_live_prediction.json` and
`stage4_verification.json`. Tests cover DNS/WHOIS boundaries, private/IPv6/multicast
addresses, cache corruption, expiry/backoff, rate limits, worker termination,
resume after interruption, local-only fitting, online explanations and exact
offline-fallback probability/model selection.

Qualification: implementation completeness is distinct from deployment quality.
The PhiUSIIL homepage shortcut was addressed with real legitimate deep links from
the replacement dataset; all eight reserved benign-pattern probes are below the
suspicious cutoff. The current URL model's low recall still limits operational use.
Email positive labels include spam/fraud, and random document partitions do not
test generalization across sources or time. See [MODEL_CARD](MODEL_CARD.md).

Stage 5 checks cover startup loading once, loaded model metadata, URL/email
contracts, cleaned-text offsets, input rejection, malformed JSON, chunked body
limits, unsafe online targets, fallback response fields, missing models, rate
window expiry, generic internal errors and log privacy. Saved-model TestClient
verification returned a 29/100 legitimate URL prediction and a 3/100 legitimate
email prediction; an injected timeout matched offline probability, model version
and feature contributions exactly. No external lookups were used in that check.
Evidence is in `artifacts/stage5_verification.json`. Installed Starlette emits a
TestClient HTTPX deprecation warning; all tests pass with the permitted HTTPX stack.
The real Uvicorn localhost smoke check also passed health, prediction and OpenAPI
requests (200) and unsafe-online-target rejection (422). The temporary server was
stopped after verification. Ruff lint and formatting checks pass across 44 files.

Stage 6 adds 11 tests for real Streamlit form interactions, empty input, successful
URL/email results, clearing/replacing stale evidence, API unavailability, HTTP
timeouts/status errors, payload/schema checks, chart values and safely escaped
overlapping highlights. Headless Edge checked the real dashboard against the
saved-model API at 1440×1000 and 390×844. URL and email predictions rendered,
email highlights appeared, there were no browser errors and no mobile horizontal
overflow. Temporary servers were stopped. Screenshots and the browser report are
saved locally under `artifacts/dashboard_*.png` and `stage6_verification.json`.
Ruff lint and formatting pass across 49 Python files. Browser tooling is used
only for verification and is not an application dependency.
