# Stages 1–7 verification

Verified through Stage 7 locally on 2026-10-19. Model-quality improvements remain separate work.

| Stage | Verified deliverables |
| --- | --- |
| 1 — Offline URLs | Validated/deduplicated CSV; 23 ordered lexical features; registered-domain partitions; LR/forest/XGBoost comparison; independent calibration; selected random forest on replacement data; preserved PhiUSIIL/XGBoost baseline; EDA notebook |
| 2 — URL explanations | TreeExplainer and scaled LinearExplainer paths; correct calibrated-wrapper handling; fixed phishing-class orientation; deterministic top features; trusted artifact loading and feature-schema validation |
| 3 — Emails | Local preparation and provenance; HTML/address cleaning; 100,000-character bound; unigram/bigram TF-IDF + logistic pipeline; isolated calibration/evaluation; registered email artifact; exact term contributions and cleaned-text highlight spans |
| 4 — Online URLs | Registered-domain-only DNS/WHOIS; public-address screening; SQLite TTL/negative cache; global lease/concurrency/rate limits; hard worker deadlines; resumable bounded collection; file-only extended-model training; 30-feature SHAP; predictor/scoring with explicit offline fallback |
| 5 — API | FastAPI lifespan model loading; health/version/experimental metadata; typed URL/email endpoints; cleaned-text highlights; byte/character limits; per-client rate limiting; private error handling; structured request logs; TestClient coverage |
| 6 — Dashboard | Reference-inspired navy/cyan Streamlit layout; URL/email forms; HTTPX API boundary; radial risk chart; signed feature chart and reasons; escaped overlapping email highlights; availability, loading, error and fallback states; responsive layout |
| 7 — Polish | Public-function annotation/docstring audit; full architecture diagram; canonical app entry point and Make alias; isolated clean-checkout install and command verification; compatible TestClient dependencies; warning-free regression checks |

Checks completed:

- `python -m pytest -q -W error --tb=short`: **135 passed, no warnings** (all test DNS/WHOIS calls mocked).
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
Evidence is in `artifacts/stage5_verification.json`. The TestClient dependency
warning observed during Stage 5 was resolved in Stage 7 using compatibility bounds.
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

Stage 7 verification:

- AST audit: 88 public application functions/methods, including constructors,
  callbacks and endpoint helpers, have parameter/return annotations and docstrings.
  Tests and ignored verification scripts are outside this application-code audit.
- Dashboard moved to `app/streamlit_app.py`, with HTTP/chart/style helpers in
  `app/`. `make app` is canonical and `make dashboard` remains an alias. The root
  theme location is documented because Streamlit discovers it from the launch directory.
- README architecture covers offline/online URL and email training, calibration,
  artifacts, API, dashboard, explanations, SQLite cache and offline fallback.
- Starlette `0.52.1` and AnyIO `4.14.2` preserve the permitted HTTPX stack without
  warning suppression or adding HTTPX2. Both are constrained existing FastAPI
  dependencies. `pip check` passes; pytest passes with all warnings treated as errors.
- A fresh local Git clone with the current source changes applied contained no
  datasets, cache, artifacts or virtual environment. A new independent virtual
  environment successfully installed `.[dev]`; all 135 tests, lint and formatting
  checks passed there. The resolved FastAPI version was `0.147.0`; the working
  environment also passed on `0.146.0` with the same Starlette/AnyIO bounds.
- Documented preparation and all three training commands ran against synthetic
  source CSVs and a frozen synthetic online snapshot. All three artifacts were
  created in that disposable checkout. Offline prediction and SHAP CLI commands,
  entry-point help, theme configuration, missing-model 503 behavior, URL/email
  HTTP predictions and both server startup checks passed. No live DNS/WHOIS
  acquisition or historical-baseline audit was run; acquisition/audit help was checked.
- GNU Make is unavailable on this Windows host; documented direct Python command
  equivalents were executed. The Make targets and dashboard alias were reviewed.
- Existing datasets and trained artifacts were preserved. Temporary servers were
  stopped and the disposable checkout/environment removed after verification.

Local evidence: `artifacts/stage7_verification.json` and `stage7_checkout.log`.
The verification used synthetic data to check installation and command wiring;
it provides no new evidence of real-world detection quality.
