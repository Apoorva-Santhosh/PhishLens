# PhishLens URL and email models

Status: Stages 1–4 implemented, including an experimental online URL pilot.
The current offline URL model was retrained on Kaggle
Phishing Site URLs on 2026-10-17 to address the PhiUSIIL homepage shortcut.
Data and artifacts are Git-ignored; metrics below describe local runs.

## Current URL training run — homepage-bias remediation

Source: [Kaggle Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls).
Download: 549,346 rows; `good` -> 0, `bad` -> 1. The preparation report records
2,175 invalid rows, 3 conflicting-label rows, 41,530 duplicates and 13,561 excluded
sanity-domain rows. 492,077 remain: 379,222 good, 112,855 bad. Real legitimate
paths are retained: 87.31% of legitimate URLs have non-homepage paths, and only
7.94% use a `www` host. We did not fabricate paths or relabel links from emails.

The eight diagnostic benign-pattern probes cover example.com, python.org,
wikipedia.org, mozilla.org and github.com. All five registered domains are
excluded before splitting. The probes were not used for model selection.
Registered-domain-disjoint partitions: fitting 211,591 rows, calibration 81,356,
validation 101,097 and testing 98,033. Seed and calibration method are unchanged.

| Model | Validation ROC-AUC | Validation positive recall |
| --- | ---: | ---: |
| Logistic regression | 0.816179 | 35.8893% |
| Random forest (selected) | 0.897275 | 56.8096% |
| XGBoost | 0.889522 | 48.8140% |

| Held-out test metric | Value |
| --- | ---: |
| Precision | 87.8074% |
| Positive recall | 56.1034% |
| F1 | 68.4632% |
| ROC-AUC | 0.897275 |

At threshold 0.5, confusion matrix `[[75189, 1651], [9303, 11890]]`: 1,651 false
alerts and 9,303 missed positive URLs. The model has insufficient recall for
automated enforcement. The lower scores are not a measured regression on the
same test set: this is a different, harder dataset with different class structure.

The original example URL's calibrated probability changed from 99.9769% to
28.8402%. All eight reserved benign-pattern probes are below the 35% suspicious
cutoff. This fixes the observed homepage shortcut on those probes without a
whitelist or threshold adjustment. It does not establish safety of actual pages.

On the new test partition's 67,110 labeled legitimate deep links, the baseline
flagged 100% and the replacement flagged 2.2545% at threshold 0.5. This is a
diagnostic cohort comparison, not matched independent evaluation of both models:
the historical baseline's fitting domains may overlap the replacement dataset.
The replacement's own fitting/calibration/validation/test domains are disjoint.
Most replacement URLs omit schemes, so the baseline cohort comparison also
reflects protocol representation shift and does not isolate the effect of paths.

Limitations: historic/noisy good/bad labels may include threats beyond confirmed
phishing; most source URLs omit a scheme; source, language and temporal bias
remain. Domain separation is not independent-source or future-attack evaluation.
Never interpret the model as a guarantee of URL safety.

Source/output hashes and structural statistics: `data/urls.provenance.json`.
Comparison evidence: `artifacts/url_bias_audit.json`. Old model and reports:
`artifacts/phiusiil_baseline/`; old prepared input: the PhiUSIIL source hashes in the baseline provenance (old CSVs removed).

## Historical PhiUSIIL training run (superseded)

Source: [PhiUSIIL, UCI](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset),
Arvind Prasad and Shalini Chandra (2024), CC BY 4.0. Original labels were inverted
to PhishLens convention (`0` legitimate, `1` phishing). Only raw URLs and labels
were retained; no supplied website or precomputed features were used.
All 235,795 source rows passed syntax validation. Removing 425 duplicate URLs
left 235,370 rows. Training used 102,901 rows, calibration 36,392, validation
49,460 and testing 46,617, with disjoint registered domains and seed 42.

| Model | Validation ROC-AUC | Validation phishing recall |
| --- | ---: | ---: |
| Logistic regression | 0.995180 | 96.4446% |
| Random forest | 0.997931 | 99.2889% |
| XGBoost (selected) | 0.998308 | 99.2756% |

XGBoost won by validation ROC-AUC; the recall tiebreaker was not needed.
Random forest had slightly higher recall, but the specified selection rule
prioritizes ROC-AUC. The selected calibrated model was not refitted after selection.

| Held-out test metric | Value |
| --- | ---: |
| Precision | 99.9122% |
| Phishing recall | 99.1748% |
| F1 | 99.5422% |
| ROC-AUC | 0.997904 |

Test confusion matrix: `[[27089, 17], [161, 19350]]`, rows true class and columns
predicted class in order legitimate/phishing. At threshold 0.5 there were 17
false phishing alerts and 161 missed phishing URLs. These metrics do not establish
deployment performance or calibration quality under changing prevalence.

**Observed source bias:** all 134,850 original legitimate URLs use HTTPS. Of
100,945 original phishing URLs, 51,749 use HTTP and 49,196 use HTTPS. Thus the
dataset lets the model associate HTTP strongly with phishing, even though HTTP
alone does not prove phishing and HTTPS does not establish legitimacy. Domain
separation does not remove this bias. Additionally, every legitimate source URL
has a `www` host and an empty/root path. Among phishing rows, only 41.35% have
a `www` host and 27.20% have a deeper path. These formatting shortcuts are severe:
the baseline check on `https://example.com/about` yielded a high phishing score,
with path length and the absence of a subdomain raising risk. This is a false alert
on an ordinary example URL, not evidence of a malicious page. Independent and
temporal evaluation with representative legitimate deep links and non-www hosts
is needed before deployment claims; high internal metrics are insufficient.

SHAP verification passed on 64 held-out URLs (32 per class): repeated explanations
were identical, all outputs finite, and the maximum reconstruction error against
the base model's raw output was 0.00000702. Domain partition disjointness was
independently rechecked. This verifies implementation consistency, not causal validity.

The model, report and metadata are in `artifacts/`. Provenance is recorded in
`artifacts/url_data_provenance.json`; explanation checks are recorded in
`artifacts/url_explanation_verification.json`.

Intended use: educational research and analyst decision support. A prediction must
not be treated as a guarantee of safety or an automatic enforcement decision.

Data: user-supplied local URL CSV, with sources and caveats documented in
[data/README.md](../data/README.md). No data or labels are invented for production.
The artifact records its dataset hash, split sizes, class metrics and library versions.
Keep source provenance and licensing records alongside each experiment.

Evaluation: compare logistic regression, random forest and XGBoost after independent
sigmoid calibration. Select by validation ROC-AUC then phishing recall. Report
held-out test precision, recall, F1, ROC-AUC and confusion matrix in the generated
`artifacts/url_report.md`. The current run's metrics are listed above.

Limitations: source bias, class imbalance, stale suffix lists, concept drift,
adversarial URL design, mislabeled popular domains and unseen domain patterns.
Lexical signals cannot inspect page contents or establish sender identity.
Calibration can degrade when deployment prevalence changes. Domain-disjoint
splits reduce domain leakage but do not eliminate source, campaign or temporal bias.
Multilingual domains are supported syntactically; fairness is not established.

Explainability: TreeExplainer handles forest/XGBoost models; LinearExplainer handles
scaled logistic regression. Both explain the fitted base model for the phishing
class. Calibration direction orients signs; nonlinear calibration is not part of
the additive SHAP decomposition. Contributions are not risk-score points, causal
effects, or evidence that a URL is safe. Correlated features and reference choices
affect attribution, and top-eight summaries omit part of the full decomposition.

Validation: synthetic tests verify deterministic contributions, reconstruction of
base-model outputs, class orientation, calibration direction, and artifact reloads
for all three estimator types. They do not establish real-data explanation quality.

Safety: Stages 1–3 perform no DNS, WHOIS, HTTP requests or redirect following.
Avoid publishing private URLs containing tokens. Only load trusted artifacts.

## Email model — Stage 3

Sources: CEAS_08, Enron, Ling, Nazario, Nigerian Fraud and SpamAssassin CSVs supplied
locally. See [Kaggle corpus](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
and [original curated sources](https://zenodo.org/records/8339691). Exact local source
hashes are recorded in `data/emails.provenance.json` and copied into artifacts.
The actual download date/location and complete source licensing chain were not
provided; check the source terms before redistribution. The unused aggregate was
removed; all six original source files remain available for reproduction.

Labels: `0` legitimate, `1` spam/fraud/phishing proxy. Nazario and Nigerian Fraud
are entirely positive; the other corpora contain spam/legitimate mixtures. Positive
metrics must not be presented as performance on exclusively confirmed phishing.

Preparation retained subject/body only and excluded sender/recipient/date/URL flags.
Of 82,486 rows, 21 exceeded the 100,000-character limit and 2 had no usable text.
Removing 5,637 normalized duplicate rows left 76,826 documents: 39,112 negative,
37,714 positive. There were no normalized-label conflicts in the supplied files.

Model: 30,000-term unigram/bigram TF-IDF with sublinear term frequency and L2
normalization, followed by CPU logistic regression. Vocabulary/IDF and coefficients
were fitted on 34,571 documents. Sigmoid calibration used 11,524 separate documents;
validation used 15,365; held-out testing used 15,366. No final refit was performed.

| Metric | Validation | Held-out test |
| --- | ---: | ---: |
| Precision | 98.1072% | 98.0418% |
| Positive recall | 98.2633% | 98.2368% |
| F1 | 98.1852% | 98.1392% |
| ROC-AUC | 0.998173 | 0.998472 |

At threshold 0.5, test confusion matrix `[[7675, 148], [133, 7410]]` gives 148
false positive alerts and 133 missed positive examples. Validation matrix is
`[[7679, 143], [131, 7412]]`. Rows are true classes, columns predicted classes.

Explanation: exact coefficient-times-TF-IDF terms in raw log-odds space, with
separate calibration-oriented display contributions. Highlights refer to cleaned
text; they are not safe HTML, causal claims or additive calibrated-probability changes.

Limitations: historic/English-heavy corpora, source-label correlation, spam/phishing
label ambiguity, remaining similar campaigns, random document rather than source
or temporal splits, loss of HTML attributes and URL identity, vocabulary blind spots,
and prevalence shift. Attachments and sender authentication are not evaluated.
High internal AUC does not establish current phishing-only or multilingual quality.

Artifacts: `email.joblib`, `email.metadata.json`, `email_report.md`, provenance and
`stage3_verification.json` under `artifacts/`. URL artifacts remain separate.

## Online URL model — experimental Stage 4 pilot

Acquisition sampled 100 unique registered domains from the prepared Kaggle URLs,
interleaving deterministically shuffled classes (seed 42). 7682 URL rows were
unsuitable for online lookup. Only public registered domains were queried; no URL
was opened. Outcomes: 30 complete WHOIS/DNS snapshots, 69 failures and one rejected
nonpublic DNS address. Failures are cached, not fabricated or imputed as safe zeros.

Features: the original 23 lexical signals plus domain age, registrar presence,
A/MX/NS counts and minimum/mean RRset TTL. IPv6 answers are screened for safety
but their count is not a model feature. Collection and file-only training are
separate. Snapshot SHA-256 and source SHA-256 are recorded in model metadata.
Stored WHOIS data contains no contact text. Runtime cache TTL is 24 hours.

The 30 complete domains match just **46 labeled URLs**: fitting 17 rows/13 domains,
calibration 14 rows/5 domains, validation 8 rows/6 domains and testing 7 rows/6 domains.
Groups use the queried ICANN registrable domain rather than private suffix tenants.
All four partitions are disjoint by that domain. Calibration contains only two
positive examples, which is inadequate support for trustworthy probabilities.

Random forest won validation ROC-AUC (1.0), ahead of XGBoost (0.9) and logistic
regression (0.8). Validation and test precision, recall and F1 are **all zero at
threshold 0.5**. Test confusion matrix is `[[6, 0], [1, 0]]`: it predicts all seven
URLs negative and misses the only positive. Test ROC-AUC 1.0 is based on ranking
one positive and is not meaningful evidence of detection quality. These metrics
must not be compared to the much larger offline test set as evidence of benefit.

The registered version is `0.1.0-online-pilot`, metadata marks it experimental,
and it is intended to exercise collection/training/inference integration only.
It is not suitable for operational decisions. A much larger representative
collection, adequate per-class/domain support, calibration assessment and temporal/
independent-source evaluation are required before considering deployment.

Additional limitations: expired domains, DNS aliases, hidden/unsupported WHOIS dates
and differing registry formats cause substantial selection bias. Today's domain
owner and registration data may not describe a historical phishing URL. Metadata
about a shared hosting apex may say little about a malicious tenant. Registrars,
MX records, TTL or domain age alone cannot establish safety. Strict validation can
reject legitimate local/IP-address use cases; those remain analyzable offline.

When online lookup fails, times out, is rate-limited or cannot use its cache, the
predictor returns the existing offline model with `mode: offline_fallback` and a
reason. A missing online model also falls back. Unsafe online targets are rejected.
No URL content, HTTP redirects, user-domain CNAME targets or WHOIS referrals are
fetched. All API processes must share one SQLite cache for cross-process rate limits.

Evidence: `url_online.joblib`, `url_online.metadata.json`, `url_online_report.md`,
`stage4_live_prediction.json` and `stage4_verification.json` under `artifacts/`;
frozen acquisition data is `data/url_online_features.csv` with a collection report.
