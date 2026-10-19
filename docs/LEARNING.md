# Learning PhishLens — Stages 1–4

## config.py
One place defines the seed, paths and input bound. Future prediction risk cutoffs
are separate from the 0.5 threshold used to compare classifiers in reports.

## features/url_offline.py
The ordered description mapping is the feature schema: training and prediction
must use identical column order. Each description explains its signal. Lengths,
delimiters, digits and entropy measure complexity or obfuscation; IPs, ports and
shorteners affect destination transparency. Keywords describe account actions.
HTTPS is encryption, not proof of legitimacy. Punycode supports valid multilingual
domains but can also encode lookalikes. TLD indicators are weak and can encode bias.
No individual feature proves phishing. Learned models combine them.

Shannon entropy sums `-p * log2(p)` for each character frequency. Repeated characters
have low entropy; diverse characters have higher entropy. It is not a randomness test.
The public suffix list distinguishes `co.uk` from an actual registered domain.
Private suffixes also separate tenants such as independent hosted sites.
Bundled suffix data avoids network access; its age is a documented limitation.

## models/data.py
Duplicate rows can inflate results. Conflicting labels indicate a data quality
problem and are rejected. Related hosts share a registered-domain group, preventing
memorized domain structure from appearing in both fit and test data. GroupShuffleSplit
splits groups rather than individual URLs: row percentages may differ from requested
group percentages. It does not stratify; every partition must contain both labels.

## models/train_url.py
Logistic regression learns a weighted sum of standardized features. Scaling is fitted
only on training data. Random forests average decision trees; XGBoost builds small
trees that correct earlier errors. All use CPU and fixed seeds.

Four independent partitions separate responsibilities: roughly 45% fit, 15%
calibration, 20% validation and 20% final test by domain groups. Calibration fits
a sigmoid to an already-fitted model's outputs: it aims to make probabilities
correspond to observed frequencies. FrozenEstimator prevents refitting the base
model. Validation chooses among calibrated candidates by ROC-AUC, then recall.
The winner is not refitted after selection, preserving calibration and isolation.

## models/evaluate.py
Precision asks how many phishing predictions were correct; recall asks how many
phishing examples were caught. F1 balances these. ROC-AUC measures ranking across
thresholds. The confusion matrix uses true classes as rows, predictions as columns:
`[[true negative, false positive], [false negative, true positive]]`.
Good AUC does not demonstrate good calibration or production generalization.

## models/registry.py
The artifact stores model and metadata together; metadata includes schema, metrics,
source hash, timestamp and library versions. Atomic replacement protects the main
artifact from partial writes. Joblib uses pickle semantics and can execute code:
trust the file's origin. The separate JSON is for inspection, not model loading.

## Notebook and tests
The notebook visualizes local data and saved metrics; production training lives
in src. Tests exercise parsing, invalid data, domain isolation, deterministic splits,
and an actual training/save/load cycle using synthetic data. Synthetic quality
metrics cannot support claims about real phishing detection.

## explain/url_shap.py

SHAP assigns each feature a share of the difference between a reference output
and the model's output for this URL. The idea comes from sharing a team's outcome
among its players: a feature's contribution averages how it changes the outcome
when considered alongside other features. Contributions explain model behavior,
not the real-world cause of phishing.

For random forest and XGBoost, TreeExplainer uses the fitted tree paths and their
training sample counts as its reference distribution. This requires no stored
URLs or test rows. For logistic regression, we apply the fitted StandardScaler,
then LinearExplainer uses the zero vector as its background. Zero in scaled space
is the training mean, so each contribution is exactly `coefficient * scaled_value`.
The returned feature value remains in its original units, so a URL length still
means characters rather than standard deviations. This linear explanation uses
the independent-feature interpretation; correlated URL signals can share influence
differently under other reference assumptions.

We explain class `1` (phishing) consistently. Positive contributions increase the
phishing-directed output relative to its baseline; negative contributions decrease
it. A legitimate prediction can still have risk-raising features. Never reverse
signs merely because the predicted label is legitimate.

Calibration wraps the fitted base model. Stage 1 stores one FrozenEstimator with
sigmoid calibration: `probability = sigmoid(-(a * score + b))`. We unwrap that
estimator for SHAP, keeping its learned trees, coefficients and scaler unchanged.
If `a < 0`, calibration preserves the score direction. If `a > 0`, calibration
reverses it, so contributions and baseline are multiplied by `-1`. If `a == 0`,
the calibration is constant, so risk influences are zero. The orientation is
reported explicitly. Cross-validation ensembles and isotonic calibration are
rejected because they need different explanation handling.

The nonlinear sigmoid means base-model contributions cannot be called additive
calibrated probability changes. Logistic regression and XGBoost explanations use
log-odds; forest explanations use uncalibrated probability. Magnitudes are not
directly comparable between these model types. The response states its units and
includes this limitation. All contributions plus `base_value` reconstruct
`explained_value`; a top-eight list is only a summary of that full decomposition.

Ranking uses absolute magnitude and schema-order tie breaking. Exact SHAP methods
and a fixed training-derived reference make repeated explanations deterministic.
Tests check reconstruction against actual model outputs, exact linear coefficients,
both predicted classes, reversed/constant calibration, and trusted artifact reloads.

Reference: [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
and [LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html).

## First real-data experiment: PhiUSIIL

The source labels are `1` legitimate and `0` phishing, so conversion flips them
before training. Label meaning must be verified from the source, not guessed from
the numbers. We kept only raw URLs and labels and recomputed all features, avoiding
webpage content and any supplied dataset-wide statistical features.

After removing 425 duplicate URLs, 235,370 rows remained. XGBoost won by validation
ROC-AUC, although random forest had marginally higher recall. Following the stated
selection rule prevents choosing a model based on whichever result looks best later.

Test ROC-AUC was 0.9979, but the dataset has a shortcut: all legitimate examples
use HTTPS, while about half the phishing examples use HTTP. A domain-disjoint split
can retain that pattern, producing strong internal scores that may not transfer.
All legitimate examples are also `www` homepages, making non-www hosts and deeper
paths potential shortcuts. The model falsely flags the ordinary example URL
`https://example.com/about`; SHAP shows its path and lack of subdomain raising risk.
This illustrates why explainability and representative evaluation matter even
when held-out metrics are excellent. The model card records these limitations.
Explanations were checked on 64 test URLs
for deterministic results and reconstruction of the base model's output.

## features/email_text.py

Cleaning is part of the saved pipeline so fitting and inference transform text
identically. Python's HTMLParser extracts visible text without opening links or
running HTML. Script/style contents are ignored. URLs and email addresses become
`urltoken` and `emailtoken`: the model can learn their presence without memorizing
individual addresses. Text is lowercased; inputs over 100,000 characters are rejected.
Removing markup does not make a message safe to render as HTML; future UI code
must escape text when drawing highlights.

TF-IDF combines term frequency with inverse document frequency, downweighting
ubiquitous words and emphasizing distinctive ones. Log-scaled counts reduce the
effect of repeated words. Unigrams are individual tokens; bigrams capture pairs
such as `verify password`. L2 normalization reduces document-length effects. Only
training documents define vocabulary and IDF; there are at most 30,000 terms.

## models/prepare_email.py and models/train_email.py

Preparation combines subject and body and retains the source name for auditing.
Sender, recipient, dates and precomputed URL flags do not enter the model. Token
sequence hashes identify duplicates after HTML/address normalization, including
punctuation differences. All copies with conflicting labels are removed during
preparation and counted; the generic training loader rejects any remaining conflict.
It also rejects invalid or oversized inputs rather than silently altering them.

The pipeline is fitted on 45% of deduplicated documents, calibrated on 15%,
validated on 20% and tested on 20%, with small rounding differences. Stratification
preserves class balance. Duplicate keys never span partitions, but this does not
group similar campaigns or hold out entire sources. Source bias and near-duplicates
can still inflate results. The positive label in these corpora includes ordinary
spam and fraud, so the measured task is broader than phishing-only detection.

## explain/email_words.py

Logistic regression's raw decision score is its intercept plus the sum of
`coefficient * TF-IDF` for each present term. Those products are exact contributions,
computed sparsely without building a dense 30,000-column document. The explainer
returns `raw_contribution` for this decomposition and `contribution` for display
after orienting by calibration direction. Nonlinear calibration means neither is
an additive change in calibrated probability. The intercept and `explained_value`
remain in raw log-odds space.

The top terms normally rank by magnitude, with a slot reserved for the opposite
direction when present. This shows mitigating evidence instead of only alarming
terms. `highlights` reference offsets in the returned `cleaned_text`; tokenizer
positions also handle bigrams across punctuation or ignored one-character words.
A term's influence is counted once even if highlighted repeatedly. Overlapping
unigrams and bigrams are separate model features, not extra highlight contributions.
Text containing only unseen words gets no influential terms and the intercept-only
decision score. Such an output is not evidence that the text is safe.

## Registry and verification through Stage 3

The registry now names artifacts `url_offline`, `url_online` or `email`, keeping
email training from overwriting the URL model. The default remains `url_offline`
for compatibility. Email artifacts validate their vocabulary against metadata.
Tests cover cleaning, length limits, duplicates/conflicts, partition separation,
unseen test vocabulary, exact contributions, highlights, calibration orientation
and saving/loading both model types together.

## Resolving the URL homepage shortcut

The initial PhiUSIIL experiment showed why high held-out scores can coexist with
poor practical behavior. Its legitimate class consisted of HTTPS `www` homepages.
We replaced that training input with downloaded, labeled Kaggle URLs containing
real legitimate deep links and non-www hosts. We did not simply add one example
URL to training, whitelist domains or lower the decision threshold.

`models/prepare_url.py` validates source columns and labels, excludes malformed
URLs and conflicting labels, removes duplicates, and records hashes and counts.
Five entire registered domains are reserved for a fixed sanity set, keeping them
away from fitting, calibration and validation. Model selection still uses only
validation ROC-AUC, with recall as tiebreaker. The old artifact and source hashes remain
available so we can compare behavior without rewriting the experiment history.
Superseded input CSVs were removed during data cleanup.

`models/audit_url.py` measures held-out metrics and false alerts on legitimate
deep links, and checks eight reserved benign URL patterns. These probes diagnose
the original error but do not prove production readiness. The old model's fit
may overlap domains in the new dataset, so cohort comparisons are descriptive;
the new model's own test partition remains domain-disjoint from its fit data.

The selected replacement is random forest (test ROC-AUC 0.8973). The original
example's probability fell from 99.98% to 28.84%, and all eight reserved probes
are below the suspicious cutoff. The test set has 67,110 legitimate deep links;
the new model flags 2.25% at threshold 0.5. Its overall positive recall is only
56.10%, so removing one shortcut has not established sufficient detection quality.
The model card records both this improvement and the remaining misses.

## features/url_online.py — metadata and safety

Domain age comes from the earliest parsed WHOIS creation date. Young domains can
be disposable, but legitimate sites can be young and old domains can be compromised.
Registrar presence measures available metadata, not registrar trustworthiness.
A/MX/NS counts describe IPv4 hosting, mail and name-server configuration. No MX
record can be normal for a website. Minimum and mean TTL summarize RRset cache
durations; short TTLs are common for legitimate CDNs as well as changing infrastructure.

We extract the ICANN registrable domain before networking. `login.customer.co.uk`
becomes `customer.co.uk`; `tenant.github.io` becomes `github.io` for WHOIS/DNS.
Offline lexical grouping includes private suffix tenants, but online training
groups by the queried apex so shared online metadata cannot leak across partitions.
All suffix extraction uses bundled data, with downloads disabled.

DNS asks only that apex name. We inspect both A and AAAA addresses, rejecting
private, loopback, link-local, reserved and multicast values before WHOIS. Checking
`is_global` alone is insufficient because it accepts some multicast addresses.
DNS aliases are not followed, and neither MX nor NS host targets are contacted.
The URL's web server, path and subdomain are never resolved or fetched. Consequently
this is apex metadata analysis, not validation of every subdomain's routing.

WHOIS uses one registry port-43 query for the apex, with a bounded response size.
Registry server addresses are validated and a numeric public IP is used for the
connection, avoiding a second name resolution. Registry infrastructure DNS is
necessary to reach WHOIS; it is separate from URL/subdomain questions. WHOIS
referrals are ignored. `python-whois` parses the response locally; raw contact
records are discarded instead of being cached or logged.

## Cache, timeouts and backoff

SQLite stores domain keys, feature JSON, status, timestamps and retry counts.
Successful snapshots live for 24 hours. Failure entries retain null/missing values
and a reason, preventing repeated failed requests. Backoff grows as `30 * 2^n`
seconds, capped at one hour. Zero MX records are known absence; a null value is an
unknown lookup result. They must not be confused.

Each caller opens its own short-lived database connection. A shared database
lease serializes misses across processes, while a process semaphore guards threads
and separate lookup instances. All application workers must share the same cache.
Cache hits bypass the lookup slot; misses during a busy/rate-limited period fall
back instead of creating an unbounded queue. A crashed process's lease expires.

Socket timeouts are not enough to stop blocked library/name-resolution calls.
Network work runs in a spawned process. The parent waits at most eight seconds for
the result, then terminates/kills and joins the worker with bounded cleanup. This
avoids accumulating timed-out threads whose underlying operations keep running.
All of these behaviors are tested using mocked transports/workers, not live calls.

## models/precompute_online.py and models/train_url_online.py

Acquisition interleaves deterministically shuffled classes and samples domains,
not individual URL rows. It is deliberately bounded and slow. Completed data is
stored in SQLite and atomically exported after each domain, allowing interruption
and resume. Refreshing still honors cache TTL/backoff. No API keys or paid services
are required. Training is a separate command that reads only CSV files and refuses
to fill missing features by networking.

The online trainer joins complete domain snapshots to source URLs, splits by
queried domain, and reuses `train_url.fit_models` for the three-estimator comparison
and independent sigmoid calibration. The online artifact contains 30 features:
23 lexical plus seven domain features. Incomplete online metadata causes inference
to use the original offline model; we do not impute unknown metadata as safe zeros.
Successful-lookup selection bias is substantial: expired phishing domains often
cannot supply today's complete features, and ownership may have changed.

The initial real collection produced 30 complete snapshots from 100 domains,
matching just 46 labeled URLs. The online test set has seven rows and one positive.
The pilot's AUC is 1.0 while recall/F1 are zero because it ranks that one positive
above negatives but predicts all probabilities below 0.5. This demonstrates why
AUC alone is insufficient, and why small test sets and calibration require scrutiny.
The artifact is explicitly versioned `online-pilot`; more representative collection
and evaluation are needed before making detection-quality claims.

## predict.py and scoring.py

`Predictor` loads URL/email/optional online artifacts once. SHAP explainers are
initialized lazily and reused. It checks schemas and chooses the model matching
available features. Online failures return `offline_fallback`, the offline model's
version and probability, and a short reason. Unsafe online inputs are rejected.
The same collected features feed both prediction and explanation, so explaining
a result never repeats a network lookup. Email behavior remains unchanged.

Risk score is a rounded display of `100 * calibrated_probability`. Labels use
unrounded probability thresholds of 0.35 and 0.70, avoiding boundary changes caused
by display rounding. A probability 0.3499 displays 35 but is still legitimate.
SHAP contributions explain base-model output, not these rounded risk points.

## schemas.py and api/main.py (Stage 5)

Pydantic contracts validate field types, allowed modes and input lengths before
inference. Reusing URL parsing and email cleaning keeps training/CLI/API inputs
consistent. Response contracts constrain calibrated probability and risk labels.
Email highlights need the cleaned text because removing HTML changes offsets.

FastAPI's lifespan loads artifacts once per process. Worker threads keep blocking
model and DNS/WHOIS operations off the event loop; an asynchronous inference lock
serializes work and protects lazy explainers without blocking health requests.
Availability and model quality are different: health exposes the online pilot's
experimental metadata even when it is loaded successfully.

The ASGI request guard counts actual received bytes before JSON parsing, so a
missing or inaccurate Content-Length cannot bypass the body limit. Its bounded
in-memory fixed-window counter limits prediction traffic by direct client IP.
Counters are local to a process and restart with it. Request IDs connect responses
to JSON logs without storing email bodies or suspicious URL strings. Validation
errors exclude Pydantic's default input-value echo, and internal errors expose
only a generic message. FastAPI TestClient context managers exercise the real
startup/shutdown lifecycle without needing a listening server.

## app/ (Stage 6)

The Streamlit app is an HTTP client, not a second inference implementation.
`client.py` submits bounded form values through HTTPX and validates responses
against the shared schemas. Errors are translated to useful messages without
echoing raw server errors or submitted email text. No global result cache shares
private inputs between users; each tab retains only its own session result.

`visuals.py` turns real model outputs into a segmented polar risk meter and a
signed contribution chart. Red bars raise the phishing estimate; cyan bars lower
it. Risk labels come from the API rather than rounded display-score thresholds.
For email highlights, overlapping term intervals are split into disjoint pieces
and the strongest contribution sets the color. Escaping every text segment before
adding trusted markup prevents an email's HTML from becoming dashboard code.

The theme defines the navy/cyan palette; the small stylesheet handles compact
panels and narrow-screen stacking. Empty states show no fabricated results.
Streamlit AppTest exercises forms and model availability; HTTP boundary tests
cover payloads, timeouts, rate limits and malformed response schemas.

## Final polish (Stage 7)

Public functions and methods carry parameter/return annotations and concise
docstrings explaining their purpose. An annotation describes the expected
contract; Pydantic and explicit validation enforce input rules at runtime. Ruff
checks style and imports, while regression tests check behavior. Running pytest
with `-W error` makes dependency deprecations visible instead of hiding them.

Starlette and AnyIO are existing FastAPI dependencies. Their compatibility bounds
keep TestClient on the project's HTTPX stack and avoid deprecated AnyIO aliases;
no warning filters or runtime monkey patches are used. Future upgrades should
rerun strict tests together rather than upgrade those components independently.

`make app` starts `app/streamlit_app.py`; `make dashboard` is a compatibility alias.
The theme remains in root `.streamlit/config.toml`, where Streamlit discovers it
when launched from the repository root. A clean install must work without local
datasets or saved models: tests create their own synthetic fixtures, and missing
artifacts produce explicit API availability errors. Synthetic training checks
exercise reproducibility and command wiring, not detection quality on real data.
