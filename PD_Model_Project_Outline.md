# Probability of Default (PD) Model — Project Reference Outline

A stage-by-stage map for building a scorecard-style PD model end to end. Use it to
keep the pipeline in order and to make sure each stage produces a documented
*decision*, not just a chart. The decisions are what make this read as credit-risk
work rather than a generic ML tutorial.

> **North star:** every analysis block should end with a written Explanation →
> Observation → Decision (the **E/O/D** convention used in the notebook). The code
> proves you can execute; the decision trail proves you can think. That trail is the
> single most impressive thing you can show a hiring manager.

---

## Stage 0 — Scope & Target Definition

Set the rules of the game before touching a model. Easy to skip, but a wrong target
definition invalidates everything downstream.

- **Define "default"** — the event you're predicting (e.g. 90+ days past due within
  the performance window). State it explicitly.
- **Windows** — observation window (when features are measured) vs performance
  window (when default is observed). Avoid leakage between them.
- **Population** — who's in scope (product, portfolio, exclusions like staff/fraud).
- **Success criteria** — what "good" looks like (e.g. Gini target, calibration
  tolerance) so validation later has a bar to clear.

**Deliverable:** a short scope statement + target definition.

---

## Stage 1 + 2 — EDA & Feature Engineering (WOE / IV) *(combined)*

Originally two stages; in practice the driver scan runs straight through from
understanding the data to encoding it, so they're **combined into one notebook**
(`01_Default_Rate_Driver_Scan_EDA`). Structure below follows the notebook's seven
sections; each analysis block uses the **E/O/D** convention (Explanation before
looking, Observation off the output, Decision that points forward to the binning in
Section 4).

- **1. First Look** — shape (307k × 122), class imbalance (92% repaid / 8% default —
  high for a mortgage book; *no resampling*, since this is a screening exercise, not
  model training, and WOE/IV are computed on the full distribution), description-lookup
  reference for feature relevance.
- **2. Data Quality & Missingness** — split the 122 columns into **statistical subsets**
  (continuous / discrete / nominal / ordinal / binary / id) and **thematic subsets**
  (`cols_by_theme`) so a wide table stays tractable. Flag disguised sentinels
  (`DAYS_EMPLOYED` = 365243, `CODE_GENDER` = XNA); rank missingness; diagnose
  **block-wise missingness** (`building_stats`, `social_circle`, `bureau_enquiries`) and
  classify the mechanism (MAR vs MNAR — e.g. `OWN_CAR_AGE` missing ↔ no car = MAR).
  **Cardinality treatment** for high-cardinality categoricals: `ORGANIZATION_TYPE` band
  mapping by default rate, `OCCUPATION_TYPE` sparse-level pooling with NaN as its own
  category.
- **3. Feature Analysis** — per-family **univariate + bivariate** scan, highest-value
  families first: external scores (`EXT_SOURCE_1/2/3`), loan financials, age &
  employment, demographics, other numeric families. Bivariate toolkit: default-rate-by-bin,
  AUC / Gini, KS / ECDF. Each family closes with a Decision stating the intended
  transform/binning for Section 4.
- **4. WOE / IV Computation** — the feature-engineering heart. **Train-only split**
  (stratified on TARGET), a hand-rolled WOE/IV helper (`WOE = ln(%good / %bad)`,
  `IV = Σ (%good − %bad) × WOE`), the **cross-feature IV ranking** (this is where the old
  "target-relationship scan" now lives), top-10 drivers, WOE encoding (fit on train,
  `transform` val/test), and an `optbinning` cross-check. IV rule of thumb: <0.02 useless,
  0.1–0.3 medium, 0.3–0.5 strong, >0.5 suspiciously strong → leakage-check.
- **5. Findings Summary** — consolidated headline drivers + data-quality conclusions,
  written to stand alone for an auditor.
- **6. Next Steps** — concrete actions carried into feature selection / modelling.
- **7. Assumptions, Limitations & Data Lineage** — the reproducibility and
  model-validation record.

**Key decisions:** which columns are usable; how sentinels are handled; which missingness
is informative (keep a flag) vs explainable (impute); bin boundaries and monotonicity;
missing-as-its-own-bin where informative; which engineered features to create.

**Leakage guards:** exclude id columns (`SK_ID_CURR`) before ranking — a unique key
yields spurious near-infinite IV; fit bins and WOE on the **train split only** and apply
to val/test, never refit on val.

**Gotchas:** keep target-relationship work *out* of univariate (that's bivariate);
peeking at TARGET in Section 3 is a diagnostic — the quantified cross-feature ranking is
the real work and lives in Section 4.

---

## Stage 3 — Feature Selection

Cut down to a clean, non-redundant, defensible feature set.

- **IV filter** — drop weak predictors below an IV threshold.
- **Multicollinearity** — correlation matrix / VIF; drop redundant variables (this is
  where the building_stats triplet-collapse decision pays off).
- **Business sense** — keep variables that are explainable to a credit committee;
  be wary of proxies for protected attributes.

**Deliverable:** final feature list with the rationale for each inclusion/exclusion.

---

## Stage 4 — Model Development

Train the model. For PD scorecards this is usually **logistic regression** —
interpretable, regulator-friendly, and the industry default.

- Train/test split (and set aside an out-of-time sample if you have dates).
- Handle class imbalance (weighting, or acknowledge and calibrate later).
- Fit the model; inspect coefficients (signs should make business sense).
- Baseline results — first-pass discrimination metrics.

**Key decisions:** model choice (and why LR over a black box for PD), imbalance
strategy, how you split.

---

## Stage 5 — Validation & Diagnostics ⭐

The stage most portfolios skip — and the one that most signals credit-risk maturity.
Discrimination alone is not enough.

- **Discrimination** — Gini / AUC, **KS statistic** (max separation between good/bad
  cumulative distributions).
- **Calibration** — predicted PD vs actual default rate by band. A model can rank
  well but be poorly calibrated.
- **Rank ordering** — default rate should increase monotonically across score bands.
- **Stability** — **PSI** (Population Stability Index) between dev and out-of-time
  samples; flags whether the population has drifted.
- **Out-of-time / out-of-sample** — does it hold up on data it never saw?

**Deliverable:** a validation report. This is the section that earns trust.

---

## Stage 6 — Scorecard Scaling *(optional, high-signal)*

Convert model output into a points-based scorecard.

- Choose scaling: base score, base odds, **PDO** (points to double the odds).
- Translate coefficients × WOE into points per attribute.
- Produce the final scorecard table (attribute → points).

Even a lightweight version signals you understand how PD models are actually deployed.

---

## Stage 7 — Business Recommendations & Monitoring

Turn the model into decisions a lender can act on.

- **Cutoff selection** — approval rate vs bad rate trade-off; expected loss at each
  cutoff.
- **Strategy** — pricing, decline, or refer bands; expected impact on the book.
- **Monitoring plan** — PSI tracking, recalibration triggers, review cadence.
- **Limitations** — what the model can't do, known risks, next iterations.

**Deliverable:** a decisions-first summary aimed at a non-technical stakeholder.

---

## The one document that matters most

When the project is done, write a **decisions-first README** — the 8–10 calls you
made and why, across the whole pipeline:

- target definition, sentinel/missingness handling, imbalance strategy, binning
  choices, feature selection rationale, model choice, validation results, cutoff logic.

That's the document a hiring manager actually reads. The notebooks prove execution;
this proves judgment.

---

## Portfolio context

This PD model is the flagship. Planned companion pieces, and what each is meant to
demonstrate:

- **PD model (this)** — end-to-end analytical judgment on messy real data.
- **Power BI loan-book dashboard** — interactive BI / data visualisation / storytelling.
- **SQL project** — data extraction, joins, aggregation, window functions.
- **Excel project** — modelling / analysis in the tool most finance teams live in.

Together they cover the four things a credit-analytics hire is checked on: judgment,
BI, SQL, and spreadsheet fluency.
