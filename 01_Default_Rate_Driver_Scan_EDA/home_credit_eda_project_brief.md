# Project 1 — Default-Rate Driver Scan (EDA)

- **Module:** Credit Risk ML — Phase 1 (Foundations & first deliverable)
- **Project type:** Exploratory Data Analysis + feature screening
- **Dataset:** Home Credit Default Risk — `application_train.csv` (Kaggle). Synthetic stand-in acceptable if the real file isn't to hand.
- **Estimated effort:** ~2–3 evenings (10–14 hours, including the presentation)
- **Prerequisites:** Types of data & missing values; hypothesis testing & confidence intervals; Pandas basics
- **Weighting toward portfolio:** High — this is the project that turns open-ended EDA into a screening deliverable you can show an interviewer.

---

## 1. Overview & context

In a real credit-risk team, nobody does "EDA" for its own sake. Before anyone builds a scorecard, an analyst screens the available features to answer one question: **which variables actually separate defaulters from non-defaulters, and by how much?** That screen decides what goes into the model and what gets dropped. It is the first thing you'd do on a new book.

This project gives your Home Credit EDA that exact spine. Instead of an aimless tour of histograms, you will produce a **ranked table of predictive power** across the dataset's features, backed by the statistical checks you've been drilling, and present it to a (hypothetical) non-technical audience. The deliverable is the kind of artifact — and the kind of communication — a portfolio analyst produces in their first week.

Two threads run through this project on top of the analysis itself:

- **Audit quality.** Your notebook is not just working code; it's an audit trail. In credit risk, a model validator or auditor has to be able to follow *why* you made each decision — why you binned a feature that way, why you dropped a column, why you treated an anomaly as you did — from the notebook alone, without asking you. Clean markdown narrative throughout is the skill being assessed, not an afterthought.
- **Stakeholder communication.** The people who act on this screen — lending managers, a risk committee — are often not technical. Your findings are delivered as a **Quarto presentation** pitched at that audience: the business answer first, jargon explained, the technical depth left in the notebook where it belongs.

The industry-standard measure of "predictive power" here is **Information Value (IV)**, built on **Weight of Evidence (WOE)**. You'll compute both from scratch so you understand what they mean rather than calling a black-box library.

## 2. Learning outcomes

On completing this project you should be able to:

1. Assess the quality of a real credit dataset — missingness, anomalies, class imbalance — and decide how to handle each.
2. Explore the relationship between individual features and a binary target using appropriate visualisations and statistics.
3. Compute Weight of Evidence and Information Value correctly, and interpret IV against standard thresholds.
4. Rank features by predictive power and defend the ranking, distinguishing genuine signal from artefacts (leakage, tiny bins, anomalies).
5. Document an analysis to audit standard — a reviewer can follow every decision from the notebook alone.
6. Communicate findings to a non-technical stakeholder audience in a clean Quarto presentation.

## 3. The task

Produce a Jupyter notebook that performs a structured EDA of the Home Credit application data and screens every usable feature for its ability to predict `TARGET` (1 = default, 0 = repaid). Two outputs come from it: a **ranked Information Value table** inside the notebook, and a **Quarto presentation** that communicates the top ~10 drivers to a non-technical audience.

Scope boundary: **this is EDA and screening only — no predictive model is built in this project.** You are deciding what *would* go into a model, not building one. (That's Project 3.)

## 4. Recommended notebook structure

Mirror the Predict → Task → Decision discipline from your 1.4 notebook: state what you expect, do the work, then write a one-or-two sentence decision. Every section should open with a short markdown cell saying what you're about to do and why, and close with what you concluded — that running narrative *is* the audit trail. Suggested sections:

1. **Setup & reproducibility** — imports, `RANDOM_STATE`, display options. Notebook must run top-to-bottom without errors.
2. **Data load & first look** — shape, dtypes, `head()`, target rate. Confirm the class imbalance up front.
3. **Data quality & missingness** — per-column missing %, flag the `DAYS_EMPLOYED == 365243` anomaly, note obviously unusable columns. Document each treatment decision.
4. **Univariate EDA** — distributions of key numeric and categorical features; note skew (income, credit) and where transforms or binning will be needed.
5. **Target-relationship scan** — for each feature, default rate by bin/category, with a visual for the most important ones. This is where signal starts to show.
6. **WOE / IV computation** — bin each feature, compute WOE per bin and IV per feature, assemble the ranked table.
7. **Findings summary** — the ranked IV table + written interpretation of the top drivers (this is the raw material your presentation is built from — replace any placeholder text with real conclusions).
8. **Next steps** — what you'd carry into scorecard development, what you'd drop, what needs more work.

## 5. Detailed expectations

**Data quality.** Report missingness per column. Handle the `DAYS_EMPLOYED` sentinel (`365243`) explicitly rather than letting it distort everything. State your treatment of missing values (and why) — you don't have to impute for a screen, but you must not silently ignore them.

**Class imbalance.** Establish and state the default rate (~8% in this data). Every "default rate by bin" number below should be read against that base rate.

**Coverage.** Screen a meaningful set of features — at minimum all the `EXT_SOURCE_*` scores, the `AMT_*` financials, age/employment, and the main categoricals (education, family status, contract type, gender, car ownership). Aim for breadth; a screen that looks at three columns isn't a screen.

**WOE/IV correctness.** Use the standard convention and state it explicitly:

- WOE for bin *i* = ln( (Good_i / Good_total) / (Bad_i / Bad_total) ), where **Good = non-default (TARGET=0)** and **Bad = default (TARGET=1)**.
- IV = Σ_i (Good%_i − Bad%_i) × WOE_i, summed over a feature's bins.
- Guard against empty bins (a zero count makes WOE blow up — add a small epsilon or merge the bin, and say which).
- For numeric features, bin sensibly (quantile bins are a reasonable default); for categoricals, each level is a bin (merge sparse ones).

**IV interpretation.** Read every IV against these standard bands, and comment where a feature falls:

| IV | Interpretation |
|---|---|
| < 0.02 | Not predictive — drop |
| 0.02 – 0.1 | Weak |
| 0.1 – 0.3 | Medium |
| 0.3 – 0.5 | Strong |
| > 0.5 | Suspiciously strong — check for leakage or an artefact before celebrating |

**Statistical backing.** Use at least one of the tests from your 1.4 notebook (t-test / Mann-Whitney for a numeric driver, chi-square for a categorical one) to corroborate that your top drivers are separating the classes, not just showing noise. Tie the conclusion back to the base rate.

**Interpretation quality.** For your top drivers, say *why* the relationship makes sense (e.g. external scores are pre-built risk measures, so high IV is expected; a suspiciously high IV on an ID-like field is a red flag, not a win).

**Audit-quality narrative.** Every non-trivial decision must be explained in a markdown cell at the point it's made — binning choices, anomaly handling, dropped columns, why a feature's high IV is or isn't trustworthy. A reader who doesn't know you should be able to reconstruct your reasoning end to end. No orphaned code cells doing something unexplained.

## 6. Deliverables

- [ ] **Notebook** (`.ipynb`) — runs top-to-bottom, reproducible via a fixed seed, no leftover errors or stale state, with clean markdown narrative throughout (the audit trail).
- [ ] **Ranked IV table** — every screened feature with its IV, sorted descending, band labelled. Displayed in the notebook (exporting to CSV is a nice-to-have).
- [ ] **Quarto presentation** (`.qmd` → rendered `.html`, reveal.js) — a stakeholder deck communicating the findings (see Section 7).
- [ ] **Clean code** — no unused cells, sensible variable names, the `.sum()`-not-`.count()` kind of bug caught.

## 7. The stakeholder presentation (Quarto)

Build a short reveal.js deck in Quarto (`format: revealjs`) aimed at a **non-technical audience** — imagine a lending manager or risk committee, not a data scientist. It should:

- **Lead with the answer.** Slide one after the title states the business takeaway ("These are the strongest predictors of default in this book"), not the methodology.
- **Explain, don't assume.** If you use "Information Value," give it a one-line plain-English gloss ("a measure of how strongly a factor separates good borrowers from bad"). No unexplained jargon.
- **Show, visually.** The ranked drivers as a clean bar chart, not a raw table dump. A default-rate-by-band chart for one or two headline drivers.
- **Be honest about caveats.** One slide on limitations — missingness, anything that looked too good to be true, what this screen does *not* tell you.
- **End on a recommendation.** What you'd take forward to the scorecard, in business language.
- **Length:** roughly 6–10 slides. Technical depth stays in the notebook; the deck is the communication layer.

This is also your first Quarto artifact — getting a `.qmd` to render to reveal.js cleanly is part of the exercise.

## 8. Rules & constraints

- **No modelling.** No logistic regression, no ML estimators. Screening only.
- **Reproducible.** Fixed `RANDOM_STATE`; the notebook must produce the same numbers on a fresh run.
- **State your conventions.** WOE direction, binning choices, missing-value treatment — all named explicitly.
- **No placeholder text in the final version.** Findings, Next Steps, and every slide must contain real content.

## 9. Marking rubric

Australian grading bands. Criteria weightings sum to 100.

| Criterion | Weight | High Distinction (85–100) | Credit–Distinction (65–84) | Pass (50–64) | Fail (<50) |
|---|---|---|---|---|---|
| **Data understanding & quality** | 15 | Missingness, the `DAYS_EMPLOYED` anomaly, and class imbalance all identified and handled with stated reasoning | Most quality issues caught and handled; minor gaps | Basic checks done; anomaly or imbalance missed or unhandled | Little/no data-quality assessment |
| **EDA depth & visualisation** | 15 | Broad, purposeful EDA; clear visuals that each answer a question; skew/anomalies surfaced | Solid coverage and readable plots; some purely decorative | Thin coverage or plots without interpretation | Minimal or uninterpreted EDA |
| **Statistical rigour (WOE/IV + testing)** | 22 | WOE/IV correct, convention stated, edge cases guarded; hypothesis tests corroborate drivers correctly | Mostly correct with a small slip (e.g. unguarded empty bin) | IV computed but with a conceptual error, or no statistical backing | WOE/IV wrong or absent |
| **Driver identification & interpretation** | 18 | Ranked table correct; top drivers interpreted with genuine credit reasoning; leakage/artefacts flagged | Ranking sound; interpretation present but shallow in places | Ranking produced but weakly or mechanically interpreted | No defensible ranking |
| **Audit quality & in-notebook narrative** | 10 | Every decision explained in clean markdown at the point it's made; a reviewer could follow the reasoning end to end with no gaps | Mostly well-documented; a few decisions left implicit | Sparse narrative; reader has to infer key choices | Code with little or no explanation |
| **Stakeholder communication (Quarto deck)** | 15 | Renders cleanly; leads with the answer; jargon explained; strong visuals; honest caveats; genuinely lands for a non-technical audience | Clear and mostly non-technical; a slide or two too dense | Deck exists but assumes technical knowledge or buries the takeaway | No deck, or unusable for the stated audience |
| **Code quality & reproducibility** | 5 | Clean, runs top-to-bottom, reproducible, no bugs | Runs with minor untidiness | Runs but messy or fragile | Doesn't run cleanly |

**Grade bands:** HD ≥ 85 · D 75–84 · Cr 65–74 · P 50–64 · Fail < 50.

## 10. Common pitfalls (learn from these before you start)

- `df[df['TARGET']==1].count()` returns a per-column count, not the number of rows — use `(df['TARGET']==1).sum()`. (You've hit this one.)
- Empty or tiny bins make WOE explode to ±infinity — guard with an epsilon or merge.
- An IV above ~0.5 usually means leakage or an ID-like field, not a great feature. Be suspicious, not pleased.
- Comparing two groups' individual confidence intervals is not the same as testing their difference — test the difference directly.
- Don't read a "default rate by bin" number without holding it against the ~8% base rate.
- Skewed features (income, credit) will mislead quantile bins if you don't look at the distribution first.
- In the deck, resist showing the ranked table as a raw dump — a non-technical audience reads a bar chart, not a DataFrame.

## 11. Stretch goals (optional — for a comfortable HD)

- Add a monotonic-binning pass for the strongest numeric drivers and show the WOE trend is monotonic.
- Export the ranked table to CSV and render it as a clean formatted table.
- Add a short "data dictionary" cell describing each screened feature in one line.
- Note which two or three features look redundant (highly correlated drivers carrying the same signal).
- Render the Quarto deck straight from the analysis (embed the charts via code chunks) so the presentation regenerates when the data changes.

---

*When you're done, bring the notebook **and** the rendered Quarto deck back and I'll mark both against Section 9, with banded feedback per criterion.*
