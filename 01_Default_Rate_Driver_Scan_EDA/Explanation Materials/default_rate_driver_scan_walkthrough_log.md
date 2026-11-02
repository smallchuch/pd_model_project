# Default-Rate Driver Scan — Cell-by-Cell Walkthrough Log

_Plain-English record of what each section of the notebook does, what we observed, and the decisions/understanding we landed on. Built collaboratively, section by section._

Notebook: `default_rate_driver_scan_notebook.ipynb`
Dataset: Home Credit Default Risk — `application_train` (307,511 rows × 122 columns)

---

## Dataset Context (Foreword)

Context the whole notebook sits on top of. Source: Kaggle "Home Credit Default Risk" competition (Home Credit Group, 2018).

- **Who the lender is.** Home Credit Group is an international **consumer-finance** lender whose purpose is lending to **thin-file / no-file borrowers** — people with little or no formal credit history, often in emerging markets. This is *structurally* higher-risk than a prime mortgage book by design. This is the key frame for the whole analysis.
- **Business problem.** Lend to underserved borrowers *responsibly*. With no traditional bureau score, they lean on **alternative data** — telco, transactional history, and pre-computed external risk scores (this is why `EXT_SOURCE_1/2/3` exist and are screened first: they're the closest thing to an off-the-shelf bureau score in this data).
- **What `TARGET` means.** Not a full default/charge-off. Official definition: **`TARGET = 1`** = client had **payment difficulties** (a late payment beyond a set number of days on at least one of the first several instalments); **`TARGET = 0`** = everyone else. "Bad" here is *early payment trouble*, a proxy for default. Worth stating in the assumptions section.
- **Table structure.** The competition ships **seven** linked tables. `application_train` / `application_test` is the static, one-row-per-application snapshot at application time — **the only table this project uses**. The other six (`bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`, `installments_payments`, `credit_card_balance`) are behavioural/historical and are joined in for a full model. Our scope is deliberately a **static application-time screen**, mirroring how a lender decides at point of application before behavioural data exists.

**Reference frame reminder:** 8% default is *roughly expected* for an underbanked consumer book — not the solvency alarm it would be for a prime AU residential mortgage portfolio (< 1%). Judge every "default rate by bin" number against the ~8% base rate.

---

## Project Setup (cells: intro through Data Loading)

**What this section does — plain English**

Plumbing that runs before any analysis. Frames the project and gets the environment into a known, repeatable state.

- **Framing.** Sets the objective — find the primary drivers of loan default, then turn them into WOE/IV binning decisions that feed the downstream PD scorecard. Establishes the **E/O/D convention**: **E**xplanation (hypothesis *before* looking), **O**bservation (what the data showed), **D**ecision (the action taken). Plus a table of contents.
- **Environment Setup.** Imports plus two notable bits:
  - `find_repo_root()` walks *up* the folder tree until it hits `CreditRiskLearning`, then adds `Core Resources` (+ `Scripts`) to `sys.path`. Makes the custom toolkit importable and the notebook runnable from anywhere the repo is cloned — no hardcoded paths.
  - Imports the house toolkit: `python_style_util`, `credit_utils` (`eda`, `quality`, `woe`, `stats`, `style`), `evaluation_utils`, `missingno`. `autoreload` picks up edits to those modules without a kernel restart.
  - `RANDOM_STATE = 42` + `np.random.seed(42)`.
- **Display Settings.** Pandas/NumPy/matplotlib cosmetics. Key line: `InteractiveShell.ast_node_interactivity = 'all'` auto-displays *every* bare expression in a cell, not just the last — that's why later cells stack several tables in one output.
- **Style Guide.** `show_tokens()` / `show_fonts()` renders the palette/font reference.
- **Data Loading.** Loads `application_train.csv` (307,511 × 122) and the column-description lookup (`encoding='cp1252'` because that file isn't UTF-8).

**Understanding we confirmed**

- `find_repo_root` + `sys.path.append`: solves portability. Python imports only from folders on its path; utils live outside the notebook folder, so their home is added first. Clone anywhere → imports + data paths still resolve.
- `ast_node_interactivity = 'all'`: lets one cell display multiple results; cost is stray expressions (or a reference to a huge object) rendering clutter. Suppress a line with a trailing semicolon.
- Reproducibility uses **two** mechanisms with the same number: `np.random.seed(42)` seeds NumPy's global RNG; scikit-learn tools take `random_state=RANDOM_STATE` explicitly (they ignore the global seed). Set early as a habit; at pure-EDA stage it isn't protecting much yet.

---

## Section 1 — First Look

**What this section does — plain English**

Establishes the basic facts of the dataset before any deep analysis: size, the target balance, and a quick way to look up what each column means.

- **Data Shape.** `df.shape` → `(307511, 122)`. Decision to bucket 122 columns into subsets before analysing is the right instinct — too many to reason about as one blob.
- **Class Imbalance.** `TARGET.value_counts` → **92% repaid / 8% default** (282,686 vs 24,825).
- **Description Lookup Reference.** Builds `description_index` — a `{column: plain-English description}` dict — by filtering the description file to the `application_{train|test}.csv` table **first**, then `.set_index('Row')`, then looping the actual columns.

**Understanding we confirmed**

- **Is 8% "high"?** Only relative to the wrong book. It's alarming for a prime AU mortgage portfolio; it's roughly *expected* for Home Credit's underbanked consumer book. Name your reference frame explicitly — a reviewer will check it.
- **"Respect the class imbalance."** Two concrete reasons:
  1. Never compare raw group counts without normalising — use rates / adjust for sample size / show sample sizes transparently.
  2. WOE/IV are naturally robust to the 92/8 split because each class is normalised against its *own* total (WOE = ln(%goods ÷ %bads) within a bin), so the overall ratio cancels. **But** imbalance still bites via **sparse bins**: only ~8% of any bin is bads, so fine bins can have tiny bad-counts → noisy WOE and *inflated* IV off a fluke. So "respect it" also means enforcing a **minimum-bads-per-bin floor** and not over-trusting granular bins. (Carry into Section 5.)
- **Why filter the description table before `.set_index('Row')`.** Column names repeat across the seven source tables, so `Row` is **non-unique** in the full file. `desc.set_index('Row').loc['SOME_COL']` on a non-unique index returns a **DataFrame of several rows**, not a scalar — so the dict would store a *Series* of competing descriptions instead of a clean string. Silently wrong, not an error. Filtering to the application table first guarantees a unique index and a scalar lookup.

**Decisions**

- **No resampling.** This is a screening exercise, not model training; WOE/IV are computed on the full distribution, so imbalance is a constraint to respect, not to correct.
- **Reframe the Class Imbalance decision note (action item).** The existing note's "priced-in riskier segment" half is correct; the "lender is vulnerable / illiquid" half is the mortgage frame leaking in and should be revised — 8% is expected for this book, not a solvency warning.

---

## Section 2 — Data Quality & Missingness

### 2.1 Statistical Subsets

**What it does — plain English**

Assigns every one of the 122 columns a *statistical* type (binary / discrete / continuous / categorical / ordinal / id), because how you analyse a column depends on its statistical role, not its raw pandas dtype. Two passes:

- **Pass 1 — auto-classify the easy ones:** `nunique == 2 → binary`; `nunique > 50 AND numeric → continuous`; everything else → `needs review` (then printed for eyeballing).
- **Pass 2 — manual assignment:** hand-label each reviewed column (`CNT_CHILDREN → discrete`, `NAME_* → categorical`, `REGION_RATING_* → ordinal`, `_MODE`/`_MEDI` building stats → continuous, etc.).
- **Invert + check:** `cols_by_type` flips the dict to `{type: [columns]}` for slicing (`df[cols_by_type['discrete']]`); a final `assert` confirms every column got a valid label.

**Understanding we confirmed**

- **`SK_ID_CURR` is silently mislabeled `continuous` (BUG / action item).** As the primary key it has 307,511 unique values + `int64` dtype, so Pass 1 auto-labels it `continuous`. The Pass 2 line meant to fix it to `'id'` actually targets `CODE_GENDER` (copy-paste slip), so it's never corrected. Because it *did* get a valid label, the final `assert` passes — a clean run that masks the error. Risk: anything looping over `cols_by_type['continuous']` would treat a meaningless ID as a predictor. **Fix: reassign `SK_ID_CURR → 'id'`; watch for it in Section 3.**
- **`CODE_GENDER` labeled `binary`.** Strictly it's 3-valued at this point (`M`/`F`/`XNA`). The label is "correct given the planned drop of the tiny `XNA` group," not correct as the data currently stands. Confirm the `XNA` count when we reach the drop.
- **What the `assert` protects.** It catches (a) unassigned columns and (b) invalid/typo'd labels (e.g. `'cateogrical'`). It **cannot** catch a valid-but-wrong label — exactly how `SK_ID_CURR = 'continuous'` slipped through. A check that only ever passes gives false confidence.

**Decisions**

- Type-driven analysis: use `cols_by_type[...]` to slice columns by statistical role downstream.
- Action item: correct the `SK_ID_CURR` labeling bug so the ID is excluded from feature analysis.

### 2.2 Thematic Subsets

**What it does — plain English**

Groups the same 122 columns by *business family* (not statistical role), so analysis and the stakeholder narrative can proceed one theme at a time. Builds `thematic_type_index` in three moves:

1. **Explicit lists** for irregular high-value core columns (`external_scores`, `loan_financials`, `demographics`, `employment`, `assets`, `contact`, `client_timeline`, `application_timing`, `id_target`).
2. **Regex rules** for the big regular families, applied only to columns not already claimed, first match wins: `document_flags`, `bureau_enquiries`, `social_circle`, `region`, `building_stats` (`(_AVG|_MODE|_MEDI)$`).
3. **Coverage check** — prints anything unassigned instead of dropping it silently. Result: **122/122 columns across 14 themes** (`building_stats` = 47, `document_flags` = 20).

`cols_by_theme` inverts it for block access: `df[cols_by_theme['external_scores']]`.

**Understanding we confirmed**

- **Two parallel indices, two jobs.** `cols_by_type` (statistical) tells you *how* to analyse a column (which plot/test/binning). `cols_by_theme` (thematic) tells you *what to analyse alongside it* (the family) and supports block keep/drop decisions and the narrative.
- **Precedence is deliberate.** Priority = explicit lists → then the five regex rules in listed order; enforced by `if col in index: continue` (skip already-claimed) + `break` on first regex match. Explicit-first means **human judgment beats the greedy pattern net** (`building_stats`'s `_AVG/_MODE/_MEDI$` suffix would otherwise vacuum up any hand-curated column with that suffix). Rule *order* lets a specific theme (`region`) claim a column before the catch-all (`building_stats`). No live collision exists in this data — it's defensive design against the collision an edit would introduce.
- **`SK_ID_CURR` is thematically correct (`id_target`) but statistically still wrong (`continuous`).** Open question for Section 3: if the continuous scan iterates `cols_by_theme`, the 2.1 bug is neutralised; if it iterates `cols_by_type['continuous']`, the ID leaks into the feature scan.

### 2.3 Missingness Assessment (Part A — sentinels, ranked missingness, fully-populated groups)

**What it does — plain English**

- **Disguised nulls.** `DAYS_EMPLOYED == 365243` is a sentinel (~1,000-year tenure = Home Credit's code for "not employed": pensioners/unemployed). Creates boolean flag `DAYS_EMPLOYED_SENTINEL` **first**, then replaces `365243 → NaN` (55,374 rows, ~18%). `CODE_GENDER == 'XNA'` (4 rows) is the same idea but the rows are **dropped**.
- **Ranked missingness chart.** `df.isnull().sum()/len(df)`, filtered to ≥5%, sorted, gradient bar chart. Columns ≥50% missing are almost all `building_stats`, plus `OWN_CAR_AGE` and `EXT_SOURCE_1`.
- **Fully-populated groups.** Loops `cols_by_theme`; themes with zero missing (`region`, `id_target`, `document_flags`, `application_timing`, `contact`) are excluded from the deeper dive.

**Understanding we confirmed**

- **Three kinds of missingness.** MCAR = missing for reasons unrelated to any data (dropping only costs sample). MAR = missingness explained by *other observed columns*. MNAR = missingness explained by *the missing value itself* / something unobserved — the gap is the signal. Hook: **MAR = explained by what you have; MNAR = explained by what you don't.**
- **Why flag `DAYS_EMPLOYED` before nulling.** It's **MNAR** — missing *because* the applicant isn't employed, which is itself predictive of default. Nulling alone would discard that signal; the flag preserves it.
- **Why the asymmetric treatment.** Drop a disguised-null when it's both tiny AND uninformative (`XNA`, 4 rows). Flag-and-keep when it's large OR meaningful (`DAYS_EMPLOYED`, 55k rows + real signal). Dropping `XNA` also yields a clean binary `CODE_GENDER` for later analysis.
- **Bar-label bug (fix).** `pct` is a fraction; the axis `PercentFormatter(xmax=1)` shows it correctly (66%), but `bar_label(fmt='%.2f%%')` formats the raw `0.66` → prints `0.66%` (100× understated). Fix: `labels=[f'{p*100:.1f}%' for p in missingness_pct['pct']]`.
- **High missingness ≠ drop.** In WOE binning, "missing" becomes its own bin with its own WOE; if being missing separates good/bad (e.g. `EXT_SOURCE_1`, ~56% missing) it *adds* IV. Need only adequate missing-bin volume + a sensible non-missing binning. Resolved formally in Sections 4.4 and 5.

**Decisions**

- Preserve informative missingness via sentinel flags; drop only tiny, uninformative disguised-nulls.
- Exclude fully-populated themes from the missingness deep dive for a cleaner view.
- Action items: fix the bar-label units; complete the `Prediction → Explanation` E/O/D relabel already self-flagged.

### 2.3 Missingness Assessment (Part B — per-theme pattern diagnosis)

**What it does — plain English**

Shifts from column-by-column to **block-by-block**, because the *pattern* of missingness reveals its mechanism.

- **Per-theme bar charts** (excl. fully-populated themes + `building_stats`), coloured by severity. Flags `social_circle` and `bureau_enquiries` as block-wise.
- **Building stats**: pivots the 47 cols into `base × suffix (AVG/MEDI/MODE)`; identical counts per base, and a nullity-correlation heatmap = 1.0 within each triplet → they miss on the same rows. Plan: possibly collapse each numeric triplet to one `AVG` + a `building_missing` sentinel (decided after univariate analysis).
- **Car data**: `(FLAG_OWN_CAR=='N').mean()` over car-age-missing rows = **1.0** → MAR, no action.
- **Bureau enquiries**: 6 cols share 41,519 missing, nullity corr 1.0.
- **Social circle**: 4 cols share 1,021 missing, nullity corr 1.0.
- **Crosstab**: bureau vs social missingness barely overlap (170 rows) → independent origins.

**Understanding we confirmed**

- **Equal count ≠ same rows.** Identical missing *counts* per base is necessary but not sufficient; the proof of "same rows" is the **within-triplet nullity correlation = 1.0**. The 0.62 global off-diagonal min is between *different attributes* and is expected to be lower.
- **Why some bases are `_MODE`-only.** `EMERGENCYSTATE`, `HOUSETYPE`, `WALLSMATERIAL`, `FONDKAPREMONT` are **categorical** — you can only take a mode, not an average/median, so no `_AVG`/`_MEDI` exists by construction.
- **Car `1.0` meaning.** `.mean()` on a boolean Series = proportion True → 100% of car-age-missing rows are non-owners. MAR and harmless because the signal ("owns no car") is already carried by `FLAG_OWN_CAR`; nulling loses nothing. If that flag didn't exist, you'd sentinel-flag it like `DAYS_EMPLOYED`. (Strictly: "structurally missing," a clean sub-case of MAR.)
- **Why test block independence.** Overlapping missing rows across two blocks → a single systemic cause (e.g. an upstream data-pull failure = likely non-informative pipeline artifact, treat together). Independent rows → separate origins, so each block's missingness may carry its own signal and is tested against `TARGET` separately in Section 5.

**Decisions**

- Diagnose missingness by pattern (block-wise vs scattered) before deciding treatment; defer informative-vs-not verdict to Section 5.
- Collapse building-stats triplets to `AVG` (+ sentinel) pending univariate results.

**Fixes applied to the notebook this session** (re-run the affected cells):

- `SK_ID_CURR` statistical type corrected to `'id'` (was silently `'continuous'`).
- Ranked-missingness bar labels fixed (fraction → true percent).
- Building-stats AVG chart labels fixed (`fmt='g%'` → real percentages).
- Building-stats correlation comment corrected (1.0 within triplet, ~0.62 across attributes).

### 2.4 Cardinality Assessment

**What it does — plain English**

Counts unique values across categorical + ordinal columns to find which need treatment before WOE/IV binning. Drops `HOUR_APPR_PROCESS_START`, `REGION_RATING_CLIENT`, `_W_CITY` (no merging/grouping action to take). The two high-cardinality problem features: `OCCUPATION_TYPE` (~18 levels) and `ORGANIZATION_TYPE` (~58 levels). Sparse-level check (levels under the threshold), then two treatments:
- `OCCUPATION_TYPE` → **frequency pooling** (rare levels merged to "other"; NaN kept as own category). Threshold set to **5%**.
- `ORGANIZATION_TYPE` → **supervised rate-banding** into `ORG_BAND`.

**ORG_BAND cell, understood in depth**

- `rate = df.groupby('ORGANIZATION_TYPE')['TARGET'].mean()` — mean of a 0/1 column = **default rate** per org type.
- `counts = value_counts()` — applicants per org type (needed for the volume floor).
- `pd.cut(rate, [0,0.06,0.08,0.10,1.0], labels=['<6%','6-8%','8-10%','>10%'])` — collapses 58 levels into 4 bands whose edges straddle the ~8% base rate.
- `band[(rate>0.10) & (counts>=500)] = <that level's own name>` — **rescues** big, high-risk org types into their own standalone band instead of pooling them; the `>=500` floor is the **min-bin-volume rule** (only break out a level with enough bodies to be stable). Small high-rate levels stay pooled in `>10%`.
- `df['ORG_BAND'] = df['ORGANIZATION_TYPE'].map(band.to_dict())` — 13 final bands.

Two-stage logic: (1) bin everyone by rate (fixes sparse levels); (2) rescue large high-risk levels (preserves signal where volume supports it).

**Understanding we confirmed**

- **Supervised binning = target leakage.** `ORG_BAND` is defined *using* `TARGET`, then screened for how well it predicts `TARGET` — circular ("drawing the bullseye around the arrows"), so its IV reads optimistically. Fix / discipline rule for ALL supervised transforms (this **and** WOE encoding): **split first, fit the binning on the TRAIN split only, then `.map()` onto test** — never let a test row influence how a bin is drawn.
- **Why two different treatments.** Frequency-pooling (OCCUPATION) is *unsupervised* — doesn't touch the target, no leakage. Rate-banding (ORGANIZATION) is *supervised* — carries the leakage caveat above.
- **Bands around the base rate** and the `>=500` volume floor both trace back to earlier principles (read every rate against ~8%; minimum bads/bin for stability).

**Decisions / action items**

- Threshold for OCCUPATION sparse-pooling set to 5% (comments + note text aligned).
- **ORG_BAND (and all WOE binning) to be (re)computed after the train/test split in Section 5.1**, not on full data — flagged inline in the notebook and in cell [74].
- Cell [69] hypothesis written; leftover 🚩 banners cleared where content exists.

---

## Section 3 — Feature Analysis

Approach: work one feature family at a time, Univariate → Bivariate → a single committed Decision that points forward to Section 5 binning. Uses the `cols_by_theme` scaffolding.

### 3.1 External Scores (`EXT_SOURCE_1/2/3`)

**What it does — plain English**

Characterises the three third-party credit scores (0–1 scale) and tests how they move with default, since these are the expected strongest drivers.

- **Univariate:** histograms (shapes differ per provider) + a missingness bar (EXT_SOURCE_1 56% missing, _3 20%, _2 0%).
- **Bivariate:** mean score by class; overlaid histograms by TARGET; **default rate by quantile bin**; discrimination (**AUC/Gini**); separation (**KS + ECDF**); and score **intercorrelation**.

**What the charts actually show (read off the plots)**

- Distributions: EXT_SOURCE_1 broadly symmetric (~0.5); _2 left-skewed, peak ~0.65, full coverage; _3 left-skewed, peak ~0.65–0.70. All bounded 0–1, no wild outliers.
- Separation: defaulters concentrate at **low** scores (overlay + ECDF). Default rate declines **monotonically** across bins — low bin ~20–23%, high bin ~2–3%, vs the 8.1% base. Textbook scorecard behaviour.
- Discrimination: Gini `_3 0.36 > _1 0.33 > _2 0.31`; coverage `_2 100% > _3 80% > _1 44%`.

**Understanding we confirmed**

- **Metrics.** AUC = probability the score ranks a random defaulter as riskier than a random non-defaulter (0.5 chance → 1.0 perfect). Gini = 2·AUC−1, the industry's scorecard metric. KS = the max vertical gap between the two ECDFs (separation at the single best cut-off). AUC and KS usually agree on ranking.
- **The `-df[c]` negation.** `roc_auc_score` assumes higher score = more likely the positive class (default). These scores run the other way (high = safe), so raw would give ~0.32 (backwards); negating points the score at the default class → 0.68. `AUC(raw) + AUC(negated) = 1`.
- **Bounded scores are a gift at binning time** — no outlier trimming needed, unlike the skewed `AMT_` financials coming in 3.2.
- **Gini vs coverage.** EXT_SOURCE_3 wins on Gini but is 20% missing; EXT_SOURCE_2 is a hair weaker but fully populated → most usable backbone in practice. Coverage changes the "best driver" answer versus reading Gini alone.
- **Why discrimination metrics only appear here.** AUC/Gini/KS measure how well an existing *score* ranks risk, so they belong with the external scores (pre-built scores benchmarked in their native metric). Raw features are screened with IV/WOE instead, since a non-monotonic/uncalibrated feature can have low AUC yet high IV.

**Decisions**

- Carry all three into Section 5 as **quantile-binned WOE features, with missing as its own bin** (missingness is informative, esp. EXT_SOURCE_1 at 56%). Retain all three pending the intercorrelation check; drop/combine only a strongly redundant pair (|r| > ~0.7).

**Changes made to the notebook**

- Wrote the empty Observation cells (univariate [86]; bivariate Observation/Decision [102]).
- Added the missing **Score Intercorrelation** code (3×3 Pearson heatmap, neutral `Blues` cmap) — the analysis was described but never coded; also matches the "compare rows across sources" instinct.
- Added a rationale line explaining why discrimination metrics appear only for this family.
- Fixed the one accessibility issue: the overlaid-by-TARGET histogram used red/green as the *sole* class cue → overrode just that call with a colour-blind-safe purple/gold `palette=` (matches the ECDF). Left the shared `binary_palette` untouched — its red/green is a deliberate risk convention used as a *redundant* cue in the bar charts.

### 3.2 Loan Financials (`AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `NAME_CONTRACT_TYPE`)

**What the charts show**

- Univariate: all four amounts heavily right-skewed even on a log axis; income has an extreme ~$100M outlier; credit/goods-price multimodal with round-number spikes; annuity ~log-normal. ~0% missing.
- Bivariate (equal-population quantile bins, ~30k each): **weak and non-monotonic**. Income near-flat around the 8.1% base; AMT_CREDIT / AMT_ANNUITY / AMT_GOODS_PRICE form an **inverted-U** (mid-range riskiest, tapering both ends). Mean-by-status: defaulters marginally lower on every amount. Contract type: cash 8.3% vs revolving 5.5% default (cash = 90.5% of book).

**Understanding we confirmed**

- **Why the inverted-U (domain read).** Low-value borrowers carry a small obligation well under capacity; top-end borrowers have the income to carry a big loan; the **middle** is where people stretch closest to their limit on an ordinary income — the most stressed segment. Raw amount can't prove this because it ignores capacity → the signal is in **ratios**, not levels.
- **Mean vs bin.** Trust the bin chart (equal-population quantiles → stable). The mean isn't wrong, it's uninformative for a non-monotonic shape — it averages the mid-peak away and can point the wrong way.
- **Log vs quantile bins.** Quantile binning is rank-based and `log` is monotonic, so log preserves rank → identical cut-points → identical WOE/IV. Log is a **visualisation aid only**; quantile bins are also robust to the income outlier (it just lands in the top bin), unlike a mean or equal-width bins.

**Decision**

- Carry raw amounts as quantile-binned WOE features but **expect low IV**. Real signal = affordability **ratios** in Section 4: repayment-to-income (`AMT_ANNUITY/AMT_INCOME` ≈ serviceability/NSR), loan-to-income (`AMT_CREDIT/AMT_INCOME` ≈ leverage), loan-to-value (`AMT_CREDIT/AMT_GOODS_PRICE` ≈ LVR; >1 = financed beyond asset value). No log needed for binning. Contract type binned as-is (2 levels).
- **Pending notebook fills (batch at Section 4 pause):** Observation [110] (univariate) and [117] (bivariate Observation/Decision) — text preserved above.

### 3.3 Age & Employment (`DAYS_BIRTH`, `DAYS_EMPLOYED` + employment categoricals)

**What the charts show**

- `DAYS_BIRTH` / `DAYS_EMPLOYED` stored as **negative days before application** → convert to positive years (`/-365.25`).
- **Age:** monotonic decline — 20–24 = 12.3% down to 65–69 = 3.7%.
- **Tenure:** monotonic decline — 0–1 yr ~11% down to 20+ yr = 4.2%; 18% missing (the `DAYS_EMPLOYED` sentinel — pensioners/unemployed — excluded from the chart).
- **Income type:** huge spread — Maternity leave ~40%, Unemployed ~37% (tiny counts) vs Pensioner ~5.3%, Student/Businessman 0% (tiny). Binning uses **fixed domain bands** (`method='width'`: 5-yr age bands, tenure milestones), not quantiles.
- **Occupation:** ordered, Low-skill Laborers ~17% → Accountants ~4.5% (~31% missing). **Organization** → `ORG_BAND`.

**Understanding we confirmed**

- **Why age/tenure are strong raw drivers.** They proxy experience, income stability and accumulated buffer, monotonic with default → expect high IV, well above the loan amounts.
- **Fixed-width vs quantile bins.** Fixed domain bands give interpretable brackets for decisioning and preserve monotonicity; the risk quantile bins don't have is **unequal populations** — thin bands at the extremes (young, old, long-tenure) must be count-checked and merged if sparse.
- **The sentinel/NaN group is heterogeneous.** It merges low-risk **pensioners** (dominant by count) with high-risk **unemployed**, so `DAYS_EMPLOYED_SENTINEL` is, on net, a **low-risk** flag — the opposite of "missing employment = danger." `NAME_INCOME_TYPE` is what disambiguates the two; the two features interact.
- **Sparse-level WOE trap.** WOE = ln(%goods÷%bads). Zero-default levels (Student/Businessman) → %bads = 0 → **WOE = ±∞**; tiny high-rate levels (Maternity/Unemployed) → noisy WOE, inflated IV. Fix: **pool** rare levels (into a level of *similar* default rate, not opposite), enforce a min goods+bads per bin, and fit the grouping on the **train split** (leakage).

**Decision**

- Age & tenure → fixed domain-band WOE (5-yr / tenure milestones); count-check thin extreme bands, merge if needed.
- `DAYS_EMPLOYED` → bin tenure + keep sentinel/missing as its own bin (informative, mostly pensioners → low risk); lean on `NAME_INCOME_TYPE` to split pensioner vs unemployed.
- Sparse employment categoricals → pool by similar rate + min-count + train-only grouping before WOE.
- **Pending notebook fills (batch at Section 4 pause):** Observation [123] and Observation/Decision [128].

### 3.4 Key Demographics (`CODE_GENDER`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`, `NAME_FAMILY_STATUS`, `NAME_EDUCATION_TYPE`, `NAME_HOUSING_TYPE`, `NAME_TYPE_SUITE`)

**What the charts show (default rate per level)**

- `CODE_GENDER`: F 7.0% / M 10.1%.
- `NAME_EDUCATION_TYPE`: Academic 1.8% → Higher 5.4% → … → Lower secondary 10.9% (clean gradient).
- `NAME_FAMILY_STATUS`: Widow ~5.8% → Single/Civil marriage ~9.9% (+ a 0% "Unknown" singleton).
- `NAME_HOUSING_TYPE`: ~6.6% → ~12.3% (with parents / rented highest).
- `CNT_CHILDREN` / `CNT_FAM_MEMBERS`: gentle rise then a noisy tail (28–33% and 100% levels = 1–2 applicants).
- `NAME_TYPE_SUITE`: narrow 7.4–9.8% (weak).

**Understanding we confirmed**

- **Gender is a protected attribute → excluded from the scorecard** on responsible-lending / anti-discrimination grounds, documented explicitly. Second-order risk: **proxy variables** — other features (occupation, organization) can encode gender and reintroduce disparate impact, so retained drivers must be checked for proxying.
- **Education is ordinal** → bin in natural order and expect a **monotonic WOE**; non-monotonicity is a flag. Sparse Academic-degree level likely merged up.
- **Discrete counts with thin tails** (`CNT_CHILDREN`/`CNT_FAM_MEMBERS`) → cap into `5+` (or `3+`) so 100%/0% singletons collapse into a stable band.
- **`CNT_CHILDREN` ≈ derivative of `CNT_FAM_MEMBERS`** → collinear/redundant; keep one, drop the other at feature selection (redundant-drivers stretch goal).

**Decision**

- Drop `CODE_GENDER` from the model (regulatory); keep as EDA context only; screen retained drivers for gender proxying.
- `NAME_EDUCATION_TYPE` → ordinal WOE in education order; strong expected driver.
- `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE` → nominal WOE, pool sparse levels ("Unknown").
- `CNT_CHILDREN`/`CNT_FAM_MEMBERS` → cap to `5+`; keep one of the two.
- **Pending notebook fills (batch at Section 4 pause):** Observation [133] and Observation/Decision [137].

---
