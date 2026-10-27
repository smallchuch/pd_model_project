<!-- Draft written by AI from project notes. Reword into my own voice before publishing (task 1.4 / 7.6). Remove TODO comments when each section is done. -->

# Probability of Default (PD) Model for Consumer Lending

An end-to-end credit risk project: from raw application data to a deployed tool that estimates an applicant's probability of payment difficulty, assigns a risk grade, and suggests an assessment pathway.

**[Live app](TODO) · [Portfolio page](TODO) · [EDA / WOE / IV deck](TODO) · [Model deck](TODO)**

---

## 1. Purpose and business problem
<!-- Task 2.3.1 -->

Lenders need a consistent way to rank applicants by credit risk at the point of application, so effort goes where it matters: faster decisions for low-risk applicants, closer review for high-risk ones.

This project builds a PD model on application-time data and shows how its output could support (not replace) a credit assessment process:

- **Question:** Which applicant characteristics best predict payment difficulty, and how well can a transparent model rank applicants by risk?
- **Output:** an estimated probability of payment difficulty, a risk grade, and a recommended assessment pathway per grade.
- **Audience:** credit risk and lending teams who need a model they can explain to policy, validation and regulators.

## 2. Dataset
<!-- Task 2.3.2 -->

| | |
|---|---|
| Source | Home Credit Default Risk (Kaggle), `application_train` table |
| Scope | Application-time data only; the six behavioural / bureau history tables are not used |
| Target | `TARGET` = 1 if the client had payment difficulties (a lender-defined proxy, not a 90+ DPD or Basel default) |
| Rows used | 307,507 after cleaning: train 246,005 / test 61,502 (80/20 split, stratified, `random_state=42`) |
| Default rate | 8.07% in both train and test |

The data is not stored in this repo. Download it from Kaggle and place it in `TODO: data folder path`.

## 3. Project structure
<!-- Task 2.3.3 -->

```
Flagship PD Model Project/
├── 01_TODO_EDA_folder/              EDA, data quality, WOE / IV driver scan
├── 02_PD_Model_Development/          Baseline logistic regression scorecard
├── 03_Challenger_Model_LightGBM/     LightGBM challenger, same features and split
├── 04_Final_Results_and_Reccomendations/  Personas, risk grades, business recommendations
└── 05_Applicant_PD_Dashboard/        Streamlit app, model registry, tests
```

<!-- TODO: confirm folder names (01 folder name; consider fixing the "Reccomendations" spelling) and add deck file locations. -->

Each notebook follows the same pattern for every step: **Explanation** (what I expect before running), **Observation** (what the output showed), **Decision** (what I did about it).

## 4. EDA, WOE and IV
<!-- Task 2.3.4 -->

- Profiled data quality and missingness; treated informative missingness (e.g. external credit scores, the employment-days sentinel) as its own bin rather than imputing.
- Binned every candidate feature and ranked them by **Information Value (IV)**, with bins fitted on train only and applied to test to avoid leakage.
- Built affordability ratios (annuity to credit, credit to goods price, and others) because raw loan amounts carried little signal on their own.
- **Gender was excluded** from modelling as a protected attribute.

**Strongest drivers:** external credit bureau scores (`EXT_SOURCE_1/2/3`), followed by employment length, affordability ratios, occupation, organisation type, income type and education.

<!-- TODO: add the top-10 IV table or a link to the chart; one line on the most interesting finding. -->

## 5. Baseline model: logistic regression scorecard
<!-- Task 2.3.5 -->

A WOE-encoded logistic regression on **12 features**, chosen for transparency: each feature's effect is visible in its coefficient, as in a traditional scorecard.

| Success criterion (set before fitting) | Target | Result |
|---|---|---|
| Test Gini | at least 0.45 | **0.493** ✅ |
| Train vs test Gini gap | at most 0.02 | 0.003 ✅ |
| Coefficient signs | all in the expected direction | ✅ <!-- TODO confirm wording --> |
| Rank ordering across test PD deciles | no reversals | ✅ |

Two features were dropped during development: one for collinearity (r = 0.94 with employment days) and one for a wrong-signed coefficient.

## 6. Challenger model: LightGBM
<!-- Task 2.3.6 -->

A gradient-boosted tree model on the same 12 features, rows and split, used to benchmark how much signal the scorecard leaves on the table.

<!-- TODO once notebook 03 is done:
| Metric | Champion (logistic) | Challenger (LightGBM) |
|---|---|---|
| Test Gini | 0.493 | TODO |
| KS | 0.362 | TODO |
| Train/test gap | 0.003 | TODO |
Decision: retain scorecard / adopt challenger, and why. -->

## 7. Borrower personas and business recommendations
<!-- Task 2.3.7 -->

Example applicants, one per risk grade, show how the model separates applicants across the risk spectrum and what each grade means for the assessment process.

| Risk grade | Predicted PD band | Example persona | Recommended pathway |
|---|---|---|---|
| Lower risk | TODO | TODO | Streamlined, subject to normal policy and affordability checks |
| Standard risk | TODO | TODO | Standard credit assessment |
| Elevated risk | TODO | TODO | Enhanced assessment, extra verification and/or manual review |
| Highest risk | TODO | TODO | Highest level of manual review, or outside risk appetite per policy |

<!-- TODO: Notion plans four personas, but the app's master scale has five grades (<3%, 3-6%, 6-10%, 10-18%, 18%+). Settle on 4 or 5 so the README, app and deck match. -->

The model **supports** lending decisions. It is not a standalone approve / decline engine: policy rules, affordability and credit judgement still apply.

## 8. Streamlit application
<!-- Task 2.3.8 -->

An interactive applicant PD tool: enter an applicant's details and see their estimated probability of payment difficulty, risk grade, and how they compare with the portfolio average.

- Champion and challenger models are loaded from a **model registry**, so new models can be added without changing the app.
- Every model has **parity tests** confirming the app reproduces the notebook's predictions exactly.
- Model limitations are disclosed in the app itself.

<!-- TODO: screenshot or GIF; add once income + affordability inputs (6.2, 6.3) are live. -->

**[Open the live app](TODO)** (free hosting; may take about 30 seconds to wake up)

## 9. How to navigate and run
<!-- Task 2.3.9 -->

**Just skimming?** Read this page, then the decks. The notebooks hold the full reasoning.

**Running it locally:**

```bash
git clone TODO
cd TODO
pip install -r requirements.txt          # Python 3.12

# Run the app
streamlit run "Projects/Flagship PD Model Project/05_Applicant_PD_Dashboard/pd_app/app.py"

# Run the tests
python -m pytest tests -v
```

Notebooks run top to bottom in order (01 → 05) once the Kaggle data is in place.

## 10. Key results and takeaways
<!-- Task 2.3.10 -->

- A transparent 12-feature scorecard reaches a **test Gini of 0.49** with almost no overfitting (train/test gap 0.003).
- Risk grades separate cleanly: on in-policy test applicants, actual default rates rise from about **2% in the lowest grade to about 27% in the highest**, closely matching predicted PDs.
- External credit scores dominate, but affordability ratios add signal that raw loan amounts did not.
- <!-- TODO: challenger verdict in one line -->
- <!-- TODO: the one business takeaway a credit manager should remember -->

## 11. Assumptions and limitations
<!-- Task 2.3.11 -->

- **Target is a proxy.** "Payment difficulty" uses undisclosed lender thresholds, so it does not map directly to 90+ days past due or a Basel default.
- **Application data only.** Bureau and repayment history tables were not used; adding them would likely improve the model.
- **Not calibrated.** The model is assessed on ranking; predicted PDs are close to actual on test, but no formal calibration step was applied.
- **Single time period, random split.** No out-of-time validation is possible with this dataset, so stability over time is untested.
- **Gender excluded**, but a full proxy check on the retained features has not yet been run.
- **Anonymised organisation types** are grouped into a catch-all band, so that lookup in the app is approximate.
- **Illustrative only.** Risk grades, thresholds and pathways are examples, not a production credit policy.
