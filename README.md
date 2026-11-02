<!-- Draft written by AI from project notes. Reword into my own voice before publishing (task 1.4 / 7.6). Remove TODO comments when each section is done. -->

# Probability of Default (PD) Model for Consumer Lending

An end-to-end credit risk project: from raw application data to a deployed tool that estimates an applicant's probability of payment difficulty, assigns a risk grade, and shows the assessment path for that grade.

**[Live app](TODO) · [Portfolio page](TODO) · [EDA deck](01_Default_Rate_Driver_Scan_EDA/deck/01_eda.html) · [Baseline deck](02_PD_Model_Development/deck/02_pd_model.html) · [Challenger deck](03_Challenger_Model_LightGBM/deck/03_challenger.html) · [Results deck](04_Final_Results_and_Reccomendations/deck/pd_recommendations.html)**

---

## 1. Purpose and business problem

Lenders need a consistent way to rank applicants by credit risk at the point of application, so effort goes where it matters: faster decisions for low-risk applicants, closer review for high-risk ones.

This project builds a PD model on application-time data and shows how its output could support (not replace) a credit assessment process:

- **Question:** Which applicant characteristics best predict payment difficulty, and how well can a transparent model rank applicants by risk?
- **Output:** an estimated probability of payment difficulty, a risk grade, and an assessment path per grade.
- **Audience:** credit risk and lending teams who need a model they can explain to policy, validation and regulators.

## 2. Dataset

| | |
|---|---|
| Source | Home Credit Default Risk (Kaggle), `application_train` table |
| Scope | Application-time data only; the six behavioural / bureau history tables are not used |
| Target | `TARGET` = 1 if the client had payment difficulties (a lender-defined proxy, not a 90+ DPD or Basel default) |
| Rows used | 307,507 after cleaning: train 246,005 / test 61,502 (80/20 split, stratified on the target, `random_state=42`) |
| Default rate | 8.07% in both train and test |

The data is not stored in this repo. Download it from Kaggle and place `application_train.csv` in `Datasets/Home Credit Default Risk/` at the repo root. Running notebook 01 creates the raw and WOE frames that notebooks 02 and 03 read.

## 3. Project structure

```
pd_model_project/
├── 01_Default_Rate_Driver_Scan_EDA/        EDA, data quality, WOE / IV driver scan (notebook + deck)
├── 02_PD_Model_Development/                Baseline logistic regression scorecard (notebook + deck, inputs/)
├── 03_Challenger_Model_LightGBM/           LightGBM challenger, same features and split (notebook + deck)
├── 04_Final_Results_and_Reccomendations/   Results deck, borrower personas, assessment path by grade
├── 05_Applicant_PD_Dashboard/              Streamlit app, model registry, config, tests
├── lib/                                    Shared helpers used by the notebooks and decks
├── Model Defense Work/                     Self-check questions on the EDA
├── PD_Model_Project_Outline.md             Stage-by-stage reference for the pipeline
├── requirements.txt
└── README.md
```

<!-- TODO: fix the "Reccomendations" spelling in the folder name once deck references are checked. -->

Each notebook follows the same pattern for every step: **Explanation** (what I expect before running), **Observation** (what the output showed), **Decision** (what I did about it).

## 4. EDA, WOE and IV

- Profiled data quality and missingness; treated informative missingness (e.g. external credit scores, the employment-days sentinel) as its own bin rather than imputing.
- Binned every candidate feature and ranked them by **Information Value (IV)**, with bins fitted on train only and applied to test to avoid leakage.
- Built affordability ratios (annuity to credit, credit to goods price, and others) because raw loan amounts carried little signal on their own.
- **Gender was excluded** from modelling as a protected attribute.

**Strongest drivers:** external credit bureau scores (`EXT_SOURCE_1/2/3`), followed by employment length, affordability ratios, occupation, organisation type, income type and education.

<!-- TODO: add the top-10 IV table or a link to the chart; one line on the most interesting finding. -->

## 5. Baseline model: logistic regression scorecard

A WOE-encoded logistic regression on **12 features**, chosen for transparency: each feature's effect is visible in its coefficient, as in a traditional scorecard.

| Success criterion (set before fitting) | Target | Result |
|---|---|---|
| Test Gini | at least 0.45 | **0.493** ✅ |
| Train vs test Gini gap | at most 0.02 | 0.003 ✅ |
| Coefficient signs | no significantly wrong sign | 11 of 12 negative. The one positive coefficient (NAME_INCOME_TYPE, +0.06) has a 95% interval spanning zero, so it passes ✅ |
| Rank ordering across test PD deciles | no reversals | ✅ from 1.4% in the lowest decile to 26.3% in the highest, with no reversals |

Two features were dropped during development: one for collinearity (r = 0.94 with employment days) and one for a wrong-signed coefficient in the first fit.

<!-- TODO: add the 95% intervals for the other 11 coefficients to notebook 02 (cell 54 still has a placeholder). -->

## 6. Challenger model: LightGBM

A gradient-boosted tree model on the same 12 features, rows and split, used to benchmark how much signal the scorecard leaves on the table.

| Metric (test unless stated) | Champion (logistic) | Challenger (LightGBM, refit) |
|---|---|---|
| Gini | 0.493 | 0.522 |
| KS | 0.362 | 0.389 |
| Train / test Gini gap | 0.003 | 0.054 |

The uplift is +0.029 Gini. The agreed criteria put that in the "retain and review" band (0.008 to 0.03), so the scorecard stays as champion. The first LightGBM fit failed the train/test gap check (0.065 against a 0.06 limit), so it was refitted with stronger regularisation (`num_leaves` 20, `min_child_samples` 300).

**Decision:** keep the scorecard as champion. Review the GBM's interactions to see whether any of the uplift is worth bringing into the scorecard.

<!-- TODO: confirm the decision wording and record the interaction review once it is done. -->

## 7. Borrower personas and assessment paths

Five example applicants, one per risk grade, show how the model separates applicants across the risk spectrum. Each borrows 500,000 over 36 months, so only the risk traits differ.

| Grade | Predicted PD band | Example persona | Assessment path |
|---|---|---|---|
| 1 · Very low risk | under 3% | The Steady Professional | Streamlined assessment |
| 2 · Low risk | 3% to 6% | The Established Worker | Standard assessment |
| 3 · Moderate risk | 6% to 10% | The Thin-Margin Earner | Standard assessment |
| 4 · Elevated risk | 10% to 18% | The Early-Career Stretch | Additional assessment |
| 5 · High risk | 18% and above | The Fragile Applicant | Refer or exclude under the selected cut-off |

<!-- TODO: confirm the assessment-path wording against your policy; it currently follows the results deck. -->

The model **supports** lending decisions. It is not a standalone approve / decline engine: policy rules, affordability and credit judgement still apply.

## 8. Streamlit application

An interactive applicant PD tool: enter an applicant's details and see their estimated probability of payment difficulty, risk grade, and how they compare with the portfolio average.

- Champion and challenger models are loaded from a **model registry**, so new models can be added without changing the app.
- Every model has **parity tests** confirming the app reproduces the notebook's predictions exactly.
- Model limitations are disclosed in the app itself.

<!-- TODO: screenshot or GIF; add once income + affordability inputs (6.2, 6.3) are live. -->

**[Open the live app](TODO)** (free hosting; may take about 30 seconds to wake up)

## 9. How to navigate and run

**Just skimming?** Read this page, then the decks. The notebooks hold the full reasoning.

**Running it locally:**

```bash
git clone https://github.com/smallchuch/pd_model_project.git
cd pd_model_project
pip install -r requirements.txt          # Python 3.12

# Run the app
cd 05_Applicant_PD_Dashboard
streamlit run pd_app/app.py

# Run the tests (from 05_Applicant_PD_Dashboard)
python -m pytest tests -v
```

**Rerunning the modelling:** put the Kaggle data in place (section 2), then run notebook 01, then 02, then 03. Run each notebook from its own folder. Notebook 01 creates the inputs that 02 and 03 read.

## 10. Key results and takeaways

- A transparent 12-feature scorecard reaches a **test Gini of 0.49** with almost no overfitting (train/test gap 0.003).
- On 50,349 in-policy test applicants, actual default rates rise from **1.9% in Grade 1 to 26.8% in Grade 5**, close to the predicted 2.0% and 26.2%.
- External credit scores dominate, but affordability ratios add signal that raw loan amounts did not.
- A LightGBM challenger reaches a test Gini of 0.52, a gain of 0.03 over the scorecard. That is a borderline uplift, so the scorecard stays as champion.
- <!-- TODO: the one business takeaway a credit manager should remember -->

## 11. Assumptions and limitations

- **Target is a proxy.** "Payment difficulty" uses undisclosed lender thresholds, so it does not map directly to 90+ days past due or a Basel default.
- **Application data only.** Bureau and repayment history tables were not used; adding them would likely improve the model.
- **Not calibrated.** The model is assessed on ranking; predicted PDs are close to actual on test, but no formal calibration step was applied.
- **Single time period, random split.** No out-of-time validation is possible with this dataset, so stability over time is untested.
- **Gender excluded**, but a full proxy check on the retained features has not yet been run.
- **Anonymised organisation types** are grouped into a catch-all band, so that lookup in the app is approximate.
- **Illustrative only.** Risk grades, thresholds and assessment paths are examples, not a production credit policy.
