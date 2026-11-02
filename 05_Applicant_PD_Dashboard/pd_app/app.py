"""Applicant PD dashboard (Streamlit). Written by AI.

Run from the 05_Applicant_PD_Dashboard folder:
    streamlit run pd_app/app.py

Layout: result panel across the top, applicant form underneath, model notes
collapsed at the bottom. Styling lives in theme.py (matches the portfolio site).
The page only collects answers and displays results. All rules live in
config/*.json, all maths in shared_stage.py and the model adapters.

    form -> validate -> engineer -> registry.score_all -> result panel
"""
from datetime import date

import streamlit as st

import registry
import shared_stage as ss
import theme

st.set_page_config(page_title="Probability of Payment Diffuculty Model", page_icon="📊", layout="wide")
theme.inject()


# ---------------------------------------------------------------- config ---
@st.cache_data
def cfg(name):
    return ss.load_config(name)


emp = cfg("employment_rules")
loan = cfg("loan_terms")
ext = cfg("ext_score_bands")
area = cfg("area_rating_options")
notes = cfg("model_notes")["known_limitations_to_disclose"]
base_rate = cfg("portfolio_stats")["in_policy_default_rate"]

income_types = list(emp["income_types"]) + list(emp["out_of_policy"]["income_types"])
org_labels = [o["label"] for o in cfg("organisation_options")["options"]]
occ_labels = cfg("occupation_options")["options"]
edu_labels = [o["label"] for o in cfg("education_options")["options"]]
area_values = [o["value"] for o in area["options"]]
area_names = {o["value"]: o["label"] for o in area["options"]}
t = loan["term_months"]
terms = list(range(t["min"], t["max"] + 1, t["step"]))
rate = loan["repayment"]["assumed_rate_pa"]

ext_bands = ext["fields"]["EXT_SOURCE_1"]["bands"]
ext_options = [b["label"] for b in ext_bands] + [ext["missing_label"]]
ext_ranges = {b["label"]: b["score_range"] for b in ext_bands}


def ext_display(label):
    return f"{label} ({ext_ranges[label]})" if label in ext_ranges else label


def money(x):
    return f"${x:,.0f}"


# ------------------------------------------------------------------ hero ---
theme.hero(
    "Logistic Regression Champion Model · LightGBM Challenger Model",
    'Applicant <span class="accent">Payment Difficulty</span> Scoring',
    "Enter an applicant and a cash loan to see the scorecard's estimated probability "
    "of payment difficulty. Illustrative portfolio model trained on Kaggle Home Credit data.",
)

result_slot = st.container()          # filled after the form is submitted

# ------------------------------------------------------------------ form ---
# Sliders for the numbers first, then dropdowns for the categories.
today = date.today()

with st.form("applicant"):
    theme.kicker("Loan and employment")
    s1, s2 = st.columns(2, gap="medium")
    loan_amount = s1.slider("Loan amount", min_value=50_000, max_value=2_000_000,
                            value=500_000, step=5_000, format="$%,d")
    goods_price = s2.slider("Purchase price", min_value=50_000, max_value=2_000_000,
                            value=500_000, step=5_000, format="$%,d")
    s3, s4 = st.columns(2, gap="medium")
    term_months = s3.select_slider("Term (months)", options=terms, value=36)
    employment_years = s4.slider("Years in current employment", min_value=0.0,
                                 max_value=45.0, value=5.0, step=0.5, format="%.1f")

    st.write("")
    theme.kicker("Applicant")
    r1, r2, r3 = st.columns(3, gap="medium")
    dob = r1.date_input("Date of birth", value=date(today.year - 35, today.month, 1),
                        min_value=date(today.year - 80, 1, 1),
                        max_value=date(today.year - 18, today.month, 1),
                        format="DD/MM/YYYY")
    income_type = r2.selectbox("Income type", income_types)
    credit_band = r3.selectbox("Credit score", ext_options, index=2, format_func=ext_display)

    r4, r5, r6 = st.columns(3, gap="medium")
    occupation = r4.selectbox("Occupation", occ_labels)
    org_option = r5.selectbox("Organisation type", org_labels)
    education = r6.selectbox("Highest education", edu_labels)

    r7, r8, r9 = st.columns(3, gap="medium")
    area_rating = r7.selectbox(area["form_label"], area_values,
                               index=area_values.index(area["default"]),
                               format_func=area_names.get)
    r9.write("")
    submitted = r9.form_submit_button("Score applicant", type="primary")

# Age in years from date of birth (validate uses it; the model does not)
age = round((today - dob).days / 365.25, 1)

form = {
    "income_type": income_type, "age": age, "employment_years": employment_years,
    "org_option": org_option, "occupation": occupation, "education": education,
    "area_rating": area_rating,
    # One credit score answer feeds all three model scores (see model notes)
    "ext_1_band": credit_band, "ext_2_band": credit_band, "ext_3_band": credit_band,
    "loan_amount": loan_amount, "term_months": term_months, "goods_price": goods_price,
}

notes_box = st.container()            # model notes always sit at the very bottom


def show_notes(row=None):
    with notes_box:
        if row is not None:
            with st.expander("What the model saw"):
                st.dataframe(row.T.rename(columns={0: "value"}).astype(str))
        with st.expander("About this model"):
            for n in notes:
                st.markdown(f"- {n}")


# ------------------------------------------------------------- scoring ---
with result_slot:
    if not submitted:
        theme.empty_result("Fill in the applicant below and press <b>Score applicant</b>.")
        show_notes()
        st.stop()

    theme.scroll_to_top()             # any submission: bring the top panel into view

    # 1. Policy: out-of-policy applicants are never scored
    if income_type in emp["out_of_policy"]["income_types"]:
        st.error(emp["out_of_policy"]["message"])
        st.caption(emp["out_of_policy"]["income_types"][income_type])
        show_notes()
        st.stop()

    # 2. Validation: errors block, warnings are shown
    errors, warnings = ss.validate(form)
    for e in errors:
        st.error(e)
    if errors:
        show_notes()
        st.stop()
    for w in warnings:
        st.warning(w)

    # 3. Score with every model in the registry (champion first)
    row = ss.engineer(form)
    results = registry.score_all(row)
    champ, others = results[0], results[1:]

    if "error" in champ:
        st.error(f"Champion model could not score: {champ['error']}")
        show_notes(row)
        st.stop()

    # 4. Plain-language loan figures
    monthly = ss.repayment(loan_amount, term_months)
    stats = [
        (money(monthly), "Monthly repayment"),
        (money(monthly * term_months), f"Total repaid over {term_months} months"),
        (f"{loan_amount / goods_price:.0%}", "Loan as % of purchase price"),
    ]

    # 5. How this PD compares with the in-policy training population
    ratio = champ["pd"] / base_rate
    pill = "safe" if ratio < 1 else ("watch" if ratio < 1.5 else "bad")
    pill_text = f"{ratio:.1f}× the portfolio average ({base_rate:.1%})"

    chal_html = ""
    if others:
        chips = "".join(
            f'<div class="chal">{r["model"]} v{r["version"]}<br>'
            + (f'<b>{r["pd"]:.1%}</b> · Grade {ss.risk_grade(r["pd"])["grade"]}'
               if "error" not in r else "could not score")
            + "</div>" for r in others)
        chal_html = f'<div class="challengers">{chips}</div>'

    grade = ss.risk_grade(champ["pd"])
    theme.result_panel(f'{champ["pd"]:.1%}', pill, pill_text, stats, chal_html,
                       grade=grade)

show_notes(row)
