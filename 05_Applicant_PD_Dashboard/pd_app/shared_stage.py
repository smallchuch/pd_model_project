"""Shared stage: form answers in, one row of engineered raw inputs out.

Skeleton drafted by AI. Function bodies marked TODO are for Chuch to write.
The config files in ../config hold every rule, so nothing here is hardcoded.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config(name):
    """Read one config file, e.g. load_config('loan_terms')."""
    
    with open(CONFIG_DIR / f"{name}.json") as fh:
        
        return json.load(fh)


def years_to_days(years):
    """Years employed -> DAYS_EMPLOYED. Blank input returns NaN.
    Rule: employment_rules.json, 'years_to_days'."""
    
    return np.nan if pd.isna(years) else -(years * 365.25)


def ext_value(field, band_label):
    bands = load_config('ext_score_bands')

    if band_label == bands['missing_label']:             # 1
        return np.nan                    # 2

    for band in bands['fields'][field]['bands']:
        if band['label'] == band_label:      # 3
            return band['value']          # 4

    raise ValueError(f"Unknown band '{band_label}' for {field}")           # name the field and the label in the message


def repayment(loan_amount, term_months):
    """Monthly repayment from amount, term and the assumed rate.
    Formula and rate: loan_terms.json, 'repayment'."""
    
    terms = load_config('loan_terms')
    
    rate = terms['repayment']['assumed_rate_pa'] / 12       # convert to monthly rate
    
    payment = loan_amount * ((rate * (1 + rate)**term_months) / ((1 + rate)**term_months - 1))
    
    return payment


def safe_div(num, den):
    """Same rule as the driver scan: zero or blank denominator returns NaN."""
    
    return np.nan if pd.isna(den) or den == 0 else num / den
    
    


def org_band(option_label):
    """Plain-language organisation option -> ORG_BAND level.
    Table: organisation_options.json."""
    
    org_bands = load_config('organisation_options')
    
    for org in org_bands['options']:
        if org['label'] == option_label:
            return org['org_band']

    raise ValueError(f"Unknown organisation option '{option_label}'")
    

def education_value(label):
    """Plain-language education label -> NAME_EDUCATION_TYPE dataset value.
    Table: education_options.json."""

    edu = load_config('education_options')

    for option in edu['options']:
        if option['label'] == label:
            return option['dataset_value']

    raise ValueError(f"Unknown education type '{label}'")


def engineer(form):
    """form: dict of answers from the app. Returns a one-row DataFrame
    with the 12 model columns."""

    # 1. Policy guard: out-of-policy applicants never get engineered
    rules = load_config('employment_rules')
    if form['income_type'] in rules['out_of_policy']['income_types']:
        raise ValueError(f"'{form['income_type']}' is outside lending policy")

    # 2. Loan maths (do this once, reuse below)
    amount = form['loan_amount']
    monthly = repayment(amount, form['term_months'])

    # 3. Build the row: one key per model column
    row = {
        'EXT_SOURCE_1':  ext_value('EXT_SOURCE_1', form['ext_1_band']),
        'EXT_SOURCE_2':  ext_value('EXT_SOURCE_2', form['ext_2_band']),
        'EXT_SOURCE_3':  ext_value('EXT_SOURCE_3', form['ext_3_band']),
        'DAYS_EMPLOYED': years_to_days(form['employment_years']),
        'ANNUITY_CREDIT': safe_div(monthly, amount),
        'AMT_GOODS_PRICE': form['goods_price'],
        'CREDIT_GOODS':  safe_div(amount, form['goods_price']),
        'ORG_BAND':      org_band(form['org_option']),
        'OCCUPATION_TYPE': form['occupation'],
        'NAME_INCOME_TYPE': form['income_type'],
        'REGION_RATING_CLIENT_W_CITY': form['area_rating'],
        'NAME_EDUCATION_TYPE': education_value(form['education']),
    }

    return pd.DataFrame([row])


def validate(form):
    """Check form answers before engineer(). Written by AI.
    Returns (errors, warnings), two lists of strings.
    errors block scoring; warnings are shown but scoring still runs.
    Rules: employment_rules.json 'age_checks', loan_terms.json 'term_months'."""
    rules = load_config('employment_rules')['age_checks']
    term = load_config('loan_terms')['term_months']
    errors, warnings = [], []

    age = form['age']
    years = form['employment_years']
    months = form['term_months']

    # 1. Employment vs age (only if both were answered)
    if not pd.isna(age) and not pd.isna(years):
        start_age = age - years
        if years > age - rules['min_working_age_hard']:
            errors.append(
                f"{years} years employed is not possible at age {age} "
                f"(would mean starting work before {rules['min_working_age_hard']}).")
        elif start_age < rules['min_working_age_warning']:
            warnings.append(
                f"Implies starting work at {start_age:g}, younger than "
                f"{rules['min_working_age_warning']}. Check the employment years.")

    # 2. Term must be one of the offered terms (12 to 60 in steps of 6)
    if months < term['min'] or months > term['max'] or (months - term['min']) % term['step'] != 0:
        errors.append(
            f"Term of {months} months is not offered. Choose {term['min']} to "
            f"{term['max']} months in steps of {term['step']}.")

    return errors, warnings


def risk_grade(pd_value):
    """PD -> grade dict from the master scale. Written by AI.
    Bands include their lower bound: PD of exactly 3% is Grade 2.
    Table: risk_grades.json."""
    for g in load_config('risk_grades')['grades']:
        if g['pd_min'] <= pd_value < g['pd_max']:
            return g
    raise ValueError(f"PD {pd_value} is outside the risk grade scale")
