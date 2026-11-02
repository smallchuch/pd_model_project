"""Parity and lookup tests for the applicant PD app. Written by AI.

Run from the 05_Applicant_PD_Dashboard folder:
    python -m pytest tests -v

What each test proves:
  1. WOE parity      the app's to_woe() reproduces the notebook's test_woe.parquet exactly
  2. PD parity       score() gives the same PD as model.predict_proba on the stored WOE
  3. EXT bands       each form band lands in its intended bin (5 distinct WOEs + Missing)
  4. Organisation    every plain-language option agrees with the notebook's org_band_map.json
  5. End to end      a sample form goes validate -> engineer -> score and returns a sane result
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = APP_DIR.parent / "02_PD_Model_Development" / "inputs"
sys.path.insert(0, str(APP_DIR / "pd_app"))

import scorecard          # noqa: E402
import shared_stage       # noqa: E402

N_ROWS = 50


@pytest.fixture(scope="module")
def model_parts():
    """Load the registry model once for all tests."""
    return scorecard.load_model()


@pytest.fixture(scope="module")
def test_rows(model_parts):
    """First 50 rows of the test set: raw inputs and the notebook's WOE."""
    features = model_parts[0]["features"]
    raw = pd.read_parquet(INPUT_DIR / "test_raw.parquet").head(N_ROWS)
    woe = pd.read_parquet(INPUT_DIR / "test_woe.parquet").head(N_ROWS)[features]
    return raw, woe


# 1. WOE parity ------------------------------------------------------------
def test_woe_matches_notebook(model_parts, test_rows):
    metadata, bp, _ = model_parts
    features = metadata["features"]
    raw, want = test_rows

    # one row at a time, the way the app sends them
    got = pd.concat([scorecard.to_woe(raw.iloc[[i]], bp, features) for i in range(N_ROWS)],
                    ignore_index=True)

    assert list(got.columns) == features
    # 1e-12 not exactly 0: floating-point rounding differs slightly across
    # numpy/Python builds (5.6e-17 seen). A real binning mismatch is 0.01 to 0.44.
    assert np.allclose(got.values, want.values, atol=1e-12, rtol=0)


# 2. PD parity -------------------------------------------------------------
def test_pd_matches_predict_proba(model_parts, test_rows):
    _, _, model = model_parts
    raw, woe = test_rows

    got = np.array([scorecard.score(raw.iloc[[i]])["pd"] for i in range(N_ROWS)])
    want = model.predict_proba(woe)[:, 1]

    assert np.allclose(got, want, atol=1e-12)


def test_score_result_shape(test_rows):
    raw, _ = test_rows
    result = scorecard.score(raw.iloc[[0]])

    assert set(result) == {"model", "version", "pd", "features_used", "test_gini"}
    assert isinstance(result["pd"], float)
    assert 0.0 < result["pd"] < 1.0


# 3. EXT bands land in their intended bins ---------------------------------
@pytest.mark.parametrize("field", ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
def test_ext_bands_hit_intended_bins(model_parts, field):
    _, bp, _ = model_parts
    cfg = shared_stage.load_config("ext_score_bands")
    bands = cfg["fields"][field]["bands"]
    binned = bp.get_binned_variable(field)

    woes = []
    for band in bands:
        value = shared_stage.ext_value(field, band["label"])
        woe = binned.transform(np.array([value]), metric="woe", metric_missing="empirical")[0]
        assert woe == pytest.approx(band["woe"], abs=1e-4), band["label"]
        woes.append(woe)

    assert len(set(np.round(woes, 6))) == len(bands)          # 5 distinct bins
    assert woes == sorted(woes)                                # worst band -> best band

    # "No Credit Score" -> NaN -> the Missing bin's own WOE (not 0)
    missing = shared_stage.ext_value(field, cfg["missing_label"])
    assert pd.isna(missing)


# 4. Organisation lookup agrees with the notebook --------------------------
def test_org_options_match_band_map():
    options = shared_stage.load_config("organisation_options")["options"]
    with open(INPUT_DIR / "org_band_map.json") as fh:
        band_map = json.load(fh)

    named = [o for o in options if all(c in band_map for c in o["covers"])]
    other = [o for o in options if o not in named]
    assert len(other) == 1          # only the catch-all "Other business..." option

    # each named option: its covered types all sit in its band
    for option in options:
        assert shared_stage.org_band(option["label"]) == option["org_band"]
    for option in named:
        for org_type in option["covers"]:
            assert band_map[org_type] == option["org_band"], f"{option['label']}: {org_type}"

    # The catch-all is NOT lossless: the 27 anonymised types span several bands
    # (e.g. Trade: type 3 and Industry: type 3 have their own bins), so they all
    # get the catch-all band. This test pins down the justification instead:
    # the catch-all uses the most common band among in-policy applicants.
    raw = pd.read_parquet(INPUT_DIR / "train_raw.parquet", columns=["ORG_BAND", "NAME_INCOME_TYPE"])
    in_policy = raw[~raw["NAME_INCOME_TYPE"].isin(
        shared_stage.load_config("employment_rules")["out_of_policy"]["income_types"])]
    assert other[0]["org_band"] == in_policy["ORG_BAND"].mode()[0]


def test_unknown_org_raises():
    with pytest.raises(ValueError):
        shared_stage.org_band("Not a real option")


# 5. End to end: form -> validate -> engineer -> score ---------------------
def sample_form():
    edu = shared_stage.load_config("education_options")["options"][1]["label"]
    org = shared_stage.load_config("organisation_options")["options"][0]["label"]
    occ = shared_stage.load_config("occupation_options")["options"][0]
    return {
        "income_type": "Working", "age": 35, "employment_years": 6,
        "org_option": org, "occupation": occ, "education": edu, "area_rating": 2,
        "ext_1_band": "Good", "ext_2_band": "Good", "ext_3_band": "No Credit Score",
        "loan_amount": 450000, "term_months": 36, "goods_price": 400000,
    }


def test_form_end_to_end():
    form = sample_form()
    errors, _ = shared_stage.validate(form)
    assert errors == []

    row = shared_stage.engineer(form)
    result = scorecard.score(row)
    assert 0.0 < result["pd"] < 1.0


def test_out_of_policy_is_not_scored():
    form = sample_form() | {"income_type": "Pensioner"}
    with pytest.raises(ValueError):
        shared_stage.engineer(form)


# 6. Risk grades: master scale boundaries ----------------------------------
@pytest.mark.parametrize("pd_value, grade", [
    (0.0, 1), (0.0299, 1), (0.03, 2), (0.0599, 2), (0.06, 3),
    (0.0999, 3), (0.10, 4), (0.1799, 4), (0.18, 5), (0.99, 5),
])
def test_risk_grade_boundaries(pd_value, grade):
    assert shared_stage.risk_grade(pd_value)["grade"] == grade


def test_risk_grade_scale_has_no_gaps():
    grades = shared_stage.load_config("risk_grades")["grades"]
    assert grades[0]["pd_min"] == 0.0 and grades[-1]["pd_max"] == 1.0
    for lower, upper in zip(grades, grades[1:]):
        assert lower["pd_max"] == upper["pd_min"]
