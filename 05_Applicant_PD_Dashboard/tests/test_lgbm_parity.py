"""Parity tests for the LightGBM challenger adapter. Written by AI.

Run from the 05_Applicant_PD_Dashboard folder:
    python -m pytest tests -v

What each test proves:
  1. Row by row     scoring one row at a time (as the app does) gives the same PD
                    as scoring the whole batch in the notebook
  2. Unseen label   a category never seen in training scores without an error
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = APP_DIR.parent / "02_PD_Model_Development" / "inputs"
sys.path.insert(0, str(APP_DIR / "pd_app"))

import lgbm_adapter  # noqa: E402

N_ROWS = 50


@pytest.fixture(scope="module")
def parts():
    return lgbm_adapter.load_model()


@pytest.fixture(scope="module")
def raw_rows():
    """First 50 rows of the test set, raw engineered features."""
    return pd.read_parquet(INPUT_DIR / "test_raw.parquet").head(N_ROWS)


# 1. Row by row matches batch ----------------------------------------------
def test_row_by_row_matches_batch(parts, raw_rows):
    metadata, cat_levels, model = parts
    features = metadata["features"]

    # batch: the same categorical handling as the notebook
    X = raw_rows[features].copy()
    for col, levels in cat_levels.items():
        X[col] = pd.Categorical(X[col], categories=levels)
    want = model.predict_proba(X)[:, 1]

    # one row at a time, the way the app sends them
    got = np.array([
        lgbm_adapter.score(raw_rows.iloc[[i]])["pd"] for i in range(N_ROWS)
    ])

    assert np.allclose(got, want, atol=1e-12, rtol=0)


# 2. Unseen category does not crash -----------------------------------------
def test_unseen_category_scores(raw_rows):
    row = raw_rows.iloc[[0]].copy()
    row["OCCUPATION_TYPE"] = "Astronaut"   # not a training level
    result = lgbm_adapter.score(row)
    assert 0.0 <= result["pd"] <= 1.0
