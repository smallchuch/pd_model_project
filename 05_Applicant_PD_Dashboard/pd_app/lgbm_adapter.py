"""LightGBM challenger: one engineered row in, the standard result dict out.

Written by AI. Same contract as scorecard.py: score(row, model_dir) returns
{model, version, pd, features_used, test_gini}. registry.py adds role and
calibration. Unlike the scorecard there is no WOE step: the four categorical
columns are converted to pandas Categorical using the levels saved from
training (category_levels.json), so each label keeps the same internal code.
A label never seen in training becomes missing, which LightGBM handles.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "registry" / "lgbm_challenger_v1"


def load_model(model_dir=MODEL_DIR):
    """Read the three files in the model folder.
    Returns (metadata, cat_levels, model)."""
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    with open(model_dir / "metadata.json") as fh:
        metadata = json.load(fh)
    with open(model_dir / "category_levels.json") as fh:
        cat_levels = json.load(fh)
    return metadata, cat_levels, model


def check_columns(row, features):
    """Fail loudly if the row is missing any column the model needs."""
    missing = [f for f in features if f not in row.columns]
    if missing:
        raise ValueError(f"Row is missing model columns: {missing}")


def to_model_frame(row, features, cat_levels):
    """Select the model columns in training order and fix the categorical levels."""
    X = row[features].copy()
    for col, levels in cat_levels.items():
        X[col] = pd.Categorical(X[col], categories=levels)
    return X


def score(row, model_dir=MODEL_DIR):
    """The only function the app calls. Returns the standard result dict."""
    metadata, cat_levels, model = load_model(model_dir)
    features = metadata["features"]
    check_columns(row, features)
    X = to_model_frame(row, features, cat_levels)
    prob = float(model.predict_proba(X)[:, 1][0])

    return {
        "model": metadata["name"],
        "version": metadata["version"],
        "pd": prob,
        "features_used": features,
        "test_gini": metadata["gini_test"],
    }
