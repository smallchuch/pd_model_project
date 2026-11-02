"""Scorecard model: one raw row in, the standard result dict out.

Skeleton and docstrings written by AI. Function bodies marked TODO are for Chuch to write.
Everything this model needs lives in its own registry folder, so a challenger
can sit next to it with different bins and coefficients.
"""
import json
import pickle
from pathlib import Path
import os

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "registry" / "logreg_baseline_v1"


def load_model(model_dir=MODEL_DIR):
    """Read the three files in the model folder.
    Returns (metadata, bp, model):
      metadata.json        -> dict
      binning_process.pkl  -> fitted BinningProcess (pickle)
      model.joblib         -> fitted LogisticRegression (joblib)"""
    
    model = joblib.load(model_dir / f'model.joblib')
    
    with open(os.path.join(model_dir, 'binning_process.pkl'), 'rb') as fh:
        bp = pickle.load(fh)
        
    with open(os.path.join(model_dir, 'metadata.json')) as fh:
        metadata = json.load(fh)    
    
    return metadata, bp, model
        


def check_columns(row, features):
    """Fail loudly if the row is missing any column the model needs.
    row: one-row DataFrame from engineer(). features: list from metadata."""

    feature_list = []
    
    for feature in features:
        if feature not in row.columns:
            feature_list.append(feature)
    if len(feature_list) != 0:
        raise ValueError (f"Row is missing model columns: {feature_list}")



def to_woe(row, bp, features):
    """Raw values -> WOE, one variable at a time.
    Must pass metric_missing='empirical' (without it, Missing scores as 0).
    Returns a one-row DataFrame with columns in the same order as `features`."""
    
    woe = {}
    for name in features:
        woe[name] = bp.get_binned_variable(name).transform(
            row[name].values, metric='woe', metric_missing='empirical')
    return pd.DataFrame(woe, columns=features)
    
    


def score(row, model_dir=MODEL_DIR):
    """The only function the app calls.
    Returns {model, version, pd, features_used, test_gini}."""
    
    metadata, bp, model = load_model(model_dir)
    features = metadata['features']
    check_columns(row, features)
    woe = to_woe(row, bp, features)
    prob = float(model.predict_proba(woe)[:, 1][0])
    
    return {
        'model': metadata['name'],
        'version': metadata['version'],
        'pd': prob,
        'features_used': metadata['features'],
        'test_gini': metadata['gini_test'],
    }
