"""Model registry: find every model folder and score a row with each one.

Written by AI.

How it works
  registry/
    logreg_baseline_v1/   metadata.json, model.joblib, binning_process.pkl
    <challenger>/         metadata.json, ...      <- drop a folder in, it appears

Each metadata.json names an "adapter": the pd_app module that knows how to
score that model (default "scorecard"). Every adapter has the same function,
score(row, model_dir), returning the standard result dict. So the app never
needs to know what kind of model it is talking to.
"""
import importlib
import json
from pathlib import Path

REGISTRY_DIR = Path(__file__).resolve().parent.parent / "registry"
ROLE_ORDER = {"champion": 0, "challenger": 1}


def list_models(registry_dir=REGISTRY_DIR):
    """Every folder with a metadata.json, champion first. Returns a list of dicts."""
    models = []
    for meta_path in sorted(Path(registry_dir).glob("*/metadata.json")):
        with open(meta_path) as fh:
            meta = json.load(fh)
        models.append({
            "name": meta["name"],
            "version": meta["version"],
            "role": meta.get("role", "challenger"),
            "adapter": meta.get("adapter", "scorecard"),
            "calibration": meta.get("calibration", "Not calibrated"),
            "dir": meta_path.parent,
        })
    return sorted(models, key=lambda m: (ROLE_ORDER.get(m["role"], 9), m["name"]))


def score_all(row, registry_dir=REGISTRY_DIR):
    """Score one engineered row with every model.
    A model that fails returns an 'error' entry instead of stopping the page,
    so one broken challenger never hides the champion's result."""
    results = []
    for m in list_models(registry_dir):
        base = {"role": m["role"], "calibration": m["calibration"]}
        try:
            adapter = importlib.import_module(m["adapter"])
            results.append(base | adapter.score(row, m["dir"]))
        except Exception as exc:
            results.append(base | {"model": m["name"], "version": m["version"],
                                   "error": f"{type(exc).__name__}: {exc}"})
    return results
