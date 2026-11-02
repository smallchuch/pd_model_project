"""
woe — Weight-of-Evidence and Information Value for feature screening.

Convention (from the project brief):
  * Good = non-default (TARGET == 0), Bad = default (TARGET == 1).
  * WOE_i  = ln( (Good_i / Good_total) / (Bad_i / Bad_total) )
  * IV     = sum_i (Good%_i - Bad%_i) * WOE_i
  * Empty/zero bins are epsilon-guarded (1e-6) so WOE never blows up to
    +/-inf.

Deliberately dependency-free of matplotlib/seaborn — this module only
computes. Plotting lives in eda.plot_woe, so woe_table()/iv() stay usable in
a headless context (a scoring job, a unit test, another project) with no
chart dependency along for the ride.

Contents
--------
woe_table : per-bin Good/Bad counts, WOE and bin IV for one feature.
iv        : single IV number for a feature (wraps woe_table).
iv_band   : label an IV value against the standard Siddiqi thresholds.
rank_iv   : ranked IV table across many screened features at once.
"""

import numpy as np
import pandas as pd


def woe_table(df, feature, target, bins=10, is_categorical=False):
    """
    Bin `feature` and compute WOE / IV per bin against a binary `target`.

    Convention: target == 1 is BAD (default), target == 0 is GOOD.
    Positive WOE = bin is safer than the portfolio average; negative = riskier.

    Parameters
    ----------
    df : source dataframe
    feature : column to bin and evaluate
    target : binary column (1 = bad, 0 = good)
    bins : number of quantile bins for numeric features (ignored if
        is_categorical=True)
    is_categorical : True to bucket by the feature's existing categories
        instead of cutting a numeric range

    Returns
    -------
    pd.DataFrame, one row per bin, columns:
        bucket, total, bad, good, dist_good, dist_bad, woe, iv_bin, bad_rate

    A "Missing" bucket is included whenever the feature has nulls, for both
    categorical and numeric features — missingness is often itself
    predictive in credit data, so it's never silently dropped.
    """
    d = df[[feature, target]].copy()

    if is_categorical:
        d["bucket"] = d[feature].astype("object").fillna("Missing")
    else:
        cut = pd.qcut(d[feature], q=bins, duplicates="drop")
        d["bucket"] = cut.cat.add_categories(["Missing"]).fillna("Missing")

    g = d.groupby("bucket", observed=False)[target].agg(["count", "sum"])
    g.columns = ["total", "bad"]
    g = g[g["total"] > 0]   # drop unused categories, e.g. an empty "Missing"
    g["good"] = g["total"] - g["bad"]

    total_good, total_bad = g["good"].sum(), g["bad"].sum()
    g["dist_good"] = (g["good"] / total_good).replace(0, 1e-6)
    g["dist_bad"] = (g["bad"] / total_bad).replace(0, 1e-6)

    g["woe"] = np.log(g["dist_good"] / g["dist_bad"])
    g["iv_bin"] = (g["dist_good"] - g["dist_bad"]) * g["woe"]
    g["bad_rate"] = g["bad"] / g["total"]

    return g.reset_index()


def iv(df, feature, target, bins=10, is_categorical=False):
    """
    Single IV number for a feature — convenience wrapper over woe_table().

    If you already have a woe_table() result in hand (e.g. because you're
    about to call plot_woe), sum its iv_bin column directly instead of
    calling this — it avoids recomputing the same table twice.
    """
    return woe_table(df, feature, target, bins=bins,
                     is_categorical=is_categorical)["iv_bin"].sum()


def iv_band(iv_value):
    """
    Label an IV value against the standard Siddiqi thresholds:
    <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong,
    >0.5 "check leakage" — the variable may be a proxy for the target
    (e.g. a field only populated after default), not a suspiciously good
    predictor.
    """
    if pd.isna(iv_value):
        return "error"
    if iv_value < 0.02:
        return "useless"
    if iv_value < 0.1:
        return "weak"
    if iv_value < 0.3:
        return "medium"
    if iv_value < 0.5:
        return "strong"
    return "check leakage"


def rank_iv(df, cols, target, bins=10, is_categorical=None):
    """
    Screen many columns at once and return a ranked IV table — the pass-1
    step before spending a chart on anything (pair with eda.plot_woe for
    pass 2, on the shortlist only).

    Parameters
    ----------
    df : source dataframe
    cols : list[str] of candidate feature columns
    target : binary column (1 = bad, 0 = good)
    bins : quantile bins for numeric features
    is_categorical : None (default) auto-detects per column from dtype —
        non-numeric columns are treated as categorical, numeric columns are
        quantile-binned. Pass True/False to force every column the same way.

    Returns
    -------
    pd.DataFrame sorted by iv (desc): feature, iv, band, is_categorical,
    error. A column that fails to bin (e.g. near-constant, too few unique
    values) gets iv=NaN, band="error" and its exception message in `error`,
    and sorts to the bottom — one bad column never kills the whole screen.
    """
    rows = []
    for c in cols:
        cat = (not pd.api.types.is_numeric_dtype(df[c])) if is_categorical is None else is_categorical
        try:
            v = iv(df, c, target, bins=bins, is_categorical=cat)
            rows.append({"feature": c, "iv": v, "band": iv_band(v),
                        "is_categorical": cat, "error": None})
        except Exception as e:
            rows.append({"feature": c, "iv": np.nan, "band": "error",
                        "is_categorical": cat, "error": str(e)})

    out = pd.DataFrame(rows)
    return out.sort_values("iv", ascending=False, na_position="last").reset_index(drop=True)
