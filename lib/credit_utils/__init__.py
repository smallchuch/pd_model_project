"""
credit_utils — shared toolkit for the Credit Risk PD model projects.

Grow this deliberately: add a function only once you've written it twice in a
notebook and know it works. That way you always know what's in here and how to
use it.

Modules
-------
style    : brand colours, fonts, chart styling   (re-exports python_style_util)
eda      : exploratory plots & summaries          -> histogram_per_group, plot_woe
quality  : data-quality & cleaning helpers        (empty — add as we go)
woe      : Weight-of-Evidence / Information Value  -> woe_table, iv, iv_band, rank_iv
stats    : hypothesis tests for driver checks      (empty — add in Section 4)

Typical use in a notebook
--------------------------
    %load_ext autoreload
    %autoreload 2

    from credit_utils.style import set_style, p, f, s
    from credit_utils.eda import histogram_per_group, plot_woe
    from credit_utils.woe import woe_table, iv, iv_band, rank_iv

    set_style()
    fig, axes = histogram_per_group(df, cols_by_theme['external_scores'],
                                    xlabel='Score (0-1)', sharex=True, kde=True)

    # WOE / IV screening (Section 5):
    #   1. rank_iv over every candidate column -> keep/drop by band
    #   2. plot_woe on the shortlist only, to check monotonicity
    ranked = rank_iv(df, candidate_cols, target='TARGET')
    shortlist = ranked.loc[ranked['iv'] >= 0.02, 'feature']
    for feat in shortlist:
        is_cat = ranked.set_index('feature').loc[feat, 'is_categorical']
        t = woe_table(df, feat, 'TARGET', is_categorical=is_cat)
        plot_woe(t, feat, iv=t['iv_bin'].sum())
"""

from .eda import histogram_per_group, frequency_per_group, plot_woe
from .woe import woe_table, iv, iv_band, rank_iv

__all__ = ["histogram_per_group", "frequency_per_group", "plot_woe",
          "woe_table", "iv", "iv_band", "rank_iv"]
