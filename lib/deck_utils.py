"""
deck_utils — shared chart helpers for the report decks, with swappable themes.

    from deck_utils import deck_style, add_baseline, strip_bar_labels
    PAL = deck_style()                 # personal jacaranda/gold/green (external portfolio)
    PAL = deck_style(theme="homestart")# HomeStart blue/green (internal presentation)

`deck_style(theme=...)` sets brand matplotlib defaults recoloured for the theme
and RETURNS a palette you use in charts:
    ax.bar(x, y, color=PAL.primary)              # or PAL.secondary / PAL.tertiary
    default_rate_by_bin(..., grade_scale=PAL.ramp, show_base_rate=False)
    add_baseline(axes, BASE)

Keep the chart theme in step with the slide theme: theme="homestart" pairs with
`theme: [default, palette-homestart.scss, brand.scss]` in the .qmd.

Base fonts/spines/grid come from python_style_util; only the colours are swapped
here, so python_style_util itself is never modified.
"""

import re
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import python_style_util as psu
from python_style_util import set_style, C, f

# ── Theme palettes ──────────────────────────────────────────────────────────
# Single source of truth: the named colours live in brand_palettes.py. Editing a
# colour there updates both these auto-themes AND any hand-picked colours you use
# in a report (e.g. color=HOMESTART.salmon).
from brand_palettes import PALETTES
THEMES = {
    name: dict(primary=p.primary, primary_dk=p.primary_dk, primary_lt=p.primary_lt,
               secondary=p.secondary, tertiary=p.tertiary, ramp=p.ramp)
    for name, p in PALETTES.items()
}

# Module-level current theme (kept for back-compat: BRAND_RAMP importers get the
# active ramp). Prefer the palette returned by deck_style() in new decks.
BRAND_RAMP = None
PAL = None


def _apply_theme(name):
    global BRAND_RAMP, PAL
    t = THEMES[name]
    ramp = mcolors.LinearSegmentedColormap.from_list(f"{name}_ramp", t["ramp"])
    attrs = {k: v for k, v in t.items() if k != "ramp"}
    PAL = SimpleNamespace(name=name, ramp=ramp, **attrs)
    BRAND_RAMP = ramp
    return PAL


_apply_theme("jacaranda")   # sensible default on import


def deck_style(theme="jacaranda"):
    """Brand matplotlib defaults at slide scale, recoloured for `theme`.
    Returns the palette (SimpleNamespace: primary, primary_dk, primary_lt,
    secondary, tertiary, ramp, name)."""
    pal = _apply_theme(theme)
    psu.sizes.update({"title": 15, "subtitle": 13, "body": 12,
                      "small": 12, "tiny": 11, "mono": 12})
    set_style()
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130,
        "axes.titlesize": 15, "axes.titlepad": 8,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.labelsize": 12,
        "patch.facecolor": pal.primary,
        "lines.color": pal.primary,
        "axes.prop_cycle": plt.cycler(color=[pal.primary, pal.secondary,
                                             pal.tertiary, pal.primary_lt]),
    })
    return pal


def small_labels(n=9):
    psu.sizes["small"] = n


def reset_labels():
    psu.sizes["small"] = 12


def add_baseline(axes, base):
    """Reference line with its label parked on a white chip in the right margin."""
    for ax in np.ravel(np.asarray(axes)):
        ax.axhline(base, ls="--", color=C.text_muted, lw=1, zorder=1.4)
        ax.annotate(f"base {base:.1%}", xy=(0.992, base),
                    xycoords=ax.get_yaxis_transform(),
                    xytext=(0, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=10, color=C.text_muted,
                    fontfamily=f("mono"), zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none",
                              boxstyle="round,pad=0.2", alpha=0.9))


def strip_bar_labels(axes):
    """Remove per-bar % labels (keeps titles/subtitles) for dense near-equal panels."""
    for ax in np.ravel(np.asarray(axes)):
        for t in list(ax.texts):
            if re.fullmatch(r"\d+(\.\d+)?%", t.get_text().strip()):
                t.remove()
