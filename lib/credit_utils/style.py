"""
style — brand design system (colours, fonts, chart styling).

Thin re-export layer over `python_style_util` so the whole toolkit is reachable
through one namespace, e.g.

    from credit_utils.style import set_style, p, f, s, C

The single source of truth for the palette/fonts still lives in
`python_style_util.py` — edit the hexes, fonts and sizes there and every module
picks the change up automatically.
"""

from python_style_util import (  # noqa: F401  (re-exported on purpose)
    palette, fonts, sizes,
    p, f, s,
    C, Shade, Size,
    set_style,
    gradient, traffic_light, diverging, diverging_cmap,
    risk_cmap, RISK_LOW, RISK_HIGH,
    multi_series, binary_palette,
    chart_title, annotate,
    style_df,
    show_palette, show_fonts, show_tokens, show_colormaps, show_style_guide,
)
