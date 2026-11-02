"""
brand_palettes — named brand colours you can pick from directly in a report's
chart code. This is the single source of truth for chart colours: deck_utils
builds its auto-themes from these, so editing a colour here changes both the
auto-styling (deck_style) and any hand-picked colours.

    from brand_palettes import HOMESTART, JACARANDA

    ax.bar(x, y, color=HOMESTART.blue)          # a specific named colour
    ax.bar(x, y, color=HOMESTART.salmon)        # accent
    ax.plot(x, y, color=HOMESTART.green)
    sns.barplot(..., palette=HOMESTART.series)  # ordered multi-series colours
    sns.heatmap(m, cmap=HOMESTART.cmap())       # sequential ramp as a colormap

Each palette exposes:
  • named brand colours  — HOMESTART.blue, .space_blue, .salmon, .green, ...
  • role aliases         — primary, primary_dk, primary_lt, secondary, tertiary
                           (these mirror the palette-*.scss slide tokens)
  • series               — an ordered list for multi-series charts
  • ramp                 — a light→dark list for graded / sequential bars
  • .cmap()              — the ramp as a matplotlib Colormap

To rebrand: edit the hexes below (and the matching palette-*.scss slide file).
"""
from types import SimpleNamespace
import matplotlib.colors as mcolors


def _pal(**kw):
    p = SimpleNamespace(**kw)
    # .cmap() returns the ramp as a continuous matplotlib colormap
    p.cmap = lambda p=p: mcolors.LinearSegmentedColormap.from_list("brand", p.ramp)
    return p


# ── HomeStart Finance — official brand-guide colours ────────────────────────
HOMESTART = _pal(
    # named brand colours (edit these to the official values if they ever change)
    blue="#0093D6",           # HomeStart Blue  (primary)
    space_blue="#002E6D",     # Space Blue      (darkest)
    saphire_blue="#0069A6",   # Saphire Blue
    light_blue="#8CC6E7",     # Light Blue
    light_grey="#DAE0E9",     # Light Grey
    finance_grey="#1D1D1B",   # Finance Grey (near-black)
    tan="#F6DFA4",            # Corporate Tan
    salmon="#FF8189",         # Corporate Salmon
    green="#80C7BC",          # HomeStart Green
    light_jade="#F3F4E9",     # Light Jade
    white="#FFFFFF",
    # role aliases (match palette-homestart.scss)
    primary="#0093D6", primary_dk="#002E6D", primary_lt="#8CC6E7",
    secondary="#80C7BC", tertiary="#FF8189",
    # chart helpers
    series=["#0093D6", "#80C7BC", "#FF8189", "#F6DFA4", "#002E6D"],
    ramp=["#DAE0E9", "#8CC6E7", "#0093D6", "#002E6D"],
)

# ── Personal — jacaranda / gold / green ─────────────────────────────────────
JACARANDA = _pal(
    jacaranda="#7A5C9E", jacaranda_dk="#3D2E4F", jacaranda_lt="#A995C0",
    gold="#C89B3C", gold_lt="#DBBE80", olive="#67724F",
    ink="#2F2F2F", page="#F7F5F0",
    primary="#7A5C9E", primary_dk="#3D2E4F", primary_lt="#A995C0",
    secondary="#C89B3C", tertiary="#67724F",
    series=["#7A5C9E", "#C89B3C", "#67724F", "#A995C0"],
    ramp=["#CDBFE0", "#8E74AE", "#5C4577", "#3D2E4F"],
)

# name -> palette (deck_utils reads this)
PALETTES = {"jacaranda": JACARANDA, "homestart": HOMESTART}
