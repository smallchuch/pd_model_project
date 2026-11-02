"""
eda — exploratory plots and quick summaries for the PD model projects.

Design rules for anything added here:
  * Take data in as an argument (df / Series) — never read a global `df`.
    That keeps every function portable across projects and testable.
  * Return the figure/axes (or the computed object); let the caller decide
    when to `plt.show()` and what to tweak per group.
  * Keep sensible keyword defaults so the common call is one line.

Contents
--------
histogram_per_group : grid of histograms for numeric columns.
overlaid_histogram_per_group : grid of histograms split/overlaid by a class variable.
mean_by_bin : grid of mean-of-y-per-bin bars — the general binned-mean engine.
default_rate_by_bin : credit-risk wrapper of mean_by_bin (binary target → default rate).
frequency_per_group : grid of frequency bars for categorical columns.
plot_woe : diverging bar chart of a woe.woe_table() result, one bar per bin.
"""

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from .style import (p, f, s, C, chart_title, binary_palette, multi_series,
                    palette, diverging_cmap)


def _compact_num(v):
    """
    Format a number compactly for axis labels: 36000 -> '36k',
    1.2e6 -> '1.2M', 0.0512 -> '0.051'. Keeps bin-edge tick labels readable
    instead of scientific notation like '3.6e+04'. Sub-thousand values
    (ratios, scores in 0–1) fall through to 2 significant figures.
    """
    a = abs(v)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if a >= div:
            body = f"{v / div:.1f}".rstrip("0").rstrip(".")
            return f"{body}{suffix}"
    return f"{v:.2g}"


def _grade_cmap(scale):
    """
    Resolve a grade-scale spec to a (Colormap, floor) pair.

    scale may be:
      * "risk" / "traffic" / "diverging" -> brand green→yellow→red
        (low value = green/safe, high = red/risk). floor 0, so the low end
        stays fully green.
      * a brand hue name (e.g. "red", "jacaranda") -> single-hue light→dark
        ramp. floor 0.25, so the lightest bar keeps some body.
      * a matplotlib Colormap, a list of colours, or a named mpl colormap.
    """
    if isinstance(scale, str) and scale.lower() in {"risk", "traffic",
                                                    "diverging", "ryg", "gyr"}:
        return diverging_cmap(), 0.0          # green(low) → yellow → red(high)
    if isinstance(scale, str) and scale in palette:
        steps = [k for k in sorted(palette[scale]) if isinstance(k, int)]
        ramp_steps = [k for k in steps if 200 <= k <= 700] or steps
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"{scale}_seq", [palette[scale][k] for k in ramp_steps])
        return cmap, 0.25
    if isinstance(scale, mcolors.Colormap):
        return scale, 0.0
    if isinstance(scale, (list, tuple)):
        return mcolors.LinearSegmentedColormap.from_list("grade", list(scale)), 0.0
    return plt.get_cmap(scale), 0.0           # any matplotlib named colormap


def _rate_ramp(values, scale):
    """
    Map values -> hex colours along `scale`, so magnitude reads as colour.
    Larger values land at the high (dark / red) end. See `_grade_cmap` for the
    accepted scale specs.
    """
    cmap, floor = _grade_cmap(scale)
    v = np.asarray(values, dtype=float)
    span = float(v.max() - v.min())
    t = (v - v.min()) / span if span else np.zeros_like(v)
    return [mcolors.to_hex(cmap(floor + (1 - floor) * ti)) for ti in t]


def _resolve_title(col, titles, i):
    """
    Pick a panel title: the auto name from the column (DAYS_BIRTH -> "Days
    Birth") unless overridden by `titles` — a dict {col: title} (by name, safe
    against dropped columns) or a list taken positionally. Falls back to the
    auto name for anything not covered.
    """
    auto = col.replace("_", " ").title()
    if titles is None:
        return auto
    if isinstance(titles, dict):
        return titles.get(col, auto)
    return titles[i] if i < len(titles) else auto


def _panel_title(ax, col, subtitle, *, titles=None, title_pad=None, i=0):
    """
    Apply the styled panel title + subtitle for one axes, honouring the shared
    `titles` / `title_pad` overrides. Used by every grid plotter so the title
    behaviour is identical across the module.
    """
    title = _resolve_title(col, titles, i)
    chart_title(ax, title, subtitle)
    if title_pad is not None:   # re-set with pad, keeping chart_title's brand style
        ax.set_title(title, loc="left", pad=title_pad,
                     fontfamily=f("heading"), fontsize=s("title"),
                     fontweight="bold", color=palette["charcoal"][500])
    return title


def histogram_per_group(data, cols, *, ncols=3, bins=50, xlabel="Value",
                        color=None, kde=False, sharex=False, suptitle=None,
                        titles=None, title_pad=None,
                        panel_w=9.0, panel_h=3.2, numeric_only=True, logx=False):
    """
    Draw a tidy grid of histograms, one panel per column.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe to pull columns from (passed in, not global).
    cols : list[str]
        Columns to plot — e.g. cols_by_theme['external_scores'].
    ncols : int, default 3
        Panels per row; rows are derived from len(cols).
    bins : int, default 50
        Histogram bins. Fewer (20-30) for spiky data, more for smooth.
    xlabel : str, default "Value"
        Shared x-axis label (the measured value, not "bin").
    color : str, optional
        Bar colour; defaults to brand jacaranda.
    kde : bool, default False
        Overlay a KDE. Good for bounded/smooth features, skip for heavy skew.
    sharex : bool, default False
        Share the x-axis across panels — only when columns share a scale.
    suptitle : str, optional
        Optional figure-level title (e.g. the group name).
    titles : list[str] | dict[str, str], optional
        Override per-panel titles (else derived from the column name). A dict
        {col: title} maps by name; a list is positional. Uncovered columns keep
        the auto title.
    title_pad : float, optional
        Extra padding (points) between each panel title and its axes, keeping
        the brand title styling.
    panel_w, panel_h : float
        Width/height in inches per panel. Total figure size scales with the
        grid: (panel_w * ncols, panel_h * nrows). Bump these to enlarge.
    numeric_only : bool, default True
        Silently drop non-numeric columns (strings/categoricals) before
        plotting, since histograms need numeric data. Set False to force all.
    logx : bool, default False
        Bin and scale the x-axis in log space. Use for heavily right-skewed
        features (income, credit) so bins are even on a log axis. Requires
        positive values — non-positive rows are ignored by seaborn.

    Returns
    -------
    (fig, axes) : so the caller can apply per-group tweaks, e.g.
        fig, axes = histogram_per_group(df, cols, sharex=True)
        for ax in axes:
            ax.set_xlim(0, 1)
        plt.show()
    """
    color = color or p("jacaranda", 300)

    if numeric_only:   # histograms need numeric data — drop string/categorical cols
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(data[c])]

    n = len(cols)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows),
                             sharex=sharex)
    axes = np.array(axes).reshape(-1)   # normalise 1x1 / 1-D / 2-D to a flat array

    for i, (ax, col) in enumerate(zip(axes, cols)):
        series = data[col].dropna()
        sns.histplot(series, bins=bins, ax=ax, color=color,
                     edgecolor="white", linewidth=0.5, alpha=0.85, kde=kde,
                     log_scale=logx)   # logx bins AND scales the x-axis in log space
        ax.axvline(series.median(), color=C.text_muted, ls="--", lw=1)   # median reference
        _panel_title(ax, col,
                     f"n={len(series):,} · {data[col].isna().mean():.0%} missing",
                     titles=titles, title_pad=title_pad, i=i)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    for ax in axes[n:]:          # blank any leftover panels in the grid
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontfamily=f("heading"), fontsize=s("display"))

    # Spacing: if constrained_layout is active (the brand set_style() enables it
    # globally), tune its pads instead of calling tight_layout() — mixing the two
    # warns and reverts the auto-spacing. Fall back to tight_layout otherwise.
    if fig.get_constrained_layout():
        try:
            fig.get_layout_engine().set(hspace=0.08, h_pad=0.12)
        except Exception:
            pass
    else:
        fig.tight_layout(h_pad=2)
    return fig, axes


def overlaid_histogram_per_group(data, cols, by, *, ncols=3, bins=50,
                                 xlabel="Value", stat="density",
                                 common_norm=False, kde=False, sharex=False,
                                 suptitle=None, titles=None, title_pad=None,
                                 panel_w=9.0, panel_h=3.2,
                                 numeric_only=True, logx=False, palette=None,
                                 **kwargs):
    """
    Draw a grid of overlaid histograms, one panel per column, with each panel
    split into overlapping distributions by the `by` class variable.

    The two-group counterpart to `histogram_per_group`: instead of one bar
    stack per panel, each level of `by` (e.g. TARGET 0/1, OWNS_CAR Y/N) gets its
    own translucent histogram drawn on the same axes, so you can compare the
    shape of a feature across classes — e.g. EXT_SOURCE_1 for defaulters vs
    non-defaulters.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe to pull columns from (passed in, not global).
    cols : list[str]
        Columns to plot — e.g. cols_by_theme['external_scores'].
    by : str
        Column whose levels split each distribution (the hue), e.g. 'TARGET'.
    ncols : int, default 3
        Panels per row; rows are derived from len(cols).
    bins : int, default 50
        Histogram bins, shared across the overlaid groups in each panel.
    xlabel : str, default "Value"
        Shared x-axis label (the measured value, not "bin").
    stat : str, default "density"
        seaborn histplot stat. "density" (default) makes imbalanced classes
        comparable — with raw "count" a rare class (e.g. defaulters) is dwarfed.
    common_norm : bool, default False
        When stat is a normalising stat, normalise each group independently
        (False) rather than across all groups, so each curve integrates to 1.
    kde : bool, default False
        Overlay a KDE per group.
    sharex : bool, default False
        Share the x-axis across panels — only when columns share a scale.
    suptitle : str, optional
        Optional figure-level title (e.g. the group name).
    titles : list[str] | dict[str, str], optional
        Override per-panel titles (else derived from the column name). A dict
        {col: title} maps by name; a list is positional. Uncovered columns keep
        the auto title.
    title_pad : float, optional
        Extra padding (points) between each panel title and its axes, keeping
        the brand title styling.
    panel_w, panel_h : float
        Width/height in inches per panel; figure scales to
        (panel_w * ncols, panel_h * nrows).
    numeric_only : bool, default True
        Silently drop non-numeric columns before plotting.
    logx : bool, default False
        Bin and scale the x-axis in log space (right-skewed features).
    palette : dict or list, optional
        Colour mapping for the `by` levels. Defaults to the brand
        binary_palette (non-default→green, default→red) for two classes,
        or multi_series() colours for more.
    **kwargs
        Extra keyword args forwarded straight to seaborn.histplot, overriding
        the styling defaults (element="step", edgecolor="white", linewidth=0.5,
        alpha=0.45). E.g. edgecolor="black", alpha=0.6, element="bars".

    Returns
    -------
    (fig, axes) : so the caller can apply per-group tweaks, same contract as
        histogram_per_group.
    """
    if numeric_only:   # histograms need numeric data — drop string/categorical cols
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(data[c])]

    hue_order = list(pd.Series(data[by].dropna().unique()).sort_values())
    if palette is None:
        palette = (binary_palette(labels=tuple(hue_order))
                   if len(hue_order) == 2
                   else dict(zip(hue_order, multi_series())))

    n = len(cols)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows),
                             sharex=sharex)
    axes = np.array(axes).reshape(-1)   # normalise 1x1 / 1-D / 2-D to a flat array

    for i, (ax, col) in enumerate(zip(axes, cols)):
        sub = data[[col, by]].dropna(subset=[col])

        plot_kw = dict(element="step", edgecolor="white", linewidth=0.5,
                       alpha=0.45)   # defaults; caller kwargs override these
        plot_kw.update(kwargs)

        sns.histplot(data=sub, x=col, hue=by, hue_order=hue_order,
                     bins=bins, ax=ax, palette=palette, stat=stat,
                     common_norm=common_norm, common_bins=True,
                     kde=kde, log_scale=logx, **plot_kw)
        _panel_title(ax, col,
                     f"split by {by} · {data[col].isna().mean():.0%} missing",
                     titles=titles, title_pad=title_pad, i=i)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    for ax in axes[n:]:          # blank any leftover panels in the grid
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontfamily=f("heading"), fontsize=s("display"))

    # Spacing: if constrained_layout is active (the brand set_style() enables it
    # globally), tune its pads instead of calling tight_layout() — mixing the two
    # warns and reverts the auto-spacing. Fall back to tight_layout otherwise.
    if fig.get_constrained_layout():
        try:
            fig.get_layout_engine().set(hspace=0.08, h_pad=0.12)
        except Exception:
            pass
    else:
        fig.tight_layout(h_pad=2)
    return fig, axes


def mean_by_bin(data, cols, y, *, ncols=3, bins=10,
                method="quantile", xlabel="Bin (low → high)",
                color=None, grade=False, grade_scale="jacaranda",
                bin_label="range", bin_num_fmt=None,
                right=True, include_lowest=False, open_top=False,
                na_as_bin=False, titles=None,
                title_pad=None, show_base_rate=True, suptitle=None,
                panel_w=9.0, panel_h=3.2, numeric_only=True,
                rotation=45, ylabel="Mean",
                value_fmt=".2g", **kwargs):
    """
    Draw a grid of mean-of-`y`-per-bin bar charts, one panel per column in `cols`.

    The general binned-mean engine: for each feature, bin it (quantile/width),
    then plot the mean of `y` within each bin — exposing the *shape* of how `y`
    moves across the feature. `default_rate_by_bin` is the credit-risk wrapper
    that feeds a binary (0/1) target (so the mean is the default rate) with
    percent framing and the risk colour scale; call `mean_by_bin` directly to
    average any numeric column, e.g. mean AMT_CREDIT across income bins.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe to pull columns from (passed in, not global).
    cols : list[str]
        Feature columns to bin — one panel each.
    y : str
        Numeric column to average within each bin. A binary 0/1 column gives a
        rate; a continuous column gives its mean.
    ncols : int, default 3
        Panels per row; rows derived from len(cols).
    bins : int, default 10
        Number of bins (deciles by default).
    method : {"quantile", "width"}, default "quantile"
        "quantile" → equal-population bins via pd.qcut (stable counts per bar).
        "width" → equal-width bins via pd.cut.
    xlabel : str, default "Bin (low → high)"
        Shared x-axis label.
    color : str, optional
        Flat bar colour used only when grade=False; defaults to brand jacaranda.
        Ignored when grade=True (the scale sets each bar's colour).
    grade : bool, default False
        Colour each bar by its value along `grade_scale`. Off by default — a
        neutral tool shouldn't imply good/bad. `default_rate_by_bin` turns it on
        with the risk scale. The value is printed on each bar regardless, so
        colour is always a redundant cue.
    grade_scale : str | Colormap | list, default "jacaranda"
        How grade=True maps value to colour:
          * a brand hue name ("jacaranda", "red", …) → single-hue light→dark ramp.
          * "risk"/"diverging" → brand green→yellow→red (low green, high red);
            only meaningful when high = worse.
          * a matplotlib Colormap, a list of hex colours, or a named mpl colormap.
    bin_label : {"range", "rank", "left", "center", "band"}, default "range"
        What the x-tick labels show. "range" → compact "36k–88k" edges;
        "rank" → 1..N (cleanest for quantile bins, where the story is the trend,
        not the cutoffs); "left"/"center" → a single compact edge or midpoint;
        "band" → clean integer bands "20–24", "25–29" for left-closed
        equal-width bins (pair with method="width", right=False).
    right, include_lowest : bool
        Forwarded to pd.cut (equal-width / explicit-edge binning only; ignored
        for quantile). right=False gives left-closed bins [a, b) — the standard
        for age/tenure bands so a value on an edge lands in exactly one band.
    open_top : bool, default False
        Relabel the final bin as "N+" (an open-ended top band, e.g. "20+").
    na_as_bin : bool, default False
        If True, append a trailing "Missing" bar showing the mean of `y` over
        rows whose feature value is NaN -- the default rate of the missing
        group, which in credit-risk data is often as predictive as any
        populated bin. Off by default: missing rows are dropped before binning,
        as before. The missing bar is drawn in a neutral muted colour and set
        slightly apart, so it reads as off-scale rather than as part of the
        ordered ramp, and it is excluded from the grade colour scale so it
        can't distort it. Rows with a non-missing feature but a missing `y`
        never contribute to any bar (mean skips NaN), missing or otherwise.
    bin_num_fmt : callable, optional
        v -> str formatter for the bin-edge numbers (every bin_label except
        "rank"). Defaults to a compact k/M/B formatter; pass your own for full
        control (e.g. lambda v: f"${v:,.0f}").
    titles : list[str] | dict[str, str], optional
        Override the per-panel titles (which otherwise come from the column
        name, e.g. DAYS_BIRTH -> "Days Birth"). A dict {col: title} is safest —
        it maps by name, so it survives numeric_only dropping a column. A list
        is taken positionally against the plotted columns. The auto title is
        kept for any column not covered.
    title_pad : float, optional
        Extra padding (points) between each panel title and its axes. Applied
        while preserving the brand title styling — use it to lift the title off
        a crowded plot without falling back to a default matplotlib title.
    show_base_rate : bool, default True
        Draw a dashed reference line at the overall mean of `y`, so you can see
        which bins sit above/below it.
    suptitle : str, optional
        Optional figure-level title.
    panel_w, panel_h : float
        Width/height in inches per panel; figure scales to
        (panel_w * ncols, panel_h * nrows).
    numeric_only : bool, default True
        Silently drop non-numeric feature columns (binning needs numeric data).
    rotation : int, default 45
        Rotation for the bin tick labels.
    ylabel : str, default "Mean"
        Y-axis label. Set it to match `y` (e.g. "Mean credit amount").
    value_fmt : str or callable, default ".2g"
        How each bar label and the base-line label are formatted. A format spec
        string is applied as ``format(v, value_fmt)`` — ".1%" for a rate,
        ",.0f" for a count, "$,.0f" for currency. Or pass a callable
        ``v -> str`` for full control.
    **kwargs
        Extra keyword args forwarded to ax.bar (e.g. alpha, linewidth).

    Returns
    -------
    (fig, axes) : same contract as the other grid plotters.
    """
    color = color or p("jacaranda", 300)

    # value_fmt: a callable v->str, or a format spec. A leading "$" is treated
    # as a literal currency prefix, since format() itself rejects "$,.0f".
    if callable(value_fmt):
        fmt = value_fmt
    else:
        _pre, _spec = (("$", value_fmt[1:]) if value_fmt.startswith("$")
                       else ("", value_fmt))
        fmt = lambda v, _pre=_pre, _spec=_spec: _pre + format(v, _spec)

    if numeric_only:   # binning needs numeric data — drop string/categorical cols
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(data[c])]

    n = len(cols)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows))
    axes = np.array(axes).reshape(-1)

    base_rate = data[y].mean()

    for i, (ax, col) in enumerate(zip(axes, cols)):
        sub = data[[col, y]].dropna(subset=[col])

        if method == "quantile":
            sub = sub.assign(_bin=pd.qcut(sub[col], bins, duplicates="drop"))
        else:   # equal-width / explicit edges — supports right & include_lowest
            sub = sub.assign(_bin=pd.cut(sub[col], bins, right=right,
                                         include_lowest=include_lowest,
                                         duplicates="drop"))

        grp = sub.groupby("_bin", observed=True)[y]
        rate = grp.mean()
        counts = grp.size()

        # x-tick labels: pick what to show, format edge numbers compactly
        nfmt = bin_num_fmt or _compact_num
        if bin_label == "rank":
            labels = [str(i + 1) for i in range(len(rate))]
        elif bin_label == "left":
            labels = [nfmt(iv.left) for iv in rate.index]
        elif bin_label == "center":
            labels = [nfmt((iv.left + iv.right) / 2) for iv in rate.index]
        elif bin_label == "band":
            # Integer bands: a left-closed bin [a, b) covers whole values a..b-1,
            # shown as "a–(b-1)" — e.g. edges 20,25,30 -> "20–24", "25–29"; a
            # width-1 band [a, a+1) collapses to just "a". Pair with
            # method="width", right=False on a whole-number feature.
            def _band(iv):
                lo, hi = iv.left, iv.right - 1
                return f"{nfmt(lo)}" if hi <= lo else f"{nfmt(lo)}–{nfmt(hi)}"
            labels = [_band(iv) for iv in rate.index]
        else:   # "range"
            labels = [f"{nfmt(iv.left)}–{nfmt(iv.right)}" for iv in rate.index]

        if open_top and len(labels):        # open-ended top band -> "N+"
            labels[-1] = f"{nfmt(rate.index[-1].left)}+" 

        # grade bars by rate (green→red risk scale) unless a flat colour is asked
        bar_color = _rate_ramp(rate.values, grade_scale) if grade else color
        bar_kw = dict(alpha=0.85, edgecolor="white", linewidth=0.5)
        bar_kw.update(kwargs)
        bars = ax.bar(range(len(rate)), rate.values, color=bar_color, **bar_kw)

        ax.bar_label(bars, labels=[fmt(v) for v in rate.values],
                     padding=3, fontsize=s("small"), fontfamily=f("mono"))

        # optional trailing "Missing" bar: the mean of y over rows whose feature
        # is NaN. Kept off the ordered ramp (its own muted colour + a small gap)
        # so it reads as a separate, off-scale group rather than the next bin.
        tick_pos = list(range(len(rate)))
        if na_as_bin:
            miss_mask = data[col].isna()
            miss_rate = (data.loc[miss_mask, y].mean()
                         if miss_mask.any() else np.nan)
            if pd.notna(miss_rate):
                miss_x = len(rate) + 0.6
                mbar = ax.bar([miss_x], [miss_rate], color=C.text_muted, **bar_kw)
                ax.bar_label(mbar, labels=[fmt(miss_rate)], padding=3,
                             fontsize=s("small"), fontfamily=f("mono"))
                labels = labels + ["Missing"]
                tick_pos.append(miss_x)

        if show_base_rate:   # overall target mean reference
            ax.axhline(base_rate, color=C.text_muted, ls="--", lw=1)
            ax.text(len(rate) - 0.5, base_rate, f" base {fmt(base_rate)}",
                    va="bottom", ha="right", fontsize=s("small"),
                    color=C.text_muted, fontfamily=f("mono"))

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(labels, rotation=rotation, ha="right",
                           fontsize=s("small"))
        _panel_title(ax, col,
                     f"{method} bins · n={counts.sum():,} · "
                     f"{data[col].isna().mean():.0%} missing",
                     titles=titles, title_pad=title_pad, i=i)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)   # headroom for value labels

    for ax in axes[n:]:      # blank any leftover panels
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontfamily=f("heading"), fontsize=s("display"))

    # Spacing: if constrained_layout is active (the brand set_style() enables it
    # globally), tune its pads instead of calling tight_layout() — mixing the two
    # warns and reverts the auto-spacing. Fall back to tight_layout otherwise.
    if fig.get_constrained_layout():
        try:
            fig.get_layout_engine().set(hspace=0.08, h_pad=0.12)
        except Exception:
            pass
    else:
        fig.tight_layout(h_pad=2)
    return fig, axes


def default_rate_by_bin(data, cols, target, **kwargs):
    """
    Default-rate-per-bin bars — the credit-risk framing of `mean_by_bin`.

    Averages a binary (0/1) `target` per bin (so the mean *is* the default rate)
    and applies the risk-plot defaults: percent labels, a "Default rate" y-axis,
    and the green→yellow→red risk grade. Every other knob (bins, method,
    bin_label, grade_scale, panel size, …) forwards straight to `mean_by_bin`,
    and anything you pass explicitly overrides these defaults — e.g.
    grade_scale="red" for a colour-blind-safe grade, grade=False for flat bars,
    or na_as_bin=True to add a trailing bar for the missing group's default rate.

    This is the workhorse bivariate risk plot: the *shape* of the rate across
    bins (monotonic / threshold / flat) is what you read before scorecard
    binning or WOE encoding.
    """
    kwargs.setdefault("ylabel", "Default rate")
    kwargs.setdefault("value_fmt", ".1%")
    kwargs.setdefault("grade", True)
    kwargs.setdefault("grade_scale", "risk")
    return mean_by_bin(data, cols, target, **kwargs)


def frequency_per_group(data, cols, *, ncols=2, top_n=15, dropna=False,
                        pct=False, color=None, suptitle=None,
                        titles=None, title_pad=None,
                        panel_w=7.0, panel_h=4.0, categorical_only=True):
    """
    Draw a grid of frequency bar charts, one panel per categorical column.

    The string/categorical counterpart to `histogram_per_group`: horizontal
    bars sorted by frequency, with the long tail pooled into an "Other" bar and
    missing values shown as their own bar so nothing is hidden.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe to pull columns from (passed in, not global).
    cols : list[str]
        Categorical columns to plot — e.g. cols_by_type['categorical'].
    ncols : int, default 2
        Panels per row; rows derived from len(cols).
    top_n : int or None, default 15
        Show only the top-N levels by frequency; the remainder are pooled into
        a single "Other (k more)" bar so the total is preserved. None = show all.
    dropna : bool, default False
        If False, missing values appear as their own "Missing" bar.
    pct : bool, default False
        Plot share of rows instead of raw counts.
    color : str, optional
        Bar colour; defaults to brand jacaranda.
    suptitle : str, optional
        Optional figure-level title (e.g. the group name).
    titles : list[str] | dict[str, str], optional
        Override per-panel titles (else derived from the column name). A dict
        {col: title} maps by name; a list is positional. Uncovered columns keep
        the auto title.
    title_pad : float, optional
        Extra padding (points) between each panel title and its axes, keeping
        the brand title styling.
    panel_w, panel_h : float
        Width/height in inches per panel. Figure scales to
        (panel_w * ncols, panel_h * nrows).
    categorical_only : bool, default True
        Silently drop numeric columns, so you can pass a mixed theme group
        (e.g. cols_by_theme['loan_financials']) and only its string/categorical
        columns are charted. Mirror of numeric_only in histogram_per_group.
        Set False to force numeric columns through as discrete categories.

    Returns
    -------
    (fig, axes) : for per-panel tweaks after the call, same as histogram_per_group.
    """
    color = color or p("jacaranda", 300)

    if categorical_only:   # bar charts are for categoricals — drop numeric cols
        cols = [c for c in cols if not pd.api.types.is_numeric_dtype(data[c])]

    n = len(cols)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (ax, col) in enumerate(zip(axes, cols)):
        counts = data[col].value_counts(dropna=dropna)
        total = counts.sum()

        # pool the long tail beyond top_n into a single "Other" bar
        if top_n is not None and len(counts) > top_n:
            other = counts.iloc[top_n:].sum()
            counts = counts.iloc[:top_n]
            counts[f"Other ({data[col].nunique() - top_n} more)"] = other

        if pct:
            counts = counts / total

        counts = counts.sort_values()   # ascending → largest bar on top in barh
        labels = ["Missing" if pd.isna(i) else str(i) for i in counts.index]

        bars = ax.barh(labels, counts.values, color=color, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
        fmt = "{:.1%}".format if pct else "{:,.0f}".format
        ax.bar_label(bars, labels=[fmt(v) for v in counts.values],
                     padding=3, fontsize=s("small"), fontfamily=f("mono"))

        _panel_title(ax, col,
                     f"{data[col].nunique():,} levels · {data[col].isna().mean():.0%} missing",
                     titles=titles, title_pad=title_pad, i=i)
        ax.set_xlabel("Share" if pct else "Count")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.12)   # headroom for the value labels

    for ax in axes[n:]:      # blank any leftover panels
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontfamily=f("heading"), fontsize=s("display"))

    # Spacing: if constrained_layout is active (the brand set_style() enables it
    # globally), tune its pads instead of calling tight_layout() — mixing the two
    # warns and reverts the auto-spacing. Fall back to tight_layout otherwise.
    if fig.get_constrained_layout():
        try:
            fig.get_layout_engine().set(hspace=0.08, h_pad=0.12)
        except Exception:
            pass
    else:
        fig.tight_layout(h_pad=2)
    return fig, axes


def plot_woe(woe_table, feature, iv=None, *, ax=None, panel_w=7.0,
            panel_h=None, title_pad=None):
    """
    Horizontal diverging bar chart of a WOE table, one bar per bin — the
    plotting counterpart to woe.woe_table(). Bin order is preserved from the
    table (first row plots on top), and colour follows sign: safer than
    average (positive WOE) vs riskier than average (negative WOE).

    Parameters
    ----------
    woe_table : pd.DataFrame
        Output of woe.woe_table() — needs 'bucket' and 'woe' columns.
    feature : str
        Feature name, used for the panel title.
    iv : float, optional
        The feature's IV (e.g. from woe.iv(), or woe_table['iv_bin'].sum()
        if you already have the table). Shown in the subtitle if given.
    ax : matplotlib Axes, optional
        Draw on an existing axes — e.g. one panel you're assembling into a
        grid by hand. Creates its own single-panel figure if None.
    panel_w : float, default 7.0
        Figure width in inches when ax is None.
    panel_h : float, optional
        Figure height in inches when ax is None; defaults to scaling with
        the number of bins so tall tables don't cramp.
    title_pad : float, optional
        Extra padding (points) between the title and the axes.

    Returns
    -------
    ax : the matplotlib Axes the chart was drawn on.
    """
    t = woe_table.iloc[::-1].reset_index(drop=True)   # first bin plots on top
    labels = t["bucket"].astype(str)
    values = t["woe"]

    pal = binary_palette(labels=("safe", "risk"))   # positional: 1st->green, 2nd->red
    colors = [pal["risk"] if v < 0 else pal["safe"] for v in values]

    if ax is None:
        panel_h = panel_h or (0.55 * len(t) + 1.2)
        _, ax = plt.subplots(figsize=(panel_w, panel_h))

    bars = ax.barh(labels, values, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5, height=0.6, zorder=3)

    # Value labels default to sitting INSIDE each bar near its outer end
    # (white text), so a long bar never collides with the y-axis bin labels
    # on the same side. But bin widths here can span orders of magnitude
    # (a normal bin next to a rare "Missing" outlier), so a fixed inset
    # sized off the largest bar would overflow tiny bars and vanish against
    # the white background. Instead: draw inside first, then measure each
    # label against its own bar in pixel space and flip any that don't
    # actually fit to sit just outside the bar in dark ink.
    ax.figure.canvas.draw()   # materialise a renderer so extents are real
    inset = 0.02 * (max(values.max(), 0) - min(values.min(), 0) or 1e-6)
    labels_drawn = []
    for bar, v in zip(bars, values):
        x_in = bar.get_width() - inset if v >= 0 else bar.get_width() + inset
        ha = "right" if v >= 0 else "left"
        txt = ax.text(x_in, bar.get_y() + bar.get_height() / 2, f"{v:+.2f}",
                      va="center", ha=ha, fontsize=s("small"), fontfamily=f("mono"),
                      color="white", fontweight="bold", zorder=4)
        labels_drawn.append((txt, bar, v))

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    for txt, bar, v in labels_drawn:
        fits = txt.get_window_extent(renderer).width + 4 <= bar.get_window_extent(renderer).width
        if not fits:   # doesn't fit inside -> flip to just outside, dark ink
            txt.set_color(C.text)
            x_out = bar.get_width() + inset if v >= 0 else bar.get_width() - inset
            txt.set_position((x_out, bar.get_y() + bar.get_height() / 2))
            txt.set_ha("left" if v >= 0 else "right")

    ax.axvline(0, color=C.text_muted, lw=1)

    subtitle = f"{len(t)} bins" if iv is None else f"{len(t)} bins · IV = {iv:.3f}"
    _panel_title(ax, feature, subtitle, title_pad=title_pad)

    ax.set_xlabel("WOE   (← higher risk        lower risk →)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.08)

    return ax
