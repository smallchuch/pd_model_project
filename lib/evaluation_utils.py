# ══════════════════════════════════════════════════════════════════════════════
# evaluation_utils.py
# Credit risk model evaluation — AUC, Gini, KS, confusion matrix, scorecard
# Drop into Core Resources/Scripts and import into any notebook
# ══════════════════════════════════════════════════════════════════════════════
#
# USAGE:
#   from evaluation_utils import evaluate_model, compare_models
#
#   results = evaluate_model(y_true, y_probs, model_name='Logistic Regression')
#   compare_models(results_a, results_b)
#
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

# ── Try to load design system — falls back to defaults if not found ───────────
try:
    sys.path.append(os.path.join(os.path.abspath('..'), 'Core Resources'))
    import python_style_util as psu
    psu.set_style()
    C_PRIMARY    = psu.p('purple', 500)
    C_SECONDARY  = psu.p('cyan',   600)
    C_DANGER     = psu.p('red',    500)
    C_SAFE       = psu.p('green',  500)
    C_WATCH      = psu.p('yellow', 600)
    C_NEUTRAL    = psu.p('grey',   400)
    C_LIGHT      = psu.p('grey',   200)
    C_TEXT       = psu.p('grey',   800)
    C_SUBTEXT    = psu.p('grey',   500)
    C_RED_LIGHT  = psu.p('red',    100)
    C_GREEN_LIGHT= psu.p('green',  100)
    C_ORANGE     = psu.p('orange', 500)
    C_ORANGE_LIGHT = psu.p('orange', 100)
    FONT_HEAD    = psu.fonts['heading']
    FONT_BODY    = psu.fonts['body']
    FONT_MONO    = psu.fonts['mono']
    S_TITLE      = psu.sizes['title']
    S_BODY       = psu.sizes['body']
    S_SMALL      = psu.sizes['small']
    S_MONO       = psu.sizes['mono']
except Exception:
    C_PRIMARY    = '#732DD9'
    C_SECONDARY  = '#0DBDCE'
    C_DANGER     = '#E53935'
    C_SAFE       = '#91C46C'
    C_WATCH      = '#CEAA40'
    C_NEUTRAL    = '#ABABAB'
    C_LIGHT      = '#EBEBEB'
    C_TEXT       = '#262626'
    C_SUBTEXT    = '#7F7F7F'
    C_RED_LIGHT  = '#FCEBEA'
    C_GREEN_LIGHT= '#F4F9F0'
    C_ORANGE     = '#FF5722'
    C_ORANGE_LIGHT = '#FFEEE8'
    FONT_HEAD    = 'Arial'
    FONT_BODY    = 'Arial'
    FONT_MONO    = 'Courier New'
    S_TITLE      = 14
    S_BODY       = 10
    S_SMALL      = 9
    S_MONO       = 9


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR SCALE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _auc_colour(auc):
    if auc >= 0.85: return C_SAFE,  C_GREEN_LIGHT,  'Excellent'
    if auc >= 0.75: return C_SAFE,  C_GREEN_LIGHT,  'Good'
    if auc >= 0.65: return C_WATCH, C_ORANGE_LIGHT, 'Fair'
    return C_DANGER, C_RED_LIGHT, 'Poor'

def _gini_colour(gini):
    if gini >= 0.70: return C_SAFE,  C_GREEN_LIGHT,  'Excellent'
    if gini >= 0.50: return C_SAFE,  C_GREEN_LIGHT,  'Good'
    if gini >= 0.30: return C_WATCH, C_ORANGE_LIGHT, 'Fair'
    return C_DANGER, C_RED_LIGHT, 'Poor'

def _ks_colour(ks):
    if ks >= 0.55: return C_SAFE,  C_GREEN_LIGHT,  'Excellent'
    if ks >= 0.35: return C_SAFE,  C_GREEN_LIGHT,  'Good'
    if ks >= 0.20: return C_WATCH, C_ORANGE_LIGHT, 'Fair'
    return C_DANGER, C_RED_LIGHT, 'Poor'


# ══════════════════════════════════════════════════════════════════════════════
# METRIC BAR HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _draw_metric_bar(ax, value, vmin, vmax, colour, label, rating, note=''):
    """Draw a single horizontal metric bar with colour coding."""
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Background track
    ax.barh(0.5, vmax - vmin, left=vmin, height=0.22,
            color=C_LIGHT, zorder=1)

    # Filled bar
    ax.barh(0.5, value - vmin, left=vmin, height=0.22,
            color=colour, zorder=2)

    # Value label
    ax.text(value, 0.5, f'  {value:.4f}',
            va='center', ha='left',
            fontfamily=FONT_MONO, fontsize=S_MONO + 1,
            fontweight='bold', color=colour, zorder=3)

    # Rating badge
    ax.text(vmax, 0.83, rating,
            va='center', ha='right',
            fontfamily=FONT_BODY, fontsize=S_SMALL,
            color=colour, style='italic')

    # Metric name
    ax.text(vmin, 0.83, label,
            va='center', ha='left',
            fontfamily=FONT_BODY, fontsize=S_BODY,
            fontweight='bold', color=C_TEXT)

    # Range note
    if note:
        ax.text(vmin, 0.15, note,
                va='center', ha='left',
                fontfamily=FONT_BODY, fontsize=S_SMALL - 1,
                color=C_SUBTEXT)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(y_true, y_probs, model_name='Model', threshold=0.5, color=None):
    """
    Full credit risk evaluation scorecard.
    Produces a visual scorecard with:
      - AUC, Gini, KS metric bars (colour coded red/amber/green)
      - ROC curve
      - Confusion matrix (colour coded)
      - Precision, Recall, F1, Accuracy

    Parameters:
    ───────────
    y_true      : array-like — actual binary labels (0/1)
    y_probs     : array-like — predicted probabilities of default
    model_name  : str — label for the scorecard
    threshold   : float — decision threshold (default 0.5)
    color       : str or None — hex colour for this model's ROC curve line

    Returns:
    ────────
    dict with keys: auc, gini, ks, precision, recall, f1, accuracy,
                    fpr, tpr, thresholds, y_pred
    """
    if color is None:
        color = C_PRIMARY

    y_true  = np.array(y_true)
    y_probs = np.array(y_probs)

    # ── Ranking metrics ───────────────────────────────────────────────────────
    auc  = roc_auc_score(y_true, y_probs)
    gini = 2 * auc - 1
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    ks   = float(np.max(tpr - fpr))
    ks_threshold_idx = np.argmax(tpr - fpr)
    ks_threshold = thresholds[ks_threshold_idx]

    # ── Threshold-based metrics ───────────────────────────────────────────────
    y_pred    = (y_probs >= threshold).astype(int)
    cm        = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (tp + tn) / len(y_true)

    # ── Colour coding ─────────────────────────────────────────────────────────
    auc_col,  auc_bg,  auc_rating  = _auc_colour(auc)
    gini_col, gini_bg, gini_rating = _gini_colour(gini)
    ks_col,   ks_bg,   ks_rating   = _ks_colour(ks)

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE LAYOUT
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.06, right=0.97,
        top=0.88, bottom=0.06
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.06, 0.95, model_name,
             fontfamily=FONT_HEAD, fontsize=S_TITLE + 2,
             fontweight='bold', color=C_TEXT, va='top')
    fig.text(0.06, 0.915, f'Threshold: {threshold}   ·   n = {len(y_true):,}   ·   Default rate: {y_true.mean():.1%}',
             fontfamily=FONT_BODY, fontsize=S_SMALL,
             color=C_SUBTEXT, va='top')

    # ── Metric bars ───────────────────────────────────────────────────────────
    ax_auc  = fig.add_subplot(gs[0, :2])
    ax_gini = fig.add_subplot(gs[1, :2])
    ax_ks   = fig.add_subplot(gs[2, :2])

    _draw_metric_bar(ax_auc,  auc,  0.5, 1.0, auc_col,  'AUC-ROC',         auc_rating,  '0.5 = random  ·  1.0 = perfect')
    _draw_metric_bar(ax_gini, gini, 0.0, 1.0, gini_col, 'Gini coefficient', gini_rating, '0 = random  ·  1.0 = perfect  ·  Gini = 2×AUC − 1')
    _draw_metric_bar(ax_ks,   ks,   0.0, 1.0, ks_col,   'KS statistic',    ks_rating,   '0 = no separation  ·  >0.4 considered strong in credit')

    # ── ROC curve ─────────────────────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[:2, 2])
    ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'AUC = {auc:.4f}')
    ax_roc.plot([0, 1], [0, 1], '--', color=C_NEUTRAL, lw=1, label='Random')

    # KS point
    ax_roc.plot(fpr[ks_threshold_idx], tpr[ks_threshold_idx],
                'o', color=ks_col, markersize=7, zorder=5,
                label=f'KS = {ks:.4f}')
    ax_roc.axvline(fpr[ks_threshold_idx], color=ks_col,
                   linestyle=':', lw=1, alpha=0.5)
    ax_roc.axhline(tpr[ks_threshold_idx], color=ks_col,
                   linestyle=':', lw=1, alpha=0.5)

    ax_roc.set_xlabel('False Positive Rate', fontfamily=FONT_BODY, fontsize=S_SMALL)
    ax_roc.set_ylabel('True Positive Rate',  fontfamily=FONT_BODY, fontsize=S_SMALL)
    ax_roc.set_title('ROC Curve',
                     fontfamily=FONT_HEAD, fontsize=S_BODY,
                     fontweight='bold', color=C_TEXT, loc='left')
    ax_roc.legend(fontsize=S_SMALL - 1, frameon=False)
    ax_roc.spines['top'].set_visible(False)
    ax_roc.spines['right'].set_visible(False)
    ax_roc.grid(color=C_LIGHT, linestyle='--', linewidth=0.8)
    ax_roc.set_axisbelow(True)
    ax_roc.tick_params(labelsize=S_SMALL)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    ax_cm = fig.add_subplot(gs[2, 2])
    ax_cm.axis('off')
    ax_cm.set_title(f'Confusion Matrix @ {threshold}',
                    fontfamily=FONT_HEAD, fontsize=S_BODY,
                    fontweight='bold', color=C_TEXT, loc='left', pad=8)

    cm_data = [
        [tp, fn],
        [fp, tn]
    ]
    cm_labels = [['TP', 'FN'], ['FP', 'TN']]
    cm_descs  = [['Caught defaulters', 'Missed defaulters'],
                 ['False alarms', 'Correct clears']]
    cm_colors = [[C_GREEN_LIGHT, C_ORANGE_LIGHT],
                 [C_RED_LIGHT,   C_GREEN_LIGHT]]
    cm_text_colors = [
        [psu.p('green', 700) if 'psu' in dir() else '#5B7944', psu.p('orange', 700) if 'psu' in dir() else '#9D3818'],
        [psu.p('red',   700) if 'psu' in dir() else '#8D2623', psu.p('green',  700) if 'psu' in dir() else '#5B7944'],
    ]

    cell_w, cell_h = 0.45, 0.38
    starts_x = [0.02, 0.50]
    starts_y = [0.55, 0.10]

    for r in range(2):
        for c in range(2):
            x, y = starts_x[c], starts_y[r]
            tc   = cm_text_colors[r][c]
            rect = mpatches.FancyBboxPatch(
                (x, y), cell_w, cell_h,
                boxstyle='round,pad=0.02',
                facecolor=cm_colors[r][c],
                edgecolor='white', linewidth=2,
                transform=ax_cm.transAxes, clip_on=False
            )
            ax_cm.add_patch(rect)

            ax_cm.text(x + cell_w/2, y + cell_h*0.72,
                       cm_labels[r][c],
                       ha='center', va='center',
                       fontfamily=FONT_BODY, fontsize=S_SMALL,
                       color=tc, transform=ax_cm.transAxes)

            ax_cm.text(x + cell_w/2, y + cell_h*0.42,
                       f'{cm_data[r][c]:,}',
                       ha='center', va='center',
                       fontfamily=FONT_MONO, fontsize=S_TITLE,
                       fontweight='bold', color=tc,
                       transform=ax_cm.transAxes)

            ax_cm.text(x + cell_w/2, y + cell_h*0.15,
                       cm_descs[r][c],
                       ha='center', va='center',
                       fontfamily=FONT_BODY, fontsize=S_SMALL - 1,
                       color=tc, transform=ax_cm.transAxes)

    # Row/col labels
    ax_cm.text(0.25, 0.98, 'Predicted Default',
               ha='center', fontfamily=FONT_BODY, fontsize=S_SMALL,
               color=C_SUBTEXT, transform=ax_cm.transAxes)
    ax_cm.text(0.73, 0.98, 'Predicted No Default',
               ha='center', fontfamily=FONT_BODY, fontsize=S_SMALL,
               color=C_SUBTEXT, transform=ax_cm.transAxes)
    ax_cm.text(-0.02, 0.73, 'Actual\nDefault',
               ha='right', va='center', fontfamily=FONT_BODY, fontsize=S_SMALL,
               color=C_SUBTEXT, transform=ax_cm.transAxes)
    ax_cm.text(-0.02, 0.28, 'Actual\nNo Default',
               ha='right', va='center', fontfamily=FONT_BODY, fontsize=S_SMALL,
               color=C_SUBTEXT, transform=ax_cm.transAxes)

    # ── Threshold metrics summary ─────────────────────────────────────────────
    metrics_summary = {
        'Precision':  precision,
        'Recall':     recall,
        'F1 score':   f1,
        'Accuracy':   accuracy,
    }

    fig.text(0.06, 0.055,
             '   '.join([f'{k}: {v:.1%}' for k, v in metrics_summary.items()]),
             fontfamily=FONT_MONO, fontsize=S_MONO,
             color=C_SUBTEXT, va='bottom')

    plt.show()

    return {
        'model_name': model_name,
        'auc':        auc,
        'gini':       gini,
        'ks':         ks,
        'precision':  precision,
        'recall':     recall,
        'f1':         f1,
        'accuracy':   accuracy,
        'fpr':        fpr,
        'tpr':        tpr,
        'thresholds': thresholds,
        'y_pred':     y_pred,
        'color':      color,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_models(*results):
    """
    Side-by-side comparison of multiple model evaluation results.
    Pass the dict returned by evaluate_model() for each model.

    Usage:
        lr_results   = evaluate_model(y_true, lr_probs,   'Logistic Regression')
        lgbm_results = evaluate_model(y_true, lgbm_probs, 'LightGBM')
        compare_models(lr_results, lgbm_results)
    """
    if len(results) < 2:
        print("Pass at least 2 model results to compare.")
        return

    colors = [C_PRIMARY, C_SECONDARY, C_ORANGE, C_SAFE]
    n = len(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    # ── ROC curves overlay ────────────────────────────────────────────────────
    ax_roc = axes[0]
    for i, r in enumerate(results):
        col = r.get('color', colors[i % len(colors)])
        ax_roc.plot(r['fpr'], r['tpr'], color=col, lw=2,
                    label=f"{r['model_name']}  AUC={r['auc']:.4f}  Gini={r['gini']:.4f}  KS={r['ks']:.4f}")

    ax_roc.plot([0,1],[0,1],'--', color=C_NEUTRAL, lw=1, label='Random')
    ax_roc.set_xlabel('False Positive Rate', fontfamily=FONT_BODY, fontsize=S_SMALL)
    ax_roc.set_ylabel('True Positive Rate',  fontfamily=FONT_BODY, fontsize=S_SMALL)
    ax_roc.set_title('ROC Curve Comparison',
                     fontfamily=FONT_HEAD, fontsize=S_BODY,
                     fontweight='bold', color=C_TEXT, loc='left')
    ax_roc.legend(fontsize=S_SMALL - 1, frameon=False)
    ax_roc.spines['top'].set_visible(False)
    ax_roc.spines['right'].set_visible(False)
    ax_roc.grid(color=C_LIGHT, linestyle='--', linewidth=0.8)
    ax_roc.set_axisbelow(True)
    ax_roc.tick_params(labelsize=S_SMALL)

    # ── Metric bar comparison ─────────────────────────────────────────────────
    ax_bar = axes[1]
    ax_bar.axis('off')
    ax_bar.set_title('Metric Comparison',
                     fontfamily=FONT_HEAD, fontsize=S_BODY,
                     fontweight='bold', color=C_TEXT, loc='left', pad=10)

    metric_keys = [
        ('auc',       'AUC-ROC',          0.5, 1.0, _auc_colour),
        ('gini',      'Gini',             0.0, 1.0, _gini_colour),
        ('ks',        'KS statistic',     0.0, 1.0, _ks_colour),
        ('precision', 'Precision',        0.0, 1.0, None),
        ('recall',    'Recall',           0.0, 1.0, None),
        ('f1',        'F1 score',         0.0, 1.0, None),
    ]

    row_h   = 1.0 / (len(metric_keys) + 1)
    bar_w   = 0.55
    bar_x   = 0.30
    label_x = 0.02

    for i, (key, label, vmin, vmax, col_fn) in enumerate(metric_keys):
        y = 1.0 - (i + 1.2) * row_h

        # Metric label
        ax_bar.text(label_x, y + row_h * 0.5, label,
                    va='center', ha='left',
                    fontfamily=FONT_BODY, fontsize=S_SMALL,
                    color=C_TEXT, transform=ax_bar.transAxes)

        for j, r in enumerate(results):
            val = r[key]
            col = col_fn(val)[0] if col_fn else colors[j % len(colors)]
            pct = (val - vmin) / (vmax - vmin)
            w   = pct * bar_w

            bar_y = y + row_h * (0.6 - j * 0.35)
            rect = mpatches.Rectangle(
                (bar_x, bar_y), w, row_h * 0.28,
                transform=ax_bar.transAxes,
                facecolor=col, alpha=0.85 if j == 0 else 0.55,
                clip_on=False
            )
            ax_bar.add_patch(rect)

            ax_bar.text(bar_x + w + 0.01, bar_y + row_h * 0.14,
                        f'{val:.4f}',
                        va='center', ha='left',
                        fontfamily=FONT_MONO, fontsize=S_MONO,
                        color=col, transform=ax_bar.transAxes)

    # Legend
    for j, r in enumerate(results):
        col = r.get('color', colors[j % len(colors)])
        rect = mpatches.Rectangle(
            (bar_x + j * 0.25, 0.02), 0.03, 0.025,
            transform=ax_bar.transAxes,
            facecolor=col, clip_on=False
        )
        ax_bar.add_patch(rect)
        ax_bar.text(bar_x + j * 0.25 + 0.04, 0.033,
                    r['model_name'],
                    va='center', ha='left',
                    fontfamily=FONT_BODY, fontsize=S_SMALL,
                    color=C_SUBTEXT, transform=ax_bar.transAxes)

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def threshold_sweep(y_true, y_probs, model_name='Model', color=None):
    """
    Plot precision, recall and F1 across all thresholds.
    Useful for finding the optimal threshold for your specific use case.

    Usage:
        threshold_sweep(y_true, y_probs, 'LightGBM')
    """
    if color is None:
        color = C_PRIMARY

    thresholds = np.linspace(0.01, 0.99, 200)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    best_f1_idx = np.argmax(f1s)
    best_t      = thresholds[best_f1_idx]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('white')

    ax.plot(thresholds, precisions, color=C_PRIMARY,   lw=2, label='Precision')
    ax.plot(thresholds, recalls,    color=C_SECONDARY, lw=2, label='Recall')
    ax.plot(thresholds, f1s,        color=C_ORANGE,    lw=2, label='F1 score')

    ax.axvline(best_t, color=C_NEUTRAL, linestyle='--', lw=1.5,
               label=f'Best F1 threshold = {best_t:.3f}')
    ax.axvline(0.5, color=C_LIGHT, linestyle=':', lw=1,
               label='Default threshold = 0.5')

    ax.set_xlabel('Threshold',      fontfamily=FONT_BODY, fontsize=S_BODY)
    ax.set_ylabel('Score',          fontfamily=FONT_BODY, fontsize=S_BODY)
    ax.set_title(f'Threshold Sweep — {model_name}',
                 fontfamily=FONT_HEAD, fontsize=S_TITLE,
                 fontweight='bold', color=C_TEXT, loc='left')
    ax.legend(fontsize=S_SMALL, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color=C_LIGHT, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=S_SMALL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.show()

    print(f"Best F1 threshold: {best_t:.3f}  (F1={f1s[best_f1_idx]:.4f}  "
          f"Precision={precisions[best_f1_idx]:.4f}  Recall={recalls[best_f1_idx]:.4f})")

    return best_t


# ══════════════════════════════════════════════════════════════════════════════
# QUICK REFERENCE
# ══════════════════════════════════════════════════════════════════════════════
#
#  SINGLE MODEL SCORECARD
#  results = evaluate_model(y_true, y_probs, 'Logistic Regression', threshold=0.5)
#
#  CUSTOM COLOUR (for multi-model consistency)
#  import python_style_util as psu
#  lr_results   = evaluate_model(y_true, lr_probs,   'Logistic Regression', color=psu.p('purple',500))
#  lgbm_results = evaluate_model(y_true, lgbm_probs, 'LightGBM',            color=psu.p('cyan',600))
#
#  COMPARE TWO MODELS
#  compare_models(lr_results, lgbm_results)
#
#  THRESHOLD SWEEP
#  best_threshold = threshold_sweep(y_true, y_probs, 'LightGBM')
#
# ══════════════════════════════════════════════════════════════════════════════
