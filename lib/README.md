# lib

Shared helpers used by the notebooks and Quarto decks. Vendored (copied) from the `Core Resources` folder on 2026-11-02. Edit the copy here; this repo does not read `Core Resources`.

| Module | Used for |
|---|---|
| `python_style_util.py` | Palette, fonts and chart styling tokens |
| `credit_utils/` | EDA, quality checks, WOE/IV, stats, style helpers |
| `evaluation_utils.py` | Model evaluation helpers (Gini, KS and similar) |
| `deck_utils.py` | Quarto deck styling and chart labels |
| `brand_palettes.py` | Brand colour palettes used by `deck_utils` |

Notebooks and decks add this folder to `sys.path` via `find_repo_root`, which walks up from the working directory until it finds a folder containing `lib/`.
