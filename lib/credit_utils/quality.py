"""
quality — data-quality and cleaning helpers for the PD model projects.

Empty for now — we'll add functions here as we lift them out of the notebook,
once they've proven useful. Same design rules as `eda`: take data as an
argument, return values rather than mutating globals, keyword defaults.

Planned / likely contents (add when needed, not before):
  * flag_sentinel       — replace disguised nulls (e.g. DAYS_EMPLOYED == 365243)
                          with NaN and return a boolean sentinel mask.
  * infer_statistical_type — binary / discrete / continuous / categorical / id.
  * missingness_table   — per-column missing count and % as a tidy frame.
  * block_missingness   — nullity-correlation check within a column family.
"""
