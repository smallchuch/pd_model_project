"""
stats — hypothesis tests to corroborate that top drivers separate the classes.

Empty for now — this backs Section 4. The aim is small wrappers around scipy
tests that return a tidy result (statistic, p-value, effect size) plus a plain
one-line verdict, so the notebook stays readable and the base rate is always in
view.

Planned / likely contents (add when needed):
  * numeric_driver_test   — t-test / Mann-Whitney for a numeric feature vs TARGET.
  * categorical_driver_test — chi-square for a categorical feature vs TARGET.
  * default_rate_by_group — group default rate vs the overall base rate.
"""
