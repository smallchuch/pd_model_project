# Flagship PD: EDA Defence Questions

Created 2026-10-02. Live copy with comments: the "Flagship PD: EDA Defence Questions" doc in Claude.

## How to use this file

Answer each question from memory with the notebook closed, then grade yourself against it. Skimming feels like understanding until you have to produce the explanation cold.

1. Close the notebook. Type 2-3 sentences under "Your answer".
2. Open the notebook and check the "Check against" pointer. Write a grade: Defended, Vague (right answer, weak reason) or Couldn't answer.
3. Rework every Vague and Couldn't answer, then re-ask it 3-4 days later and log the new grade.

Q3, Q5, Q6, Q10, Q14 and Q17 are the likeliest to come up in an interview or a model review, so start there. For a harder pass, ask Claude to act as a skeptical model validator and put the questions one at a time, pushing back when your reason doesn't match the data.

---

## Target and framing

The target is a proxy with undisclosed thresholds, and every bin is read against an 8.1% base rate. You should be able to say both without looking.

### Q1. What does `TARGET` actually measure, and why can't you call it a Basel default? What would you need from Home Credit to fix that?

Your answer:

Check against: Section 8 assumptions in the EDA notebook; the target-definition note in the notebook 02 handoff.

Grade:

### Q2. The base rate is about 8.1%. How did you use it when reading every bin, and what does a bin at 8.0% vs 12% tell you?

Your answer:

Check against: the default-rate-by-bin outputs in Section 3 and the discipline rules in the 3.5-to-modelling handoff.

Grade:

---

## Split and leakage

The rule is split first, fit on train only, apply to validation. These four questions test whether you can explain why, not just recite it.

### Q3. Why must the split come before binning, and what exactly gets inflated if you bin first?

Your answer:

Check against: split cell (5.1) and the "split first, fit on train only" rule in the 3.5-to-modelling handoff.

Grade:

### Q4. Why stratify on `TARGET`? What would go wrong with a plain random split at about 8% events?

Your answer:

Check against: the 80/20 stratified split in 5.1 and the resulting default rate in each split.

Grade:

### Q5. `ORG_BAND` was first built on the full data. Walk through exactly how that leaks and how the train-only refit fixes it.

Your answer:

Check against: the ORG_BAND leakage note in 3.4 and the refit-in-5.1 caveat.

Grade:

### Q6. Your `calc_woe_iv` helper re-derived bin edges from whatever frame it was given. Why does that silently break validation, and what is the fix?

Your answer:

Check against: `calc_woe_iv` (5.2) and the fit-edges-on-train, apply-to-val note in the handoff's open fixes.

Grade:

---

## Binning choices

Every binning treatment in the plan has a reason tied to the data. Defend each one by naming the alternative you rejected.

### Q7. Why quantile bins for amounts but fixed domain bands for age and employment? When would you pick the other way?

Your answer:

Check against: the carry-forward decisions table in the 3.5-to-modelling handoff; 3.2 and 3.3.

Grade:

### Q8. Why is no log transform needed before quantile binning?

Your answer:

Check against: the income and credit rows of the carry-forward table (quantiles are rank-invariant).

Grade:

### Q9. Why is missing its own bin for the EXT scores but not for the loan amounts?

Your answer:

Check against: 2.3 missingness and 3.1 (missingness in EXT_SOURCE_1 is informative).

Grade:

### Q10. What is the `DAYS_EMPLOYED` sentinel, and how did `NAME_INCOME_TYPE` help you interpret it? What would you have wrongly concluded without that?

Your answer:

Check against: 3.3 Age and Employment, and the NAME_INCOME_TYPE row of the carry-forward table.

Grade:

### Q11. Why does a zero-default level (Student, Businessman) cause infinite WOE, and why is pooling by similar default rate better than pooling by size?

Your answer:

Check against: 3.3 and the guard on zero-bad bins in the discipline rules.

Grade:

### Q12. Why did you treat education as ordinal and occupation as nominal, and what would you expect to see in the WOE if ordinal was right?

Your answer:

Check against: 3.4 Key Demographics and the NAME_EDUCATION_TYPE row of the carry-forward table.

Grade:

---

## IV, Gini and interpretation

These questions check whether you can explain the metrics to a non-technical reader and read them with suspicion.

### Q13. Explain WOE and IV to a credit manager in two sentences, no formulas.

Your answer:

Check against: the IV bands (Siddiqi) in the discipline rules and the `calc_woe_iv` output in 5.2.

Grade:

### Q14. Why is IV above 0.5 a warning rather than a win?

Your answer:

Check against: discipline rule 5 in the 3.5-to-modelling handoff.

Grade:

### Q15. EXT_SOURCE_3 has a single-feature Gini of 0.36. What does that number mean in plain terms, and why can't you simply add the three EXT Ginis together?

Your answer:

Check against: 3.1 External Scores and the Score Intercorrelation cell.

Grade:

### Q16. Loan amounts had weak, non-monotonic signal but your ratios should be cleaner. Why, and how does that map to LVR and serviceability in your day job?

Your answer:

Check against: 3.2 Loan Financials and 4.2 default-rate-by-bin for the ratios.

Grade:

---

## Feature selection and fairness

Excluding `CODE_GENDER` is the start of the fairness answer, not the end. Be ready to explain what you have and haven't checked.

### Q17. Why exclude `CODE_GENDER`, and why is that not the end of the matter? Which retained features might proxy for it, and how would you test it?

Your answer:

Check against: 3.4 Key Demographics, the CODE_GENDER row of the carry-forward table, and the "no proxy check run yet" note in the notebook 02 handoff.

Grade:

### Q18. `CNT_CHILDREN` and `CNT_FAM_MEMBERS`: why keep one, and how do you pick?

Your answer:

Check against: 3.4 and the CNT row of the carry-forward table.

Grade:

### Q19. Which features would you drop for collinearity, and what evidence would convince you to keep both of a pair?

Your answer:

Check against: the feature-selection list in the 3.5-to-modelling handoff and the overlap pairs identified in notebook 02 (1.1a).

Grade:

---

## Defending the whole screen

A validator will ask where the work is weakest. Naming your own limits first is what separates a defensible model from a confident one.

### Q20. A validator says "your top driver is just a bureau score someone else built." What is your response, and is it a real weakness?

Your answer:

Check against: 3.1 External Scores and the intercorrelation check.

Grade:

### Q21. What is the single decision in the notebook you are least confident in, and what test would settle it?

Your answer:

Check against: your own Observation and Decision cells across Section 3.

Grade:

### Q22. Name two things the notebook cannot tell you.

Your answer:

Check against: Section 8 assumptions and the "known gaps carried over" list in the notebook 02 handoff.

Grade:

---

## Re-ask log

Log each re-ask here so the weak spots are visible as they shrink. Add a new section for each notebook as you finish it, for example notebook 02 on coefficient signs, VIF and regularisation, then notebook 03 on tuning and challenger models.

| Date | Questions re-asked | Grades | What I reworked |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
