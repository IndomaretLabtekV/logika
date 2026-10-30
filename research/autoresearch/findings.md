# Research Findings — LOGIKA UI 2026 credit KS

## Research Question

What, if anything, can still raise the KS (DPD ≥ 90 within 12 months) of a trees + repayment-Markov ensemble whose
current features look saturated (OOF KS ≈ 0.527, public 0.56822 for v72)?

## Current Understanding (outer loop 0, 30 Oct 2026)

The data come from a generator with one latent risk per customer that drifts slowly over time. The generator writes
that latent into several channels with independent noise:

- repayment lateness, as a Markov process whose entry rate rises with installment index and amount;
- e-commerce unit prices, the same shift in every category;
- a pre-decision e-commerce surge for later defaulters (2.8x in the last day, back to 1x at ~90 days);
- an account net-flow collapse for later defaulters in the last ~4 months.

The interest rate is a rate card on observables plus noise unrelated to the target.

The v22 features plus the f251 Markov block already extract each channel. A booster fitted on the OOF residual stops
after 1–2 trees, and segment calibration is flat (except decision month 5).

Test is a random split: adversarial AUC 0.501 and the same months. Train labels show no random-flip floor.

Public KS sits ~0.04 above OOF for every model family, which is unexplained. On 9k rows the public score cannot
rank files whose OOF differs by ≤ 0.002. Measured examples:

| File | OOF | Public |
|---|---|---|
| v72 | 0.5258 | 0.56822 |
| v73 | 0.5258 | 0.56719 |
| v74 eight_seg | 0.5271 | 0.56533 |
| v74 xgb_seg16 | 0.5278 | 0.56225 |

## Key Results (before this workspace; scripts in eda/)

Positive:

- f251 repayment Markov: +0.003 public.
- Diverse members at 8 seeds:
  - lgb_x 0.5258 and xgb_d8 0.5260;
  - per-has_history-segment models: xgb_seg 0.5264, 16 seeds 0.5278;
  - eight_seg blend 0.52706 (+0.0008 vs six, bootstrap P 0.83).

Negative (single LGB or LR screen, controls where noted):

- Features:
  - O−E lateness (266);
  - price shrinkage and factor model (268);
  - teammate's 638-column spec (271);
  - aux repayment-latent model (277; the second stage gives the same per-fold gain for control blocks);
  - lender residuals (283);
  - kNN target mean and density (284);
  - e-commerce LLR timing (286);
  - cross-channel recent index (287);
  - bureau score change (288; LR chi2 14 but single LGB −0.0004 to −0.0044 on 3 seeds).
- Labels and metric:
  - pseudo-label (literature: ≈ 0 at 70k labels);
  - label-noise down-weighting (272);
  - KS-cut-focused weights (282).
- Models:
  - LGB/XGB tuning (273/274);
  - monotone, DART, RF/ET not finished or neutral;
  - GAM blend (269);
  - TabM/NN blends (team history).

## Outer loop 1 (30 Oct 10:40): conclusion

- H1, segment-specific blend: refuted. Nested selection loses (−0.00055/fold) and the chosen sets are unstable.
- H2, raw-event model: refuted.
  - Alone it is strong for a model that sees no engineered feature: 0.5127.
  - With the engineered features (5-seed hybrid) it reaches 0.5233.
  - Its rank blend with eight_seg is neutral (+0.00006).
- H3 (new-loan simulation) was dropped because the hybrid model already had its inputs.

Conclusion: every route converges at OOF ≈ 0.527:

- features (a residual booster stops at 1-2 trees);
- representation (raw sequences);
- blending (weights, segments).

The remaining gap to the top-8 public scores (~0.004) is within public noise for 9k rows (sd ~0.011). The best use of
the score budget is the submission with the highest locked-in public (v72, 0.56822), whose kernel already reproduces
end to end.

## Lessons and Constraints

- Single-seed single-LGB deltas have per-fold sd ~0.003; require ≥ 3 seeds before believing |Δ| < 0.003.
- A high LR-screen chi2 can fail in trees (288). The screen is a filter, not a verdict.
- Any second stage boosted on the blended OOF shows +0.0007/fold even for useless blocks. Always run control blocks.
- Weak-but-different members (GAM 0.516) do not raise KS; members must be both ~0.525 and decorrelated.
- Do not spend submissions to rank near-identical files; the public noise is larger than the OOF differences.

## Open Questions

- Why public sits ~0.04 above OOF for every model family is still unexplained (not a split shift, not label noise).


- Does a model that sees raw event sequences (not aggregates) make different, useful errors?
- Does a structural simulation of the new loan's 12 installments add what trees cannot compose?
- Do the blend weights differ by segment?
