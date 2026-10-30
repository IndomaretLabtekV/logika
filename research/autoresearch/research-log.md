# Research Log

- 2026-10-30 09:40 WIB: workspace initialised. Bootstrap from ~25 experiments since 28 Oct (eda/250-288, see
  findings.md). Outer loop 0 decided BROADEN. Features look saturated (residual booster stops at 1-2 trees). The
  remaining levers are a different representation (H2), structure (H3), and segment-specific blending (H1).
  Order: H1 (minutes), H2 (hours), H3.
- 2026-10-30 09:55 WIB: H1 REFUTED (eda/289). Nested segment-specific blend: −0.00055/fold (se 0.00053), won 5/15.
  The chosen set per segment changes from fold to fold (9 different pairs), and per-segment isotonic recalibration of
  eight_seg alone also loses (−0.00037, 4/15). Rules out segment-wise blending and recalibration as a lever. Next: H2.
- 2026-10-30 10:35 WIB: H2 REFUTED (eda/290-292). Event-set model on raw sequences (MPS, ~3 min per seed):
  - Raw events + application fields only: OOF 0.51266, Spearman 0.950 with eight_seg. As a rank member it loses
    at every weight (−0.0005 to −0.0010).
  - Hybrid (raw events + all engineered features): 0.5210 per seed, 0.52328 as a 5-seed mean, Spearman 0.982.
    The blend is neutral: +0.00006, P 0.54, 7/15 folds.
  - Event order and cross-table timing hold no information beyond the aggregates.
- 2026-10-30 10:40 WIB: OUTER LOOP 1 → CONCLUDE (coherent negative result). Three independent routes all converge
  on OOF ≈ 0.527: features (residual booster 1-2 trees), representation (raw-sequence model), and blending
  (segment-wise and weights). H3 (new-loan simulation) dropped: the hybrid model already sees every installment
  and the new loan's amount/rate/installment and found nothing.
