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
- 2026-10-30 10:20 WIB: REOPENED by the user's 4-member recipe (LightGBM per segment, XGBoost per segment, MLP,
  LightGBM v10 nested). Plain MLP on engineered features (291 mlp, 5 seeds): 0.52186; next to eight_seg it is neutral
  (+0.00002). recipe3 = equal rank mean of lgb_seg + xgb_seg + mlp: OOF 0.52890, +0.00209 vs eight_seg, bootstrap
  P 0.89, 10/15 folds.
  - All 10 three-of-five MLP seed subsets stay positive (+0.0007 to +0.0021 mean fold delta), and weight 1/3 sits near
    the best weight.
  - The two segment models alone lose (−0.00083); the gain comes from MLP + segment trees together, not from either
    one. CONFIRMATORY (recipe fixed before seeing OOF).
  - Written: submissions/v75/v75_recipe3.csv. Waiting for lgb_nested (293) for recipe4.
- 2026-10-30 10:35 WIB: recipe variants on existing OOF (exploratory, ~9 compared, expect some winner's curse):
  - 4 segment trees + MLP at weight 1/3: 0.52953 (+0.0025, P 0.94).
  - Control "six + MLP 1/3": −0.0003. The pairing with segment trees is what matters.
  - The seqfull hybrid is a worse partner than the plain MLP.
  RULE LOCKED for the final (eda/295) before mlpseg, extra MLP seeds and lgb_v10 finish: tree part 2/3 =
  mean of the segment trees [+ lgb_v10], smooth part 1/3 = MLP seeds [+ mlpseg]. Candidates A-D.
- 2026-10-30 10:52 WIB: LOCKED-RULE RESULTS (295) with the 10-seed MLP:
  - A (4 segment trees 2/3 + MLP 1/3): 0.52766, +0.0011 vs eight_seg, P 0.78, 7/15.
  - C (A + MLP-per-segment in the smooth part): 0.52861, +0.0016, P 0.85, 10/15.
  - recipe3 re-run with the 10-seed MLP: +0.0010, P 0.75, 7/15 (was +0.0021 with 5 seeds).
  Lesson: the exploratory +0.0021..+0.0025 was partly the luck of the 5 MLP seeds; the MLP alone did not change
  (0.52186 → 0.52192). The real pairing effect is ~+0.001 to +0.0016 OOF, the same size as earlier gains that did
  not show on public. Waiting for lgb_v10 (B, D).
- 2026-10-30 11:05 WIB: lgb_v10 (nested-tuned LightGBM, 3 seeds, eda/293) OOF 0.52368. Adding it to the tree part
  lowers the blend: B 0.52745 (+0.0007, P 0.70) < A, and D 0.52843 (+0.0014, P 0.82) < C. Best locked candidate:
  C = 4 segment trees (2/3) + MLP 10 seeds & MLP-per-segment 5 seeds (1/3), OOF 0.52861, +0.0016 vs eight_seg,
  P 0.85, 10/15. File submissions/v75/v75_recipe_C.csv (Spearman 0.9957 vs v72). Nothing else running.
- 2026-10-30 11:10 WIB: submitted v75_recipe_C (OOF 0.52861) → public 0.56374. Every file that moved away from v72
  scored lower on public, and the order is the reverse of OOF:

  | File | OOF | Public |
  |---|---|---|
  | v72 | 0.5258 | 0.56822 |
  | v73 | 0.5258 | 0.56719 |
  | eight_seg | 0.5271 | 0.56533 |
  | C | 0.5286 | 0.56374 |
  | xgb_seg16 | 0.5278 | 0.56225 |

  The files share members (segment trees), so these are not independent draws. Either the public 9k favours
  v72-like ranking by chance, or the segment/MLP gains do not transfer to test. Decision: final stays v72.
