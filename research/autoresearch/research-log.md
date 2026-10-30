# Research Log

- 2026-10-30 09:40 WIB: workspace initialised. Bootstrap from ~25 experiments since 28 Oct (eda/250-288, see
  findings.md). Outer loop 0 decided BROADEN. Features look saturated (residual booster stops at 1-2 trees). The
  remaining levers are a different representation (H2), structure (H3), and segment-specific blending (H1).
  Order: H1 (minutes), H2 (hours), H3.
- 2026-10-30 09:55 WIB: H1 REFUTED (eda/289). Nested segment-specific blend: −0.00055/fold (se 0.00053), won 5/15.
  The chosen set per segment changes from fold to fold (9 different pairs), and per-segment isotonic recalibration of
  eight_seg alone also loses (−0.00037, 4/15). Rules out segment-wise blending and recalibration as a lever. Next: H2.
