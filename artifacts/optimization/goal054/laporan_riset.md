# Riset KS OOF 0,54

**Target belum tercapai pada eksperimen lima fold yang selesai.**

Lima fold stratified seed 31027, 70.000 OOF. Public baseline 0,56393 menurut pengguna; public semua kandidat baru belum diketahui.

| Kandidat | Bobot baru | KS pooled | Rata-rata KS fold | Δ pooled | Fold membaik |
|---|---:|---:|---:|---:|---:|
| public_056393_recipe | 0 | 0.528510 | 0.530955 | +0.000000 | 0/5 |
| calendar_transfer_l20p0003_balanced_d0p5_w1.0 | 1 | 0.528666 | 0.531710 | +0.000156 | 5/5 |
| calendar_transfer_l20p0003_balanced_d1p0_w1.0 | 1 | 0.527954 | 0.529947 | -0.000556 | 2/5 |
| calendar_transfer_l20p0003_unweighted_d0p5_w1.0 | 1 | 0.528812 | 0.531816 | +0.000302 | 5/5 |
| calendar_transfer_l20p0003_unweighted_d1p0_w1.0 | 1 | 0.528936 | 0.531291 | +0.000426 | 2/5 |
| calendar_transfer_l20p003_balanced_d0p5_w1.0 | 1 | 0.528174 | 0.531630 | -0.000336 | 5/5 |
| calendar_transfer_l20p003_balanced_d1p0_w1.0 | 1 | 0.528652 | 0.531202 | +0.000142 | 3/5 |
| calendar_transfer_l20p003_unweighted_d0p5_w1.0 | 1 | 0.528181 | 0.531361 | -0.000329 | 3/5 |
| calendar_transfer_l20p003_unweighted_d1p0_w1.0 | 1 | 0.529173 | 0.531310 | +0.000663 | 3/5 |
| ebm_denoised_class_balanced_w1.0 | 1 | 0.521115 | 0.523545 | -0.007395 | 0/5 |
| ebm_denoised_class_balanced_w0.25 | 0.25 | 0.528044 | 0.530723 | -0.000466 | 3/5 |
| ebm_denoised_class_balanced_w0.5 | 0.5 | 0.526949 | 0.529906 | -0.001561 | 2/5 |
| ebm_denoised_class_balanced_w0.75 | 0.75 | 0.524312 | 0.526778 | -0.004198 | 2/5 |
| ebm_denoised_unweighted_w1.0 | 1 | 0.520973 | 0.525582 | -0.007537 | 0/5 |
| ebm_denoised_unweighted_w0.25 | 0.25 | 0.528605 | 0.531470 | +0.000095 | 3/5 |
| ebm_denoised_unweighted_w0.5 | 0.5 | 0.527224 | 0.530458 | -0.001286 | 2/5 |
| ebm_denoised_unweighted_w0.75 | 0.75 | 0.524811 | 0.527332 | -0.003699 | 1/5 |
| kumo_offset_l20p003_balanced_d0p5_w1.0 | 1 | 0.528594 | 0.531964 | +0.000084 | 4/5 |
| kumo_offset_l20p003_balanced_d1p0_w1.0 | 1 | 0.526354 | 0.530820 | -0.002156 | 3/5 |
| kumo_offset_l20p003_unweighted_d0p5_w1.0 | 1 | 0.529299 | 0.532907 | +0.000789 | 5/5 |
| kumo_offset_l20p003_unweighted_d1p0_w1.0 | 1 | 0.527570 | 0.530399 | -0.000940 | 2/5 |
| kumo_offset_l20p03_balanced_d0p5_w1.0 | 1 | 0.528817 | 0.532025 | +0.000307 | 4/5 |
| kumo_offset_l20p03_balanced_d1p0_w1.0 | 1 | 0.527878 | 0.530653 | -0.000632 | 2/5 |
| kumo_offset_l20p03_unweighted_d0p5_w1.0 | 1 | 0.529077 | 0.532086 | +0.000567 | 4/5 |
| kumo_offset_l20p03_unweighted_d1p0_w1.0 | 1 | 0.528937 | 0.532272 | +0.000427 | 4/5 |
| tabpfn35_fast_v5_cache_e8_ctx50000_w1.0 | 1 | 0.522576 | 0.527914 | -0.005934 | 1/5 |
| tabpfn35_fast_v5_cache_e8_ctx50000_w0.25 | 0.25 | 0.527393 | 0.530690 | -0.001117 | 2/5 |
| tabpfn35_fast_v5_cache_e8_ctx50000_w0.5 | 0.5 | 0.526444 | 0.529849 | -0.002066 | 1/5 |
| tabpfn35_fast_v5_cache_e8_ctx50000_w0.75 | 0.75 | 0.524526 | 0.529218 | -0.003984 | 2/5 |
| tabpfn35_large_denoised_cache_e8_ctx50000_w1.0 | 1 | 0.526224 | 0.529078 | -0.002286 | 0/5 |
| tabpfn35_large_denoised_cache_e8_ctx50000_w0.25 | 0.25 | 0.528415 | 0.531410 | -0.000095 | 3/5 |
| tabpfn35_large_denoised_cache_e8_ctx50000_w0.5 | 0.5 | 0.528333 | 0.530612 | -0.000177 | 3/5 |
| tabpfn35_large_denoised_cache_e8_ctx50000_w0.75 | 0.75 | 0.527419 | 0.530158 | -0.001091 | 1/5 |
| tabpfn35_large_v5_cache_e8_ctx50000_w1.0 | 1 | 0.522748 | 0.528604 | -0.005762 | 2/5 |
| tabpfn35_large_v5_cache_e8_ctx50000_w0.25 | 0.25 | 0.527396 | 0.531079 | -0.001114 | 2/5 |
| tabpfn35_large_v5_cache_e8_ctx50000_w0.5 | 0.5 | 0.525704 | 0.530761 | -0.002806 | 2/5 |
| tabpfn35_large_v5_cache_e8_ctx50000_w0.75 | 0.75 | 0.524695 | 0.530109 | -0.003815 | 1/5 |

Model yang belum menyelesaikan lima fold tidak diberi skor pooled: kumo_relational_v5_e8_ctx50000_h1, kumo_relational_v5_e8_ctx50000_h1_readout, tabpfn35_fast_v5_e8_ctx50000

Model and feature selection reuse training labels and folds. This is not an untouched holdout; changed submissions have unknown public score.

Audit menghitung KS dengan SciPy dan sklearn, memeriksa semua ID outer-training/validation, konteks pretrained, fingerprint fitur dan checksum prediksi.
