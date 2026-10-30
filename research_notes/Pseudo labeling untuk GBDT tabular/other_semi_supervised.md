# Semi-supervised and transductive techniques, other than plain pseudo-labeling, for raising KS/AUC of a tabular GBDT ensemble

Context: credit-risk binary task, metric KS, about 70k labeled rows, about 30k unlabeled test rows, about 13.6% positives, rank-averaged GBDT ensemble with OOF KS 0.526. The data is probably synthetic and external data is not allowed. Project memory says the current best public score is 0.56822 (v72: 6 GBDT plus a repayment-Markov feature, 3 seeds). It also says public KS has an SD of about 0.011 on roughly 9k rows, and that a cross-fitted month logit offset gained +0.001 OOF but lowered the public score.

Research budget: about 15 searches and fetches. Primary sources were used where they could be fetched. Several Kaggle forum pages, including Jahrer's Porto Seguro post discussion/44629, do not render without login, so secondary sources stand in for them where noted.

---

## Q1. Adversarial validation: detecting shift, choosing folds, and importance-weighting training rows

### Takeaway
Adversarial validation is cheap and well worth running as a **diagnostic**. If the train-vs-test AUC is about 0.5, which is typical for a random split of synthetic data, there is nothing to correct and every adaptation equals the baseline. In the only controlled study found (Uber, AdKDD'20), **inverse-propensity weighting and propensity-matched validation sets were neutral or harmful on 6 of 7 benchmark datasets**. The method that helped was **dropping the features that drive the shift**.

### Cited Findings
- Uber's paper compares three adversarial-validation strategies: (a) iterative **feature removal**, which drops the top adversarial-importance features until adversarial AUC falls below a threshold; (b) **validation selection**, which uses propensity-score matching to build a validation set that looks like test; and (c) **inverse propensity weighting**, which weights training rows by p/(1-p) with p = P(test|x) and trims weights near p = 1. — [Pan et al., arXiv 2004.03045](https://ar5iv.labs.arxiv.org/html/2004.03045)
- On the AutoML3 datasets (test AUC %), the **ADA dataset had adversarial AUC of about 49%, i.e. no shift, and every method matched the baseline** (92.14 vs 92.12–92.17). On the six shifted datasets, **IPW was below baseline on all six** (for example AA 70.68 to 66.18, B 57.12 to 54.47, E 81.93 to 79.66). **Validation selection was also below baseline on all six.** Feature removal with DT/RF importance gained about 0.75–3.7 points. Feature removal with GBDT importance collapsed on one dataset (RL 64.07 to 52.44). — [Pan et al.](https://ar5iv.labs.arxiv.org/html/2004.03045)
- On Uber's internal MaLTA data, averaged over the later test snapshots (my averaging of their table), the baseline was about 62.0 AUC, validation selection about 62.0, **IPW about 62.5 (small gain)**, and feature removal with GBDT about 64.2. The paper's text claims gains (3.9%, 1.2–5.5%) that do not match its own tables, so use the tables. — [Pan et al.](https://ar5iv.labs.arxiv.org/html/2004.03045)
- In theory, importance-weighted CV is an unbiased model-selection method under covariate shift, where ordinary CV is not. — [Sugiyama et al., Importance-weighted CV (JMLR 2007)](https://pure.korea.ac.kr/en/publications/covariate-shift-adaptation-by-importance-weighted-cross-validatio/)
- Importance weighting has **high variance when train and test differ a lot**, and this motivates robust or regularized variants. — [Li et al., Robust Importance Weighting for Covariate Shift, AISTATS 2020](https://proceedings.mlr.press/v108/li20b.html)
- Another tabular CTR paper uses adversarial validation to explain a gap between local CV and the leaderboard and to pick features. It is a diagnostic use. — [arXiv 2310.03778](https://arxiv.org/pdf/2310.03778)

### Inferences
- **How to implement:** concatenate train and test features (no target), set `is_test` = 0/1, and run 5-fold LightGBM (shallow, about 200 trees) to get OOF AUC and feature importance. Then:
  - AUC < about 0.55: no action. Do **not** reweight, because the weights are almost constant and only add noise.
  - AUC well above 0.5: check which features drive it. ID-like, time-like, and row-index-like columns are the usual culprits. Drop them or convert them to relative versions. This is the strategy with the best evidence.
  - IPW in LightGBM/XGBoost/CatBoost: pass `sample_weight = clip(p/(1-p), lo, hi)`, using the OOF adversarial p for training rows, and normalize to mean 1. Evaluate with **unweighted** OOF KS and with **weighted** OOF KS, since the weighted one estimates test KS.
- **For this competition:** the project has a time/month structure. Memory notes customer risk drift and that the month offset hurt on public. If test covers different months than train, adversarial AUC will be high only because of date features. Then the useful action is the Uber-style one: remove or relativize the drifting date-derived features, not reweight. If test is a random sample of the same months, adversarial validation will probably show AUC of about 0.5, and the expected gain is zero.
- **Using adversarial p to choose folds** (validating on the most test-like 20% of train) mainly makes model selection more honest. It does not raise the score directly. Given the public SD of about 0.011, it can help avoid chasing public noise.
- **Gain/cost:** about 30 minutes for the diagnostic. Expected gain is zero unless the diagnostic shows real shift. Run it once to settle the question.

### Gaps
- I found no Kaggle write-up with before/after leaderboard numbers for IPW from adversarial validation. Folk wisdom on Kaggle is mixed, but I could not fetch primary forum posts.
- I found no study of KS specifically, as opposed to AUC, under importance weighting.

---

## Q2. Knowledge distillation: one student trained on teacher soft targets over train plus test

### Takeaway
The distillation literature shows that a student can **match** an ensemble. Occasionally a tree student slightly **exceeds** a non-tree teacher in AUC. I found no evidence that a GBDT student trained on GBDT-ensemble soft labels beats the GBDT ensemble on a rank metric. For a rank-averaged GBDT ensemble with plentiful labels, distillation is mainly **compression**, and the expected KS gain is about 0. The variant with real upside is **cross-family distillation**: an NN or other non-tree student fitted to GBDT soft labels on train plus test, then **blended**, not substituted.

### Cited Findings
- "Practical Knowledge Distillation" (Lee, Apostolopulos, Markov, tabular) distills ensembles into a single gradient-boosting model with data sampling and denoising. The authors frame it as compressing for real-time inference "without performance loss", not as a gain. — [arXiv 2302.12360](https://ar5iv.labs.arxiv.org/html/2302.12360)
- In distillation from tabular foundation models into tree students (health data), the best CatBoost student reached about **0.877 AUC, slightly above the best teacher**, and XGBoost students reached about 0.875. The authors describe distillation as **occasionally** improving AUC. — [arXiv 2605.18702](https://arxiv.org/pdf/2605.18702)
- A secondary summary of a TabICLv2-to-XGBoost pipeline says gains concentrate on **low-dimensional** data and that **averaging several teachers adds little for tree students**. It is unverified against the primary paper. — [ownyourai.com summary](https://ownyourai.com/?p=49721)
- Exact "born-again tree ensembles" (Vidal & Schiffer, ICML 2020) aim to **reproduce** an ensemble's decision function in a single tree, not to improve on it. — [ICML 2020 slides](https://ICML.cc/media/icml-2020/Slides/6682.pdf)
- SLaM (NeurIPS 2023) notes that teacher soft pseudo-labels on unlabeled data are often noisy and "lead to impaired student performance", so methods are needed to correct for teacher noise. The setting is neural. — [SLaM, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d56b84c063265da949fe0feb815dcce8-Abstract.html)
- Ensemble-to-student distillation with unlabeled data can weight unlabeled rows by teacher disagreement. The setting is neural. — [search result, Wu et al. via arXiv 2204.00548](https://www.arxiv.org/pdf/2204.00548)
- In the AMEX Default Prediction competition (credit default, metric built from Gini and top-4% capture), a 15th-place gold solution used **a Transformer trained with LightGBM knowledge distillation**. This is the closest credit-risk precedent: a cross-family student, used inside a blend. — [AMEX competition summary](https://realvincentyuan.github.io/Spacecraft/amex-default-prediction-kaggle-competition-summary/index.html)

### Inferences
- **Why a same-family student does not help a rank metric:** a GBDT student fitted to the ensemble's averaged logit learns a smoothed approximation of the same function. The ranking it produces is close to the teacher's, with approximation error added. The teacher's ensemble already has the variance reduction that distillation would carry over. Born-again gains in NNs come from label smoothing and dark knowledge across classes. Binary classification has little of that.
- **Implementation (if tried):**
  1. Build cross-fitted teacher scores: OOF for train and averaged fold predictions for test. Convert the rank-mean to a logit-scale target, for example the logit of an isotonic- or Platt-calibrated probability.
  2. Train the student on 100k rows (train plus test) with **regression on the teacher logit** (LightGBM `objective='regression'`, or `cross_entropy` with soft labels in [0,1], which LightGBM `xentropy` supports). For train rows, optionally mix in the hard label: `y_soft = a*y + (1-a)*p_teacher`.
  3. **Leakage trap:** the student's OOF score on train is not honest if the teacher's OOF used the same folds incorrectly, or if test soft labels came from models that saw the validation fold. Use **nested folds**: for each outer fold k, build teacher labels only from models that never saw fold k, then validate the student on fold k against **hard labels**.
- **The best-value variant:** an MLP or ResNet student on GBDT soft labels plus RankGauss inputs, trained on train plus test, then rank-blended with the trees. Diversity is what helps a rank-average. Memory shows the additive GAM blend gained only +0.0004 AUC and about 0 KS. That suggests the tree ensemble already captures most of the signal, so set expectations low (about +0.000–0.003 KS, inside public noise).
- **Gain/cost:** a GBDT student costs about 1 kernel run, with expected gain about 0 and a risk of leakage-inflated OOF. An NN student costs more (GPU), with small expected gain that is honestly measurable only through nested CV.

### Gaps
- I found no paper or Kaggle write-up showing a GBDT-ensemble-to-GBDT-student distillation (with or without unlabeled test rows) beating the teacher on AUC/KS.
- I could not obtain the AMEX 15th-place write-up's numeric gain from distillation.

---

## Q3. Co-training and tri-training with different feature views or model families

### Takeaway
There is little modern tabular evidence that co-training or tri-training helps **when labels are plentiful**. The published tabular co-training and tri-training benchmarks are all low-label regimes (1–20% labels, or 50–500 labeled rows). With 70k labels, these methods reduce to "pseudo-labeling with extra steps", and the same confirmation-bias risks apply.

### Cited Findings
- Co-training with optimal weights pairs XGBoost with a TSVM, motivated by the complementary properties of two strong heterogeneous classifiers (larger diversity). The snippet did not give benchmark numbers. — [arXiv 2006.04097](https://pdf.arxiv.org/pdf/2006.04097)
- ReCom (Information Sciences 2022), an RL-based co-training method for tabular data evaluated on 32 datasets, describes earlier tabular co-training as heuristic or greedy. — [ReCom](https://cris.iucc.ac.il/en/publications/recom-a-deep-reinforcement-learning-approach-for-semi-supervised-/)
- Recent tabular SSL benchmarks use **1%, 5%, and 10% labeled fractions** on OpenML datasets, and the reported gains are concentrated at 1%. — [arXiv 2604.16412](https://arxiv.org/pdf/2604.16412)
- In TabTransformer's semi-supervised comparison, **GBDT with pseudo-labeling performed worse than the average** of the methods compared, in a few-labels/many-unlabeled setting. — [TabTransformer, arXiv 2012.06678](https://arxiv.org/pdf/2012.06678)
- Classic tri-training is a strong baseline in **NLP** neural SSL. — [Ruder & Plank, arXiv 1804.09530](https://ar5iv.labs.arxiv.org/html/1804.09530)
- A regularized and curriculum pseudo-labeling method for tabular data (including GBDT) is reported as superior across models and datasets. — [arXiv 2302.14013](https://arxiv.org/pdf/2302.14013)

### Inferences
- **Implementation sketch:** view A is application-level features and view B is behavior/repayment-history features. Alternatively, use family A = LightGBM and family B = CatBoost. Each model labels test rows on which it is confident and the other model is uncertain. These rows are added with hard or soft labels to the other model's training set, and the process iterates 1–2 rounds. Tri-training: a row is added to model k when the other two models agree.
- **Why the prior is low here:** with 70k labels and 13.6% positives (about 9.5k positives), adding 30k noisy pseudo-labels mostly reinforces each model's existing decision boundary. The independence assumption between views is also weak in credit data: application and behavior features are correlated through the customer's latent risk, and memory shows loan-pair correlation of 0.47 to 0.26.
- **If it is tried anyway:** it is only meaningful to test it as a variant of the pseudo-labeling experiment that is covered elsewhere, with nested folds so that pseudo-labels for fold k come from models that never saw fold k.
- **Gain/cost:** low expected gain, moderate cost (several training rounds), high risk of an inflated OOF.

### Gaps
- I found no tabular benchmark of co-training or tri-training against a strong GBDT ensemble at 10k–100k labels.

---

## Q4. Unsupervised feature learning on train plus test jointly (frequency encoding, aggregates, DAE/VIME/SCARF embeddings, clustering, kNN)

### Takeaway
1. The **cheapest and most reliable** win in this family is **target-free statistics computed on train plus test**: frequency/count encoding, group sizes, and group means or deviations of numeric features within entity/category groups. On a synthetic dataset these can also expose generator structure.
2. **DAE representations** have one famous win (Porto Seguro 2017, DAE plus NN, with train plus test used for the DAE). That win was an NN-based solution blended with GBDT, not DAE features fed to GBDT.
3. **Contrastive SSL** (VIME/SCARF) gains are small and mostly not statistically significant, even under label scarcity. At full labels, there is no evidence that they help GBDT.

### Cited Findings
- **Porto Seguro 1st place (Jahrer):** a denoising autoencoder with **swap noise**, where about 15% of a row's features are replaced with values from other rows (example rate 0.15). Inputs used the **RankGauss** transform, and the network was a deep stack such as 221-1500-1500-1500-221, yielding about 4,500 hidden-activation features. Jahrer: "One can use train+test features to build the DAE. The larger the testset, the better." — [fast.ai forum quoting Jahrer](https://forums.fast.ai/t/porto-seguro-winning-solution-representation-learning/8499); also [Kaggle Book summary](https://packtpub.com/en-in/product/the-kaggle-book-9781801817479/chapter/modeling-for-tabular-competitions-9/section/denoising-with-autoencoders-ch09lvl1sec53)
- Jahrer said the "secret sauce" is the noise, not the layers. Implementations exist in the `kaggler` library (DAE class with `swap_prob`). — [Kaggler docs](https://kaggler.readthedocs.io/en/stable/_modules/kaggler/preprocessing/autoencoder.html); [Kaggle Workbook summary](https://packt-plus-non-live.prod.packtpub.com/en-ar/product/the-kaggle-workbook-9781804611210/chapter/the-most-renowned-tabular-competition-porto-seguros-safe-driver-prediction-1/section/examining-the-top-solution-ideas-from-michael-jahrer-ch01lvl1sec04)
- In Tabular Playground Feb 2021, a transformer DAE whose features fed a linear model reached CV RMSE of about 0.841–0.843. The author says this was "close but not as good as GBDT". — [Ryan Cheung, Denoise-Transformer-AutoEncoder](https://github.yxlr123.workers.dev/ryancheunggit/Denoise-Transformer-AutoEncoder)
- **SSL pretraining study (14 OpenML tasks, 2026 preprint):** gains over training from scratch were largest at 5% labels (+0.0074 AUC) and 10% labels (+0.0055), and **turned negative at 20% labels** (0.9145 vs 0.9173). SSL vs scratch was **not significant** (Wilcoxon p = 0.63, Holm-adjusted p = 1.0). VIME, SCARF, SubTab, and the authors' own objective did not differ significantly. At 10% labels, random forest (0.9015) beat the SSL model (0.8954), which beat LightGBM (0.8875) and XGBoost (0.8819). Baselines used a single seed. The study did not test SSL embeddings as GBDT features. — [arXiv 2608.24381](https://arxiv.org/html/2608.24381)
- SCARF (ICLR 2022) reports gains over no pretraining and over autoencoder alternatives on 69 OpenML-CC18 datasets, but **does not compare against GBDT**. — [SCARF, Google Research](https://research.google/pubs/scarf-self-supervised-contrastive-learning-using-random-feature-corruption/)
- Mechanics: SCARF corrupts about 60% of features by sampling from those features' marginals and trains with InfoNCE. VIME uses Bernoulli masks with mask-estimation and reconstruction losses. — [arXiv 2401.04266](https://arxiv.org/pdf/2401.04266)
- kNN features: in the Avito competition write-up, kNN with large k and cosine similarity worked well but was very slow. The winners also used a "semi-supervised" retrain on train plus test with test pseudo-labels. — [FastML on Avito](https://fastml.com/kaggle-vs-industry-as-seen-through-lens-of-the-avito-competition)

### Inferences
- **Ranking within this family (for this task):**
  1. **Train-plus-test target-free aggregates** (count encodings, per-entity sizes, and rank-within-group of features computed over all 100k applications): cheap, deterministic, no leakage risk because no target is used, and GBDT uses them directly. The project has probably already mined most of these (memory lists many feature dead ends). The remaining check is whether existing aggregates were computed on **train only**. Recomputing them on train plus test makes test-row features consistent, because test rows then see the same population statistics. This matters most for count/frequency features whose distribution differs between a 70k and a 100k pool.
  2. **kNN features without targets on train plus test** (distance to the k-th neighbor, local density, cluster ID or k-means distances on standardized key features): cheap with FAISS/sklearn. Gains for GBDT are usually small, because trees already partition the space. **kNN features that use targets** (mean target of neighbors) are a form of target encoding and **must be cross-fitted**, or the OOF is leaked.
  3. **DAE with swap noise on train plus test**, used to (a) feed an MLP head trained on labels, and blend that MLP with the trees for diversity, or (b) append a low-dimensional bottleneck (16–64 dims) as GBDT features. Path (a) is the proven Porto Seguro path. Path (b) has weak evidence. Cost is moderate (GPU on Kaggle).
  4. **VIME/SCARF contrastive pretraining:** lowest priority. Evidence shows small, non-significant gains, only under label scarcity, and the regime here (70k labels) is outside where it helps.
- **Synthetic-data angle:** if the data was produced by a generator, for example a GAN or CTGAN-style synthesizer, DAE or density features can capture joint-distribution artefacts that trees do not split on easily. This is speculative and I have no source for it on this dataset.
- **Honest validation:** unsupervised features fitted on train plus test (without labels) do **not** leak labels. The DAE can therefore be fitted once on all 100k rows and the usual 5-fold CV used downstream. Anything that touches labels (an MLP head, target kNN) must be fitted inside the folds.

### Gaps
- I could not fetch Jahrer's original post (Kaggle login wall). Exact gini numbers for DAE+NN vs LightGBM in his blend are unverified here.
- I found no benchmark that measures DAE/SCARF embeddings **as features for GBDT** at 10k–100k labels.

---

## Q5. Test-time label-distribution adjustment (prior shift, segment-wise) for KS

### Takeaway
A **global** prior shift does nothing for AUC/KS, because both depend only on ranking and a monotone transform preserves it. **Segment-wise** shifts, such as adding a per-month or per-segment logit offset, change the ordering **across** segments and can move KS either way. They help only if the segment base rates in test truly differ from what the model implies. The project already tested this: a cross-fitted month offset gave +0.001 OOF and lowered public by 0.004. That is consistent with the offset fitting noise or a shift that does not carry over to test.

### Cited Findings
- The population AUC and its Bayes-optimal scorer do not depend on the class prior, so AUC is unaffected by label skew. — [arXiv 2504.11284](https://arxiv.org/pdf/2504.11284)
- AUC-ROC depends only on score ranking, is unchanged by monotone transforms, and says nothing about calibration. — [arXiv 2506.14540](https://arxiv.org/pdf/2506.14540)
- Calibration leaves binary AUC unchanged. It can matter when the posterior is built by marginalizing over a joint that includes a group variable, which is the closest result to the segment-wise case. — [NeurIPS 2023 supplement](https://proceedings.neurips.cc/paper_files/paper/2023/file/4b0c1645f3d6a1730931e65ecbf91ac3-Supplemental-Conference.pdf)
- Methods exist to estimate test-time AUC under positive-class distribution shift without knowing the test prior. — [NeurIPS 2024, AUC maximization under positive distribution shift](https://papers.neurips.cc/paper_files/paper/2024/file/3f8c7eb848ffec848f3ed2b7ca44915d-Paper-Conference.pdf)
- Project memory reports that v71 plus the cross-fit month offset (eda/264) scored 0.56442 public vs 0.56649 without it, despite +0.001 OOF. (Internal source: memory file `best-submission-trees-markov.md`.)

### Inferences
- **Mechanism for KS:** KS = max over thresholds of (TPR − FPR). With segment-specific offsets δ_s, the score becomes logit(p) + δ_s. Rankings within each segment are unchanged, but segments slide past each other. That helps only if the model is mis-calibrated **between** segments in test.
- **When it is worth trying:**
  - (a) There is a segment variable with a known or estimable test base rate that differs from train. EM-style prior estimation (Saerens et al. 2002) can estimate test priors per segment from unlabeled test scores, but it needs good calibration within each segment.
  - (b) A rank-averaged ensemble mixes models whose segment calibrations differ. A per-segment isotonic calibration before averaging can then fix that. A safer form of (b) is to average **calibrated probabilities or logits** instead of raw ranks.
- **Recommendation:** treat further segment-prior tweaks as **low value and high noise** here. The public evaluation (SD about 0.011) cannot resolve gains of 0.001–0.003. If any adjustment is kept, its offsets must come from cross-fitted OOF (already done) and have a mechanistic reason to transfer to test, for example a known change in test-period base rate.
- **One rank-safe check:** compare rank-averaging with logit-averaging of calibrated members. Both are monotone-invariant at the member level, but they combine differently and can shift KS by a few thousandths. It is free to test on OOF.

### Gaps
- I found no published study of segment-wise prior correction aimed at maximizing **KS**. The analysis above is derived from first principles.

---

## Q6. Best expected gain/cost for a strong GBDT ensemble with plentiful labels

### Takeaway
With 70k labels and a mature GBDT ensemble, the literature shows that semi-supervised and transductive methods mostly pay off **under label scarcity**. The realistic gains here are small, about +0.000–0.005 KS, which is below public-LB noise (SD about 0.011). The ranking by expected gain/cost is:
1. Adversarial validation as a diagnostic, with Uber-style feature removal if shift appears.
2. Recomputing target-free aggregates on train plus test.
3. A cross-family student (an MLP on RankGauss or DAE features, optionally distilled from GBDT soft labels on train plus test) blended with the trees.
4. kNN or cluster features.
5. Co-training or tri-training and SCARF/VIME, which are the lowest priority.

### Cited Findings
- Adversarial-validation **feature removal** was the only one of three strategies with consistent gains in the controlled study. IPW and validation selection lost AUC on 6 of 7 public datasets. — [Pan et al.](https://ar5iv.labs.arxiv.org/html/2004.03045)
- SSL-pretraining gains vanish or reverse at 20% labels and are not significant even at 5–10%. — [arXiv 2608.24381](https://arxiv.org/html/2608.24381)
- The proven tabular use of unlabeled rows in competitions is the train-plus-test DAE with swap noise feeding an NN, which is then blended. — [fast.ai / Jahrer](https://forums.fast.ai/t/porto-seguro-winning-solution-representation-learning/8499)
- In credit default prediction (AMEX), cross-family distillation (Transformer distilled from LightGBM) was part of a gold-medal blend. — [AMEX summary](https://realvincentyuan.github.io/Spacecraft/amex-default-prediction-kaggle-competition-summary/index.html)
- Distillation into tree students is mostly about fidelity or compression, with occasional small AUC gains. — [arXiv 2302.12360](https://ar5iv.labs.arxiv.org/html/2302.12360); [arXiv 2605.18702](https://arxiv.org/pdf/2605.18702)

### Inferences
**Comparison table**

| Technique | Expected KS gain here | Cost | Leakage/overfit risk | Verdict |
|---|---|---|---|---|
| Adversarial validation (diagnostic) | 0 directly; avoids bad choices | Very low | None | **Do it once** |
| Adversarial feature removal | Only if adversarial AUC is well above 0.5 | Low | Low; can drop useful features (the GBDT-importance variant collapsed on one dataset) | Conditional |
| Importance weighting (IPW) | about 0 or negative | Low | Variance | Skip unless shift is strong and weights are clipped |
| Target-free train-plus-test aggregates/frequency | Small and positive at best | Low | None (no target) | **Good value** |
| kNN/cluster features (target-free) | Small | Low to medium | None; target-kNN must be cross-fitted | Optional |
| DAE (swap noise, train plus test) leading to an MLP blended with GBDT | Small; diversity | Medium to high (GPU) | Low if the MLP head is fitted inside the folds | Best "big" bet if time allows |
| DAE embeddings as GBDT features | About 0 | Medium | Low | Low priority |
| Distillation, GBDT ensemble to GBDT student | About 0 (compression) | Low | **High**: nested folds needed | Skip |
| Distillation, GBDT ensemble to NN student on train plus test, blended | Small | Medium | High without nested folds | Fold into the DAE/MLP bet |
| Co-training/tri-training | About 0 at 70k labels | Medium | High (confirmation bias) | Skip |
| SCARF/VIME pretraining | About 0 at full labels | High | Low | Skip |
| Segment-wise prior/offset at test time | Within noise; already hurt on public | Low | Overfits to OOF/public | Skip unless there is a mechanistic reason |

**Honest validation protocol for any of these**
- Hold a fixed 5-fold split and repeat with 3 seeds. Report the **paired** delta in OOF KS vs baseline with a bootstrap (resample rows, recompute both KS) and P(delta > 0). Also report AUC, because KS is a noisier statistic with a single threshold.
- Anything that uses labels to create test-side information (soft labels, pseudo-labels, target kNN, co-training) needs **nested cross-fitting**: the information used to train for fold k must come from models that never saw fold k's labels.
- Unsupervised steps (DAE, frequency, kNN without targets, adversarial classifier) can be fitted on all 100k rows, because no labels are involved.
- Do not accept a change based on public LB alone. With SD about 0.011, only OOF deltas confirmed by bootstrap across seeds are meaningful.

### Gaps
- No source quantifies these techniques on a synthetic credit-risk dataset of this size. All gain estimates in the table are my inferences from the cited evidence, not measured numbers.
- I did not verify whether this competition's test set is a random split or a time-forward split. That fact determines whether adversarial validation can find anything.
