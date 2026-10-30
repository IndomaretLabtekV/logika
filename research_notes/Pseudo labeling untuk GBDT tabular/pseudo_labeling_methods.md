# Pseudo-labeling and self-training for tabular GBDTs (LightGBM / XGBoost / CatBoost), binary classification

Context the notes are written against: credit-risk binary task, about 70k labeled rows, about 30k unlabeled test rows, about 13.6% positives, KS metric (rank-based, same family as AUC), current model a rank-average of 6 GBDTs with OOF KS 0.526, data probably synthetic.

Research budget note: about 15 tool calls. Kaggle notebook and discussion pages (for example Deotte's "Pseudo Labeling - QDA - [0.969]" and Santander discussion/89003) did not render through the fetch tool, so the Kaggle evidence below comes from secondary sources. Where I state practitioner knowledge I could not verify, it is placed under Inferences or Gaps and marked as such.

---

## Q1. What pseudo-labeling variants exist for tabular GBDT (hard/threshold, soft, weighting, iterative, test-time)?

### Takeaway
Four variants matter in practice: (a) hard labels on a confident subset, using a fixed probability threshold or a growing percentile curriculum; (b) soft labels, where the predicted probability is the target, trained with a cross-entropy objective that accepts targets in [0,1]; (c) a down-weight on pseudo rows; (d) iterative rounds with the student re-initialized each round. Academic tabular work (Kim et al. 2023, CAST) uses hard labels with a fixed threshold of 0.6 or a 20%-step curriculum, and re-weights confidence by a density/likelihood term. Kaggle practice (NVIDIA Grandmasters playbook) prefers soft labels with k fold-specific pseudo-label sets.

### Cited Findings
- **Basic recipe (Kaggle):** "You use your best model to infer labels on data that lacks them (for example, test data or external datasets)." Pseudo labels are added to the training data and the model is retrained, typically over several cycles (baseline, then pseudo-labeled, then improved pseudo-labels). Stronger teachers (ensembles) give better pseudo-labels, and multi-round pseudo-labeling usually beats a single pass — [NVIDIA, Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- **Soft vs hard:** "Using soft labels (probabilities instead of hard 0/1s) adds regularization and reduces noise." Soft pseudo-labels "add more signal, reduce noise, and let you filter out low-confidence samples." The same source suggests using pseudo-labels to pretrain and then to "Fine-tune on the initial data as a last step to reduce noise introduced earlier," which is mainly relevant to NNs. It frames pseudo-labeling as a close relative of knowledge distillation — [NVIDIA playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- **Kaggle Book definition and caveat:** pseudo-labeling adds "examples from the test set whose predictions you are confident about to your training set". It "may even worsen it if there is some added noise in the pseudo-labeled data", and whether it works in a competition cannot be known in advance. The book credits team Wizardry in Santander Customer Transaction Prediction with introducing it to Kaggle — [The Kaggle Book (Packt), via search excerpt](https://packtpub.com/en-eg/product/the-kaggle-book-9781801817479/chapter/modeling-for-tabular-competitions-9/section/pseudo-labeling-ch09lvl1sec52) (page returned 403 on fetch, so these quotes come from search snippets only)
- **Hard-label recipes in the tabular literature (Kim, Kim & Song 2023, "Revisiting Self-Training with Regularized Pseudo-Labeling for Tabular Data"):**
  - Fixed-threshold PL (FPL): confidence threshold 0.6, chosen empirically.
  - Curriculum PL (CPL): start at the top 20% most confident, add 20% per step until no unlabeled data remain.
  - Hard labels only.
  - Classifiers are re-initialized each cycle to limit accumulated confirmation bias, and the best cycle is kept by validation score.
  - Regularized score: f(x) = (αγ + 1)c / (α + 1). Here c is classifier confidence, γ is a min-max-scaled log-likelihood from a naive-Bayes-style density with continuous features binned into 10 bins, and α ∈ {0.25, 0.5, 0.75, 1.0}.
  - Models: XGBoost and LightGBM (plus TabNet, Saint, MLP), with Optuna at 100 trials and early-stopping patience 50.
  — [arXiv 2302.14013](https://arxiv.org/html/2302.14013v3)
- **CAST (Cluster-Aware Self-Training, arXiv 2310.06380), aimed explicitly at GBDTs:**
  - Regularized confidence c_r = α(γ∘c) + (1−α)c, where γ is a class-wise density prior: a KDE (CAST-D) or empirical likelihood (CAST-L) fit on labeled data over BorutaSHAP-selected features.
  - It plugs into FPL (τ = 0.6, stop when the self-trained classifier underperforms) or CPL (top 20%, +20% per iteration).
  - Re-initializes each round. α is grid-searched over [0.2, 0.75], and Appendix G suggests α ≤ 0.7.
  — [CAST, arXiv 2310.06380](https://arxiv.org/html/2310.06380)
- **Soft-label support in GBDT libraries:**
  - LightGBM `objective=cross_entropy` (alias `xentropy`): "label is anything in interval [0, 1]", "with optional linear weights".
  - LightGBM `cross_entropy_lambda` (`xentlambda`) is an "alternative parameterization of cross-entropy".
  - LightGBM `binary` "requires labels in {0, 1}". Weights "should be non-negative."
  — [LightGBM Parameters docs](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- **CatBoost** documents separate `Logloss` and `CrossEntropy` losses. CrossEntropy's formula uses a target t_i rather than a class c_i, which implies probabilistic targets. The fetched page did not state the allowed range explicitly — [CatBoost loss functions](https://catboost.ai/docs/en/concepts/loss-functions-classification)
- **Data-centric selection (DIPS, Seedat et al. 2024):** before each round, a selector keeps only "useful" labeled and pseudo-labeled rows: average confidence ≥ τ_conf and aleatoric uncertainty < τ_al, both computed over training checkpoints (for GBDT, over boosting iterations). It works with any pseudo-labeler, uses an XGBoost backbone in the synthetic experiment, and is "computationally cheap" — [DIPS, arXiv 2406.13733](https://arxiv.org/html/2406.13733)
- **Threshold choice in general:** "fixed thresholds usually lead to subpar results, highlighting the significance of dynamic thresholding for best results." Other schemes include class-wise average-prediction thresholds, a fixed proportion of top-confidence samples grown to 0.5, and percentile curricula. The survey describes the key trade-off as "between the number of examples going to be pseudo-labeled and the error they induce" — [Self-Training: A Survey, arXiv 2202.12040](https://arxiv.org/html/2202.12040v5)
- **Semi-supervised boosting (pseudo-labeling built into boosting):**
  - ASSEMBLE (Bennett, Demiriz & Maclin, KDD 2002) alternates between pseudo-labeling unlabeled data with the current ensemble and fitting the next base learner on labeled plus pseudo-labeled data. It is equivalent to maximizing margin on labeled and unlabeled data, works with any cost-sensitive base learner, and its decision-tree variant won the NIPS 2001 Unlabeled Data Competition.
  - SemiBoost (Mallapragada et al., TPAMI 2009) wraps any supervised learner and was "comparable to state-of-the-art" SSL on 16 datasets.
  — [ASSEMBLE paper](https://www.d.umn.edu/~rmaclin/publications/bennett-kdd02.pdf); [SemiBoost](https://typeset.io/pdf/semiboost-boosting-for-semi-supervised-learning-4arht7t8ta.pdf); family overview in [Exploiting Unlabeled Data to Enhance Ensemble Diversity, arXiv 0909.3593](https://arxiv.org/pdf/0909.3593)

### Inferences
- **Concrete variant menu for this task:**
  1. **Soft PL.** Append the 30k test rows with target = teacher probability. Train LightGBM `xentropy`, CatBoost `CrossEntropy`, or XGBoost `binary:logistic` with float labels.
  2. **Hard PL.** Add only test rows whose teacher score falls in the bottom q0 quantile (label 0) or the top q1 quantile (label 1).
  3. **Weighting.** Either variant with sample weight w ∈ {0.1, 0.3, 0.5, 1.0} on pseudo rows.
  4. **Rounds.** Repeat for 2–3 rounds with a re-initialized student each round.

  "Test-time training" in the NN sense has no direct GBDT analogue. For trees, the closest equivalent is transductive soft PL on the very rows being predicted.
- **XGBoost float labels (unverified in this session):** from memory, XGBoost's `binary:logistic` accepts float labels in [0,1] and rejects labels outside that range. This was not checked against the docs here; see Gaps.
- **Hard thresholds under 13.6% prevalence:** use class-specific quantiles rather than a symmetric 0.6 probability threshold, which is meaningless for an uncalibrated, imbalanced model. A large bottom slice is nearly pure negatives. A top slice of the same size will contain many negatives at a KS of about 0.53, roughly AUC 0.83–0.85. Taking, say, the bottom 40% as 0 and the top 3–5% as 1 also distorts the class prior of the augmented set. This argues for soft labels, or per-class weights that restore the 13.6% prior.
- **Soft PL vs distillation:** with soft labels on all test rows, soft PL is mathematically self-distillation / ensemble distillation into each member. Its main mechanism is smoothing each base learner toward the ensemble, plus giving the trees split points at test-set density. It is not "new label information."

### Gaps
- I did not retrieve the exact XGBoost docs line on float labels for `binary:logistic`, or explicit CatBoost text on the allowed [0,1] target range for CrossEntropy.
- No primary source was retrieved for team Wizardry's Santander recipe (thresholds, model, CV handling). Kaggle discussion/89003 was not fetched.
- I found no source giving a recommended numeric weight for pseudo rows in GBDTs.

---

## Q2. When does self-training help? Low-label vs plentiful-label regime, confirmation bias, calibration, imbalance

### Takeaway
Every quantitative tabular result found comes from a low-label regime: 10–30% of rows labeled, with hundreds to a few thousand labels. Gains there are modest even for XGBoost/LightGBM (about +0.1 to +3 accuracy/F1 points), and naive confidence selection sometimes hurts. The literature gives no evidence of material gains when labels are plentiful (70k labeled vs 30k unlabeled, ratio about 2.3:1) and the teacher is already a strong ensemble. The theory says unlabeled data helps only under cluster/compatibility assumptions, and label noise propagates into pseudo-labels.

### Cited Findings
- **Gains grow as labels shrink:** Kim et al. report larger gains with fewer labels (for example, Saint on 6-month mortality with 500 labels: R-FPL about +29%, R-CPL about +23% relative). The paper does not test self-training with fully labeled data — [arXiv 2302.14013](https://arxiv.org/html/2302.14013v3)
- **Even in low-label settings, GBDT gains are small and inconsistent:**
  - XGBoost Christine: 0.7071 → 0.7111.
  - Sylvine: 0.9290 → 0.9301.
  - MNIST: 0.9639 → 0.9646.
  - Plain CPL on 6M mortality *hurt* (0.5534 → 0.5479).
  - Christine showed no clear gain for the regularized methods.
  - The results come from single runs, so treat them as indicative — [arXiv 2302.14013](https://arxiv.org/html/2302.14013v3)
- **CAST (10% labeled, 10 seeds), XGBoost:**
  - 6M mortality F1 0.4055 → 0.4221 (vanilla FPL) → 0.4444 (CAST-L).
  - Diabetes 0.6613 → 0.6613 (vanilla FPL, no gain).
  - Vanilla CPL on 6M mortality hurt (0.4029 < 0.4055).
  - For neural SSL models, naive self-training gives "marginal improvements and sometimes makes performance worse" — [CAST](https://arxiv.org/html/2310.06380)
- **Calibration does not fix selection:** in CAST, calibration baselines (temperature scaling, histogram binning, spline, latent GP) showed "little to no distinction" from naive confidence for pseudo-label selection, and some were worse — [CAST](https://arxiv.org/html/2310.06380)
- **Plain trees are poor self-training base learners:** Tanha et al. found that standard decision-tree learning "cannot be effective in a self-training algorithm" because "the basic decision tree learner does not produce reliable probability estimation." Laplace correction, no-pruning, NBTree, grafting and distance-based fixes help. Tree *ensembles* (Random Subspace, Random Forest) gave an extra improvement — [Tanha, van Someren & Afsarmanesh, IJMLC](https://link.springer.com/article/10.1007/s13042-015-0328-7)
- **Theory:**
  - Unlabeled data helps under the cluster/smoothness assumption: "two examples in a high density region should have identical class labels." It is "beneficial for training only if such a relationship exists."
  - Ben-David et al. show no guaranteed sample-size advantage without strong class-conditional assumptions.
  - Singh et al. ("Unlabeled data: now it helps, now it doesn't") report cases of no effect.
  - The risk is "the snowball effects of cascading pseudo-labeling mistakes."
  — [Self-Training: A Survey](https://arxiv.org/html/2202.12040v5)
- **Confirmation bias mitigation:** re-initialize the classifier each round and regularize the selection with density — [arXiv 2302.14013](https://arxiv.org/html/2302.14013v3); [CAST](https://arxiv.org/html/2310.06380)
- **Label noise propagates:** noise in the labeled set propagates into pseudo-labels, and PL "barely improves upon the supervised baseline" under noise. DIPS-style filtering of hard/ambiguous rows restores gains; with about 30% label noise on synthetic data, DIPS gained "around +20%" over supervised and greedy PL — [DIPS](https://arxiv.org/html/2406.13733)
- **Imbalance:** Kim et al. include 6M mortality (about 3% positive) and Bank Marketing (about 12% positive, close to this task's 13.6%). Bank Marketing XGBoost F1 went 0.5635 → 0.5730 with R-CPL at 3,000 labels. Pretraining sometimes degraded the skewed datasets — [arXiv 2302.14013](https://arxiv.org/html/2302.14013v3)
- **Distribution mismatch (neural evidence, for contrast):** "Performance of SSL techniques can degrade drastically when the unlabeled data contains a different distribution of classes than the labeled data." A well-regularized supervised baseline is often very strong — [Oliver et al., NeurIPS 2018](https://arxiv.org/abs/1804.09170)

### Inferences
- **Expected gain for this task is small.** With 70k labels and only 30k unlabeled rows, the labeled set already pins down the decision function far better than in any of the studied settings. The best a priori expectation is a KS change within fold noise, roughly ±0.002–0.005.
- **Credit risk is a bad fit for the cluster assumption.** Credit-risk data is noisy-label by nature (default is stochastic given features), so the cluster/low-density assumption is weak. The task resembles the DIPS "noisy label" setting more than a clean-cluster one, and naive PL tends to fail there.
- **Where gains can still come from:**
  1. Regularizing each base GBDT toward the stronger ensemble (distillation). This gain may disappear once the models are rank-averaged anyway.
  2. Exposing trees to test-set covariate density, which helps if train and test differ in covariate distribution.
  3. In synthetic data, a test set generated by the same generator as the train set, so the joint structure is identical.
- **What would make PL more likely to help:** a train/test covariate shift, or a teacher that is much stronger than the individual students.
- **Calibration matters less than ranking for selection, but matters for soft targets.** Soft targets should be on a calibrated probability scale, not raw rank-averages. Rank-averages are uniform on [0,1] and would imply a ~50% positive rate instead of 13.6%. Map the rank-average back to probabilities, for example with isotonic regression fit on OOF, before using it as a soft target.

### Gaps
- No found paper evaluates GBDT self-training in a high-label regime (tens of thousands of labels, with an unlabeled set smaller than the labeled one).
- No found paper reports AUC/KS (rather than accuracy/F1) for GBDT pseudo-labeling.
- I did not retrieve Oliver et al.'s explicit numbers on how SSL gain varies with the number of labels.

---

## Q3. How to generate pseudo-labels so OOF validation stays honest; correct protocol to measure gain

### Takeaway
Pseudo-labels used to train the fold-k student must come only from teachers that never saw fold k's labels. The NVIDIA playbook states "you must compute k sets of pseudo-labels so that validation data never sees labels from models trained on itself." Fix the fold split before any pseudo-labeling. Report CV only on original labeled validation rows, and compare against a baseline that uses the same folds and seeds.

### Cited Findings
- "With k-fold … you must compute k sets of pseudo-labels so that validation data never sees labels from models trained on itself" — [NVIDIA playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- General CV hygiene: any preprocessing or feature selection must happen inside each training fold, not on the full dataset ahead of time. Otherwise validation information leaks into training — [codemia CV-in-LightGBM note](https://codemia.io/knowledge-hub/path/cross-validation_in_lightgbm) (secondary source)
- Kaggle community view: training on pseudo-labeled test rows is not label leakage per se, because no true test labels are used. The concern is overconfidence — [fast.ai forum thread](https://forums.fast.ai/t/pseudo-labeling-in-keras/461)
- A public TPS notebook trains LightGBM with pseudo-labels plus 5-fold stratified CV but does not explain how pseudo rows are handled per fold. Such notebooks cannot be trusted for gain estimates — [kaggler-tv AutoLGB pseudo-label notebook](https://nbviewer.jupyter.org/github/kaggler-tv/blog/blob/master/_notebooks/2021-04-18-autoencoder-pseudo-label-autolgb.ipynb)
- Academic protocol: Kim et al. use 3-fold stratified CV with a held-out labeled test split, and select the best self-training cycle on a validation split. CAST averages 10 seeds — [arXiv 2302.14013](https://arxiv.org/html/2302.14013v3); [CAST](https://arxiv.org/html/2310.06380)

### Inferences (recommended protocol for this task)
1. **Fixed folds.** Use the same 5 (or 10) stratified folds as the current v-series OOF, and the same seeds.
2. **Fold-specific teachers.** The existing CV already gives, for each fold k and each of the 6 GBDTs, a model trained on folds ≠ k. The teacher for fold k is the rank-average of those 6 fold-k models' *test* predictions, mapped to probabilities (isotonic fit on folds ≠ k OOF only).
   - Do **not** use the usual "average of all fold models" test prediction as the pseudo-labels for fold k. Four of those five models trained on fold k's labels, so fold-k label information leaks into the student through the test pseudo-labels.
   - The leak is usually small but inflates OOF in exactly the direction you are trying to measure.
3. **Fold-k student.** Train on (folds ≠ k, true labels) ∪ (30k test rows, fold-k pseudo-labels, weight w). Predict fold k, which gives the student OOF. Compute KS on the 70k original rows only.
4. **Transductive check (optional, more faithful).** At submission time the student predicts the same test rows it was trained on. To mimic this, also add fold k's *own* rows as unlabeled, with pseudo-labels from the fold-k teacher. That teacher never saw fold k, so the step is legal. Then score fold k.
   - This measures the transductive effect.
   - With soft labels it will mostly pull fold-k predictions toward the teacher. If the student-on-own-rows result is no better than simply blending teacher and student, there is no real gain.
5. **Fair baseline.** Compare against (a) the existing 6-model rank-average OOF and (b) a "teacher+student blend" with no PL. Run at least 3 seeds and report the mean and std of the KS delta per fold. With 70k rows and KS around 0.53, seed-to-seed noise in OOF KS is plausibly ±0.002–0.004, so treat smaller deltas as noise.
6. **Iterating.** Round 2 teachers must again be fold-specific: the fold-k student's test predictions from round 1. Never mix folds.
7. **Final submission.** Retrain each student fold with its fold-specific pseudo-labels, or use full-train teachers. The CV estimate only holds for the fold-specific construction, so keep the submission pipeline identical to the CV pipeline.
8. **Selection overfitting.** Tuning thresholds, weights and rounds on the same OOF adds optimistic bias. Hold out one fold, or use nested CV, for the final go/no-go if the measured gain is small.

### Gaps
- I found no published tabular study that explicitly quantifies how much the "average of all fold models" pseudo-label leak inflates OOF. The direction is clear; the magnitude is not documented.

---

## Q4. Empirical numbers: typical AUC/KS gains for tabular GBDT pseudo-labeling

### Takeaway
I found no reliable published AUC/KS gain for GBDT pseudo-labeling in a label-rich setting. Academic numbers are accuracy/F1 in low-label regimes: XGBoost/LightGBM gains of about +0.001 to +0.03 absolute, sometimes negative for naive variants. Kaggle write-ups list pseudo-labeling as a pipeline stage, often in synthetic Playground competitions, but give no isolated with-versus-without gain.

### Cited Findings
- **XGBoost (Kim et al., labels about 600–7,000):**

  | Dataset | Baseline | With self-training (best) |
  |---|---|---|
  | Albert (acc) | 0.6310 | 0.6629 (CPL) |
  | Jasmine (acc) | 0.7922 | 0.7993 |
  | Madeline (acc) | 0.6930 | 0.7261 |
  | Philippine (acc) | 0.7219 | 0.7311 |
  | Coupon (acc) | 0.7142 | 0.7250 |
  | Bank Marketing (F1) | 0.5635 | 0.5730 |
  | Splice (acc) | 0.9464 | 0.9531 |
  | MNIST (acc) | 0.9639 | 0.9646 |
  | LightGBM, 6M mortality (F1) | 0.5325 | 0.5379 |

  — [arXiv 2302.14013](https://arxiv.org/html/2302.14013v3)
- **XGBoost (CAST, 10% labels, 10 seeds):**
  - Ozone F1 0.2803 → 0.3133 (CPL CAST-L).
  - Cmc b-acc 0.4638 → 0.4811.
  - Diabetes 0.6613 → 0.6719.
  - Under 20% feature corruption, 6M mortality improved 0.4035 → 0.4277 (CAST-L).
  — [CAST](https://arxiv.org/html/2310.06380)
- **DIPS:** reaches the same performance with 60–70% fewer labeled examples for UPS and greedy PL on tabular data (12 datasets, 10% labeled). Variance across pseudo-labelers fell from 0.46 to 0.14 — [DIPS](https://arxiv.org/html/2406.13733)
- **Kaggle Playground S6E3, 1st place:** the pipeline includes "pseudo labeling" alongside stacking, hill climbing and 22 LightGBM models. No isolated PL gain is reported — [Kaggle S6E3 1st place writeup](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm) (search snippet only)
- **TPS Apr 2021 (synthetic Titanic):** a public CatBoost "Run Pseudo label" notebook exists — [Kaggle notebook](https://www.kaggle.com/code/gomes555/tps-apr2021-catboost-run-pseudo-label) (not fetched; no numbers)
- **NVIDIA playbook:** BirdCLEF 2024 soft pseudo-labels "helped our model generalize better", with no figures given — [NVIDIA playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)

### Inferences
- **Pattern from the unverified Kaggle cases.** The big Kaggle PL successes recalled from memory share a trait: the test set is an exact draw from a simple known generator, and the model class matches the generator. Instant Gratification (2019) used sklearn `make_classification` data with QDA plus PL on high-confidence test rows (thresholds near 0.01/0.99); the notebook title shows "[0.969]". Santander 2019 is the other example. These are not GBDT-on-noisy-credit settings. Instant Gratification details could not be verified this session.
- **Realistic prior.** For a 70k-label GBDT ensemble on a noisy credit target, expect between −0.003 and +0.005 KS. Any claimed gain above about 0.01 should be suspected as leakage (see the Q3 protocol).

### Gaps
- No primary-source AUC/KS numbers for GBDT PL on Kaggle tabular competitions were retrievable. Kaggle discussion/notebook bodies did not render.
- Deotte's Instant Gratification PL notebook (thresholds, AUC before and after) could not be fetched; the "0.969" in its title is unverified as to metric.

---

## Q5. Academic work on self-training for tree ensembles / semi-supervised boosting

### Takeaway
The tree-specific literature agrees on four points:
- Self-training with trees depends on probability-estimate quality. Single trees are bad, ensembles better.
- Selection must be regularized beyond raw confidence, for example by density/likelihood (Kim et al., CAST) or training-dynamics filtering (DIPS).
- Semi-supervised boosting (ASSEMBLE, SemiBoost) is effectively iterative pseudo-labeling inside the boosting loop. It was competitive in the early-2000s low-label benchmarks, but there is no modern evidence it beats a tuned supervised GBDT with abundant labels.

### Cited Findings
- **Tanha et al.:**
  - Standard decision trees are ineffective self-training base learners because of poor probability estimates.
  - NBTree, no-pruning + Laplace correction, grafting and distance-based measures "do not produce better performance when used on the labeled data only, but they do benefit more from the unlabeled data in self-training."
  - Random Subspace and Random Forest ensembles give "an extra improvement."
  — [IJMLC paper](https://link.springer.com/article/10.1007/s13042-015-0328-7)
- **Self-training for multi-target regression with tree ensembles:** proposes four reliability estimates based on ensemble mechanisms for selecting pseudo-labeled examples, treating prediction reliability as central — [Knowledge-Based Systems 241](https://www.datalearner.com/academic/journal-papers/0950-7051/volumes-and-issues/241/paper-detail/7417)
- **ASSEMBLE:** a margin-based semi-supervised boosting method that works with any cost-sensitive base learner. Its decision-tree variant won the NIPS 2001 unlabeled-data competition — [Bennett et al. KDD 2002](https://www.d.umn.edu/~rmaclin/publications/bennett-kdd02.pdf)
- **SemiBoost:** a boosting wrapper around any supervised learner, "comparable to state-of-the-art" SSL on 16 datasets — [SemiBoost](https://typeset.io/pdf/semiboost-boosting-for-semi-supervised-learning-4arht7t8ta.pdf)
- **Later extensions:** Chen & Wang add a local smoothness regularizer to ASSEMBLE "to induce a more reliable self-labeling process". MCSSB replaces exponential loss with Bregman divergence — [arXiv 0909.3593](https://arxiv.org/pdf/0909.3593)
- **Majority-vote ensembles** such as Random Forest and AdaBoost can set a pseudo-labeling threshold automatically from vote margins — [Self-Training: A Survey](https://arxiv.org/html/2202.12040v5)
- **CAST:** most recent semi/self-supervised tabular methods (VIME, SubTab, SCARF and others) are neural and not applicable to GBDTs, which motivates a GBDT-compatible confidence regularizer — [CAST](https://arxiv.org/html/2310.06380)

### Inferences
- **Selection signal for this ensemble.** With 6 GBDTs times folds times seeds already available, the natural "reliability" signal is inter-model disagreement on each test row: the std of ranks across models. A DIPS/CAST-style rule would keep or up-weight test rows where all models agree and down-weight rows with high disagreement or that sit in low-density regions.
- **Neural methods, for contrast.** VIME, SCARF and similar methods do not transfer to GBDTs and are not worth pursuing for this pipeline.

### Gaps
- I found no modern head-to-head benchmark of ASSEMBLE/SemiBoost against tuned LightGBM/XGBoost/CatBoost with large labeled sets.
- No academic paper retrieved tests self-training for LightGBM/CatBoost specifically with soft (cross-entropy) targets.
