# Pseudo-labeling and transductive tricks in top Kaggle tabular solutions (focus: credit risk)

Context for the reader: our task is credit-risk binary classification (90+ DPD in 12 months), metric KS, about 70k train and 30k test rows, a rank-averaged GBDT ensemble with OOF KS 0.526, and data that is probably synthetic.

Source-quality labels used below:
- [WRITEUP] = the winner's own Kaggle discussion or writeup post, read in full (through a text proxy because Kaggle pages render with JS).
- [BLOG] = a sponsor or engineering blog written by or about the winner (for example NVIDIA's blog about Chris Deotte).
- [SECONDARY] = a third-party summary or a search snippet, not checked against the original.
- None of the claims below were checked by running the winners' code. All numbers are as reported by the authors.

## 1. Did top solutions in the named competitions use pseudo-labeling? Details and gains

### Takeaway
Explicit pseudo-labeling on test is documented in only two places: the Santander 2019 1st place (hard labels on the most confident test rows, with a small gain) and the AmEx 2022 gold solution by Chris Deotte (soft-label distillation onto train+test, with a large NN gain but no gain over the LGBM teacher on its own). Home Credit's 1st place did not use it, and IEEE-CIS's 1st place does not mention it. In these competitions the transductive gains came mostly from features or post-processing computed on train+test (Santander counts, IEEE uid averaging, Porto Seguro DAE), not from pseudo-labels.

### Cited Findings

**Santander Customer Transaction Prediction (2019), 1st place, team Wizardry (fl2o et al.) [WRITEUP]**
- Pseudo-labels were hard labels on the extremes of the test ranking. The LGBM used "2700 highest predicted test points as 1 and 2000 lowest as 0". The NN used "5000 highest and 3000 lowest". The post mentions no multi-round iteration. — [Santander #1 Solution, discussion/89003](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)
- Measured gain, NN only: private LB 0.92497 without pseudo-labels and 0.92546 with them (+0.0005 AUC). No with/without comparison is given for LGBM. The LGBM with shuffle augmentation plus pseudo-labels scored 0.92522 public / 0.92332 private. The final blend (2.1 NN : 1 LGBM) scored about 0.927 public. — [same writeup](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)
- The larger transductive gain came from count/uniqueness features computed on "data + test (only including real test samples)". fl2o's comment: "Using data + true samples from test > data alone". Uniqueness computed with the full test set "worked really well on CV, giving >.92x results but didn't apply to test as is!" The fix was to exclude the synthetic ("fake") test rows identified in YaG320's kernel. — [same writeup](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)
- How fake rows were identified: a test row with at least one value that is unique within the test set is real, and a row with no unique values is synthetic. — [SECONDARY: KazukiOnodera repo script](https://github.com/KazukiOnodera/santander-customer-transaction-prediction/blob/master/py/990_2nd_place_solution_golf.py); [SECONDARY: Medium case study](https://medium.com/analytics-vidhya/santander-customer-transaction-prediction-an-end-to-end-machine-learning-project-2cb763172f8a)
- Data augmentation (shuffling feature values within each class, since the features are independent given the class): LGBM duplicated and shuffled target==1 rows 16 times and target==0 rows 4 times. The NN shuffled within class on the fly for every batch. — [WRITEUP](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)
- The Kaggle Book credits Wizardry's Santander post as a well-known introduction of pseudo-labeling for tabular data, and warns that it "may even worsen" results when the pseudo-labels are noisy. — [SECONDARY: Kaggle Book, pseudo-labeling section](https://packtpub.com/en-eg/product/the-kaggle-book-9781801817479/chapter/modeling-for-tabular-competitions-9/section/pseudo-labeling-ch09lvl1sec52)

**AmEx Default Prediction (2022), credit default**
- Chris Deotte's gold solution (the writeup title says 14th place; a third-party summary says 15th) [WRITEUP]: the teacher was Martin's LGBM, trained with nested K-fold. A Transformer+GRU was pretrained on train OOF and test predictions used as soft targets. Quote: "both train and test data are used for knowledge distillation". It was then fine-tuned on hard 0/1 train labels at a lower learning rate. Four cosine cycles alternated pretraining and fine-tuning. "The big help was pretraining with test predictions." — [Chris Deotte 14th place gold writeup](https://www.kaggle.com/competitions/amex-default-prediction/writeups/chris-deotte-14th-place-gold-nn-transformer-using-)
- Numbers from that writeup (AmEx metric): Transformer without distillation, CV 0.790. Transformer with distillation, CV 0.798, public 0.799, private 0.8076. LGBM teacher alone, CV 0.799, public 0.799, private 0.8070. 50/50 blend, CV 0.800, LB 0.801. So distillation brought the NN up to the GBDT's level (+0.008 CV). The ensemble gain over the GBDT alone was only about +0.001. — [same](https://www.kaggle.com/competitions/amex-default-prediction/writeups/chris-deotte-14th-place-gold-nn-transformer-using-)
- Warnings from the same comment thread: Jake called pseudo-labeling/distillation "a tight-rope walk for leakage and over-fitting". Nischay Dhankhar said he usually sees large CV gains from distillation but smaller LB gains. Chris said the nested CV is what keeps the CV estimate leak-free. — [same](https://www.kaggle.com/competitions/amex-default-prediction/writeups/chris-deotte-14th-place-gold-nn-transformer-using-)
- 13th place (CI-2-K / Uncle Giba): a snippet says they trained on soft labels only for confident predictions (about <0.1 or >0.8), and that only one or two pseudo-labeled models were in the final ensembles. No gain figure. NOT verified against the full post. — [SECONDARY: search snippet of 13th place writeup](https://www.kaggle.com/competitions/amex-default-prediction/writeups/ci-2-k-uncle-giba-13th-place-gold-solution)
- 1st place (daishu): the post mentions no pseudo-labeling or distillation. It says features were selected "when cv and lb boosting", meaning public-LB feedback guided feature selection. Third-party summaries describe the post as "not detailed". — [daishu 1st place post, discussion/348111](https://www.kaggle.com/competitions/amex-default-prediction/discussion/348111); [SECONDARY: BulletTech summary](https://bullettech.github.io/BulletTech/Main_Course/Machine_Learning/2022-08-30-AMEX-Kaggle-Summary/)
- The repo mirrored as the "1st solution" (mlcontests) has a README with only run instructions, and a GitHub code search for "pseudo" in it returned nothing. — [mlcontests repo](https://github.com/mlcontests/Kaggle-American-Express-Default-Prediction-1st-solution)

**Home Credit Default Risk (2018), 1st place [WRITEUP]**
- No pseudo-labeling. Adversarial validation found that train and test could be separated with over 0.98 AUC, but "We discovered this towards the end, but were not able to make any use of it." The team speculated that "some of the top teams were in fact exploiting some kind of adversarial validation/ pseudolabeling." — [Home Credit 1st place, discussion/64821](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64821)
- What failed: imputing the EXT_SOURCE features with a model (the imputed features reached 0.78x AUC) "did not help our base models, either for CV or on public LB." — [same](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64821)

**IEEE-CIS Fraud Detection (2019), 1st place (Chris Deotte & Konstantin Yakovlev) [BLOG]**
- Pseudo-labeling is not mentioned. The transductive trick was post-processing: each test prediction was replaced with the average prediction for its client UID. No separate gain figure is given. Adding UID-aggregated features raised local AUC from 0.9363 to 0.9472. Final scores: public 0.9677, private 0.9459. Validation used GroupKFold with months as groups. Feature selection dropped 19 features whose AUC was ≤0.5 when trained on the first month and tested on the last. — [NVIDIA blog on the IEEE 1st place solution](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/)
- A third-party summary of the 1st place post says the core idea was client identification (uid) built from card, D, C and V columns. — [SECONDARY: eda-ai-lab summary](https://eda-ai-lab.tistory.com/448)

**Porto Seguro Safe Driver (2017), 1st place (Michael Jahrer)**
- The winner trained a denoising autoencoder with "swap noise" (input values replaced by values from other rows) and used it for representation learning. The final blend was 1 LGBM + 5 NNs. A commentator notes the technique "requires stacking together training and test data", i.e. the DAE is fit on unlabeled train+test. — [SECONDARY: Hybrid Bootstrap paper](https://arxiv.org/pdf/1801.07316); [SECONDARY: fast.ai forum thread](https://forums.fast.ai/t/porto-seguro-winning-solution-representation-learning/8499?page=2); [SECONDARY: Kaggle Workbook](https://packt-plus-non-live.prod.packtpub.com/en-ar/product/the-kaggle-workbook-9781804611210/chapter/the-most-renowned-tabular-competition-porto-seguros-safe-driver-prediction-1/section/examining-the-top-solution-ideas-from-michael-jahrer-ch01lvl1sec04)

**Instant Gratification (2019, synthetic data from sklearn make_classification)**
- Chris Deotte's public notebook "Pseudo Labeling - QDA - [0.969]" (best score shown: 0.97059). I could not extract the procedure or the with/without-pseudo-label numbers from the page. — [Kaggle notebook](https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969)
- 6th place: the key was to cluster each class (GMM, 3 clusters per class) before modeling. — [SECONDARY: search snippet](https://developer.nvidia.com/blog/competition-and-community-insights-from-nvidias-kaggle-grandmasters)

### Inferences
- The documented gains from pseudo-labels on test are small for GBDTs. Santander showed about +0.0005 AUC on private (NN). In AmEx the big gain went to a weak NN student (+0.008), and the blend gained about +0.001 over the GBDT teacher. With ~30k test rows and a GBDT ensemble, a realistic expectation for us is "zero to tiny". A KS gain of +0.001 to 0.002 is within noise on 30k rows.
- Santander worked partly because of a special structure: features independent given the class, and fake test rows. Our data has neither, as far as we know.
- In credit-risk competitions, joint train+test computation (counts, uid aggregates, DAE) helped more than pseudo-labels. Every case relied on a specific structural property: an identifiable client id, value uniqueness, or row-level noise.

### Gaps
- I could not open the full posts for IEEE-CIS (discussion/111284) and AmEx 2nd/3rd place, so I cannot confirm whether they tried pseudo-labeling and rejected it.
- Give Me Some Credit (2011): I found no source on top-solution transductive tricks.
- AmEx 13th place pseudo-label thresholds come from a search snippet only.

## 2. Playground Series (synthetic data from a deep generative model): what worked best

### Takeaway
Recent winning Playground solutions rely on large OOF libraries with stacking or hill-climbing (AutoGluon, Ridge, LGBM meta-models) plus the original dataset. Pseudo-labeling appears in Grandmaster tips, but in the sources found no winner reports measured pseudo-labeling gains on Playground data. The only measured number on synthetic Playground data is a tiny self-distillation CV gain in regression.

### Cited Findings
- S5E8 (Bank Marketing, binary, AUC), 1st place "JAPE" by Optimistix [WRITEUP]: AutoGluon multi-layer stacking was the main ensembler and beat Ridge, hill-climbing and LGBM. Some models were run "with and without the original" dataset (+18 OOFs). Pseudo-labeling was NOT used; it is listed among things he "didn't get around to including". Label "flipping" post-processing was applied routinely. Final 136-OOF ensemble: public 0.97844 / private 0.97798 (flipped: 0.97801). OOF pruning lowered CV slightly but improved LB in one run. — [S5E8 1st place, discussion/603210](https://www.kaggle.com/competitions/playground-series-s5e8/discussion/603210)
- S5E8 2nd and 3rd place titles: "Yet another ensemble" and "OOF Stacking + AutoGluon". — [S5E8 discussion list](https://www.kaggle.com/competitions/playground-series-s5e8/discussion)
- NVIDIA Grandmasters Playbook [BLOG]: 1st places in Predict Calorie Expenditure (hill-climbing over XGB/CatBoost/NN/linear), Podcast Listening Time (3-level stack) and the April 2025 Playground (3-level stack). Rainfall binary got 2nd place with a GBDT+NN+SVR ensemble and no feature engineering. Fertilizer: a 100-seed XGB ensemble gave about 0.379 MAP@3 vs about 0.376 for single-seed averages. The pseudo-labeling section's example is BirdCLEF 2024 (audio), not a tabular Playground. — [NVIDIA Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- Playground S6E1 (Student Test Scores, regression), Chris Deotte thread [WRITEUP/discussion]: his recipe runs inside each fold. Model 1 is trained on the training fold and predicts the validation fold and test. Model 2 is trained on train+val+test using those predictions as labels. Chris argues the gain is from "more features not more targets". He cites his AmEx gold as using this. He gave no numbers and said he hadn't "pushed it to its limit here". — [S6E1 discussion/666888](https://www.kaggle.com/competitions/playground-series-s6e1/discussion/666888)
- Same thread, measured result (user "broccoli beef", XGB 5-fold, self-distillation without test rows): CV RMSE went from 8.7686 (round 0) to 8.7605 (round 1) and 8.7598 (round 2, best), then crept back up to 8.7611 by round 5. Other users: a hill-climb with pseudo-labels raised CV "by a bit", and the bigger gain came from feature engineering. One user (mjmagno) saw gains that "shrank after hyperparameter tuning". — [same](https://www.kaggle.com/competitions/playground-series-s6e1/discussion/666888)
- Playground S4E10 (Loan Approval) train and test were generated by a deep learning model trained on the original Loan Approval dataset. — [SECONDARY: tutorial page](https://nosratabadi.github.io/Data-Science-Capstone/loan_approval_prediction.html)
- March 2026 Playground (churn): the 1st place used LLM agents to build a 4-level stack of 150 models selected from 850. No mention of pseudo-labeling. — [SECONDARY: NVIDIA blog search snippet](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/)

### Inferences
- On synthetic Playground data, the methods that appear repeatedly among winners are a large diverse OOF library, a strong stacker or hill-climb, seed averaging, and the original data as extra rows or as a flag. Pseudo-labeling is at most a minor add-on that the 1st place skipped (S5E8).
- Rank averaging is not named in the winning writeups found. Stacking and hill-climbing on OOFs are the norm. For a rank metric like KS, rank averaging is a reasonable default, but no source here shows it beating a stacker.
- Pseudo-label gains that "shrank after hyperparameter tuning" (S6E1) match the AmEx pattern: the technique mainly helps weak or under-tuned students.

### Gaps
- No Playground binary-classification winner found with a measured pseudo-labeling gain or loss.
- No Playground credit/loan-default winner writeup (for example S4E10 Loan Approval, S4E1 Bank Churn) was retrieved, so I cannot confirm the effect of the original data there.

## 3. Reported failures, cases with CV gain but LB/private loss, and other transductive tricks

### Takeaway
The main failure mode is leakage: pseudo-labels or test-derived features computed without nested folds inflate CV ("large CV gains, smaller LB gains"). Santander also showed that computing transductive features over contaminated test rows looked great on CV but did not transfer. Adversarial validation is used mainly as a diagnostic. Home Credit's winner found strong train/test separability and still could not use it.

### Cited Findings
- AmEx: "a tight-rope walk for leakage and over-fitting". Nischay Dhankhar reported "large CV gains" from distillation "but smaller LB gains". Chris attributed his CV/LB agreement (0.798 CV vs 0.799 public) to nested K-fold for the teacher. — [AmEx 14th place writeup](https://www.kaggle.com/competitions/amex-default-prediction/writeups/chris-deotte-14th-place-gold-nn-transformer-using-)
- Leakage control recommended by the NVIDIA playbook: generate k separate sets of pseudo-labels so that a validation fold never receives labels from a model trained on that fold. Use soft labels, because they "add regularization, reduce noise" and allow filtering low-confidence rows. Stronger teachers (ensembles, multi-round) give better pseudo-labels. — [NVIDIA Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- Chris Deotte (S6E1): use hard labels only for high-confidence predictions, while soft labels can use all rows. A confidence threshold is worth tuning as a hyperparameter. — [S6E1 discussion/666888](https://www.kaggle.com/competitions/playground-series-s6e1/discussion/666888)
- Santander: uniqueness features over the full test set gave ">.92x" CV "but didn't apply to test as is". Synthetic test rows had to be removed first. — [Santander #1](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)
- Home Credit: adversarial validation AUC >0.98 between train and test, "not able to make any use of it". Imputation features from an auxiliary model did not help CV or LB. — [Home Credit 1st](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64821)
- Other transductive tricks used by winners:
  - Features computed on train+(real) test: Santander count/uniqueness features. — [Santander #1](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)
  - Per-entity prediction averaging on test: IEEE uid. — [NVIDIA IEEE blog](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/)
  - Unsupervised representation learning on train+test: Porto Seguro DAE. — [SECONDARY: fast.ai thread](https://forums.fast.ai/t/porto-seguro-winning-solution-representation-learning/8499?page=2)
  - Distillation of a GBDT into an NN using test predictions: AmEx. — [AmEx 14th](https://www.kaggle.com/competitions/amex-default-prediction/writeups/chris-deotte-14th-place-gold-nn-transformer-using-)
  - Time-consistency feature screening (train on the first month, test on the last): IEEE. — [NVIDIA IEEE blog](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/)
  - Public-LB feedback in feature selection: AmEx 1st (daishu). This adapts to the public test split. — [discussion/348111](https://www.kaggle.com/competitions/amex-default-prediction/discussion/348111)

### Inferences (applied to our task: GBDT ensemble, OOF KS 0.526, 70k/30k, synthetic)
- If we try pseudo-labeling, do it the leak-free way. Inside each outer fold, the teacher is trained only on that fold's training part. Test pseudo-labels for fold k come from fold-k teachers only. Use soft labels, or hard labels on the extreme tails only, Santander-style (for example the top/bottom few percent of the test rank). Judge the result on the outer OOF KS, not on in-fold metrics.
- Expect at most about +0.001 to 0.003 KS. The documented GBDT-teacher→GBDT-student cases show negligible gains. The clear wins were (a) weak students (NN) and (b) structural transductive features. A KS standard error on about 30k rows with ~a few % positives is likely of the same order, so a single public-LB delta cannot confirm a gain.
- The cheaper, better-documented transductive diagnostics: adversarial validation train vs test (does the synthetic generator shift the distribution?), and frequency/count features computed on train+test (Santander-style). In a synthetic generator, value frequencies can carry signal (repeated values copied from the original data), but they need nested-CV checking.
- A Santander-style "real vs synthetic" split does not apply unless our test contains injected fake rows. Adversarial validation would reveal that.

### Gaps
- No public, quantified case found of pseudo-labeling raising CV but lowering private LB in a credit competition. The evidence is qualitative (comments in the AmEx thread).
- No found source measures pseudo-labeling with KS as the metric.
