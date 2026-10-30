# Data Dictionary and Project Guide

This file explains what every column in the clean data means, how the original BRFSS codes were
changed, and what each notebook does and why.

The original data is the Kaggle version of the 2023 BRFSS survey. It still uses the CDC answer codes,
so the meaning of each code was checked against the official 2023 BRFSS codebook:
https://www.cdc.gov/brfss/annual_data/annual_2023.html

The question wording below is a short summary. Check the exact wording in the CDC codebook before
quoting it in the paper.

---

## 1. Special codes used by BRFSS

BRFSS does not leave answers empty. It uses codes instead:

| Code | Meaning | What we did |
|------|---------|-------------|
| 7 or 77 | Don't know / not sure | Set to missing |
| 9 or 99 | Refused (or missing for calculated variables) | Set to missing |
| 88 | "None" (for the health days questions) or "no insurance" (for the insurance question) | Recoded to a real value (0 days / no insurance) |
| 14 | Don't know / refused for the age group variable | Set to missing |
| blank | Question not asked or no answer | Missing |

Missing values are **not** removed. They are filled in later inside the model pipeline (notebook 05),
using the median of the training data, and an extra 0/1 column is added to say the value was missing.

---

## 2. Target: `diabetes`

**BRFSS variable:** `DIABETE4`, "Has a doctor, nurse or other health professional ever told you that you have diabetes?"

| Original code | Meaning | Number of people | Clean value |
|---------------|---------|------------------|-------------|
| 1 | Yes | 59,786 | **1** (diabetes) |
| 2 | Yes, but only during pregnancy | 3,253 | removed |
| 3 | No | 358,706 | **0** (no diabetes) |
| 4 | No, prediabetes or borderline diabetes | 10,594 | removed |
| 7 | Don't know | 683 | removed |
| 9 | Refused | 296 | removed |
| blank | Missing | 5 | removed |

- 14,831 records were removed, leaving **418,492 records**.
- 14.3% of the clean data has diabetes (1) and 85.7% does not (0).
- Pregnancy-only diabetes was removed because it is not the long-term diabetes we want to predict.
- Prediabetes was removed so the outcome is a clean yes/no. Counting prediabetes as "no" would put
  people who are at higher risk into the negative class.

### Dropped column: `DIABTYPE`

"What type of diabetes do you have?" (1 = type 1, 2 = type 2). This is only asked to people who already
said they have diabetes, so it is empty for everyone else. Keeping it would tell the model the answer
(target leakage), so it was removed.

---

## 3. Features (18 columns)

### Yes / no features

For all of these, **1 = yes** and **0 = no**.

| Clean column | BRFSS variable | Question (short) | Original codes | Yes in clean data | Missing |
|--------------|----------------|------------------|----------------|-------------------|---------|
| `high_bp` | `_RFHYPE6` | Ever told by a health professional that you have high blood pressure | 1 = No, **2 = Yes**, 9 = missing | 40.5% | 0.39% |
| `high_chol` | `_RFCHOL3` | Had cholesterol checked and told it was high | 1 = No, **2 = Yes**, 9 = missing, blank = never checked | 41.6% | 12.76% |
| `chol_check` | `_CHOLCH3` | Had cholesterol checked in the last 5 years | 1 = checked in last 5 years, 2 = not in last 5 years, 3 = never, 9 = missing | 90.5% | 6.11% |
| `smoker` | `SMOKE100` | Smoked at least 100 cigarettes in your whole life | 1 = Yes, 2 = No, 7, 9 | 38.6% | 5.20% |
| `stroke` | `CVDSTRK3` | Ever told you had a stroke | 1 = Yes, 2 = No, 7, 9 | 4.2% | 0.28% |
| `heart_disease` | `_MICHD` | Ever had coronary heart disease or a heart attack | 1 = Yes, 2 = No | 8.4% | 0.97% |
| `phys_activity` | `EXERANY2` | Any physical activity or exercise in the past month, apart from your job | 1 = Yes, 2 = No, 7, 9 | 75.4% | 0.28% |
| `cost_barrier` | `MEDCOST1` | In the past 12 months, needed a doctor but could not afford it | 1 = Yes, 2 = No, 7, 9 | 8.5% | 0.33% |
| `diff_walk` | `DIFFWALK` | Serious difficulty walking or climbing stairs | 1 = Yes, 2 = No, 7, 9 | 15.9% | 4.10% |
| `has_insurance` | `PRIMINS1` | Has any health insurance | 1 to 10 = a type of insurance, **88 = no insurance**, 77, 99 | 94.5% | 4.29% |

Notes:

- **`high_bp` and `high_chol` are coded the other way round in BRFSS** (1 = No, 2 = Yes), because they are
  CDC calculated "risk factor" variables. We changed them so that 1 = yes like the others.
- **`chol_check`**: codes 2 and 3 both became 0, so 1 means "checked in the last 5 years" and 0 means
  "not checked in the last 5 years or never checked".
- **`high_chol` missing values are not random.** Most come from people who never had their cholesterol
  checked, so they could not be told it was high (see notebook 03). This is why the model gets a
  "was missing" indicator column.
- **`has_insurance`**: the original question asks for the main *type* of insurance (employer, Medicare,
  Medicaid, military, etc.). We only kept whether the person has any insurance or not.

### Ordered (ordinal) features

These keep their numbers because the order means something (a higher number is more or worse).

**`gen_health`** (`GENHLTH`): "Would you say that in general your health is ..."

| Value | Meaning |
|-------|---------|
| 1 | Excellent |
| 2 | Very good |
| 3 | Good |
| 4 | Fair |
| 5 | Poor |

7 and 9 were set to missing (0.28% missing). Higher = worse health.

**`age_group`** (`_AGEG5YR`): 5 year age groups calculated by the CDC.

| Value | Age | Value | Age |
|-------|-----|-------|-----|
| 1 | 18-24 | 8 | 55-59 |
| 2 | 25-29 | 9 | 60-64 |
| 3 | 30-34 | 10 | 65-69 |
| 4 | 35-39 | 11 | 70-74 |
| 5 | 40-44 | 12 | 75-79 |
| 6 | 45-49 | 13 | 80 or older |
| 7 | 50-54 | | |

14 (don't know / refused) was set to missing (1.77% missing).

**`education`** (`_EDUCAG`): highest level of school completed.

| Value | Meaning |
|-------|---------|
| 1 | Did not graduate high school |
| 2 | Graduated high school |
| 3 | Attended college or technical school |
| 4 | Graduated from college or technical school |

9 was set to missing (0.51% missing). Higher = more education.

**`income`** (`_INCOMG1`): yearly household income from all sources.

| Value | Income (USD) |
|-------|--------------|
| 1 | Less than 15,000 |
| 2 | 15,000 to less than 25,000 |
| 3 | 25,000 to less than 35,000 |
| 4 | 35,000 to less than 50,000 |
| 5 | 50,000 to less than 100,000 |
| 6 | 100,000 to less than 200,000 |
| 7 | 200,000 or more |

9 (don't know / refused / missing) was set to missing. This is the feature with the most missing
values (19.92%).

### Number features

| Clean column | BRFSS variable | Meaning | What we changed | Missing |
|--------------|----------------|---------|-----------------|---------|
| `bmi` | `_BMI5` | Body mass index (kg/m²), calculated by the CDC from height and weight | BRFSS stores it multiplied by 100 (2740 = 27.40), so it was divided by 100 | 9.30% |
| `mental_health_days` | `MENTHLTH` | Days in the past 30 days when mental health was not good (stress, depression, emotions) | 88 ("none") became 0; 77 and 99 became missing; range 0-30 | 1.82% |
| `physical_health_days` | `PHYSHLTH` | Days in the past 30 days when physical health was not good (illness, injury) | 88 ("none") became 0; 77 and 99 became missing; range 0-30 | 2.43% |

### Sex

**`sex`** (`_SEX`): sex calculated by the CDC.

| Original code | Meaning | Clean value |
|---------------|---------|-------------|
| 1 | Male | **1** |
| 2 | Female | **0** |

So `sex` really means "is male". 47.4% of the clean data is male. There are no missing values.
This choice of 1 = male is only a label and does not change the models; for the plots and tables
it is turned back into the words "Male" and "Female".

---

## 4. Columns that are created later (not saved in the clean file)

| Column | Made in | Meaning |
|--------|---------|---------|
| `bmi_category` | notebook 03 | BMI groups for plots: <18.5 (underweight), 18.5-25 (normal), 25-30 (overweight), 30-35, 35-40, 40+ (obese classes) |
| `age_band` | notebooks 03 and 04 | Wider age groups for the subgroup analysis: 18-34 (age groups 1-3), 35-49 (4-6), 50-64 (7-9), 65+ (10-13). The 5 year groups are too small at young ages (the 18-24 group has only 305 diabetes cases) |
| `sex_label` | notebooks 03 and 04 | "Male" / "Female" text version of `sex` |
| `missingindicator_<column>` | notebook 05 (inside the pipeline) | 0/1 column added by `SimpleImputer(add_indicator=True)` that says the original value was missing. Only added for columns that had missing values in the training data |

---

## 5. What each notebook does

| Notebook | What it does | Why |
|----------|--------------|-----|
| `01_download_data` | Downloads the CSV from Kaggle into `data/raw/` | Gets the 2023 BRFSS data (433,323 records, 20 columns) |
| `02_data_cleaning` | Creates the `diabetes` target, drops `DIABTYPE`, turns special codes into missing, recodes every column as described above, saves `data/processed/diabetes_clean.csv` | The raw file still uses CDC codes, which the models cannot use directly |
| `03_eda` | Class balance, missing values, BMI distribution, diabetes rate for each feature, by age and sex, by income and education, correlation heatmap | To understand the data before modelling and to make figures for the paper |
| `04_split_data` | Splits the clean data into training (60%), calibration (20%) and test (20%) sets, stratified on `diabetes`, with `random_state=42` | Each set has the same 14.3% diabetes rate. The calibration set is kept apart so that Platt scaling and isotonic regression are fitted on data the models have not seen |
| `05_train_models` | Trains 4 models with 3 imbalance settings (12 runs), tunes them with 5-fold cross-validation, saves the models and predicted probabilities | Answers RQ1 (which model) and gives the data for RQ2 (effect of imbalance correction) |

### Data split

| Set | Records | Diabetes cases | Used for |
|-----|---------|----------------|----------|
| Training | 251,095 | 35,872 | Training and tuning the models (cross-validation) |
| Calibration | 83,698 | 11,957 | Fitting Platt scaling and isotonic regression |
| Test | 83,699 | 11,957 | Final results only, used once |

---

## 6. Model and run names

The 12 runs are named `<model>_<setting>`. These names are used for the files in `models/` and for
the columns in `results/predictions/`.

| Short name | Model |
|------------|-------|
| `lr` | Logistic Regression |
| `dt` | Decision Tree |
| `rf` | Random Forest |
| `xgb` | XGBoost |

| Setting | Meaning |
|---------|---------|
| `none` | No imbalance correction (the reference) |
| `weighted` | Class weighting: the diabetes class gets about 6 times more weight. `class_weight="balanced"` for LR, DT and RF, `scale_pos_weight=6.0` for XGBoost |
| `oversampled` | Random oversampling: diabetes records in the training folds are copied until both classes are the same size |

For example `xgb_weighted` is XGBoost with class weighting.

### Pipeline steps inside every model

1. **Imputer**: fill missing values with the median of the training folds and add the 0/1 "was missing" columns.
2. **Scaler**: standardise every column (Logistic Regression only, tree models do not need it).
3. **Oversampler**: copy diabetes records (only for the `oversampled` setting, only on the training folds).
4. **Model**.

Because all of these are inside one pipeline, they are fitted on the training folds only, so nothing
from the validation, calibration or test data leaks into training.

---

## 7. Output files

| File | Made by | Contents |
|------|---------|----------|
| `data/raw/Diabetes Health Indicators.csv` | 01 | Original Kaggle data |
| `data/processed/diabetes_clean.csv` | 02 | Clean data, 418,492 rows, 19 columns (target + 18 features) |
| `data/processed/train.csv`, `calibration.csv`, `test.csv` | 04 | The three data sets |
| `results/figures/fig_*.png` / `.pdf` | 03 | Figures for the paper (sized for IEEE two column format) |
| `results/tables/table_feature_summary.csv` | 03 | Mean of each feature for people with and without diabetes |
| `results/tables/table_binary_feature_rates.csv` | 03 | Diabetes rate for yes and no answers of each yes/no feature |
| `results/tables/table_correlation_with_diabetes.csv` | 03 | Spearman correlation of each feature with diabetes |
| `results/tables/table_subgroups.csv` | 03 | People, cases and diabetes rate for each age band and sex (all data) |
| `results/tables/table_split_sizes.csv` | 04 | Size and diabetes rate of each data set |
| `results/tables/table_test_subgroups.csv` | 04 | People and cases for each age band and sex in the test set |
| `results/tables/table_cv_results.csv` | 05 | Cross-validation ROC-AUC, log loss and best settings for the 12 runs |
| `models/<run>.joblib` | 05 | The 12 trained pipelines |
| `results/predictions/calibration_predictions.csv`, `test_predictions.csv` | 05 | Predicted diabetes probability from each of the 12 runs, plus `diabetes`, `age_group` and `sex` |

---

## 8. What calibration means

### The idea

A model does not only say "diabetes" or "no diabetes". It gives a **probability**, for example
"this person has a 30% risk of diabetes".

A model is **well calibrated** when these probabilities can be taken literally:
if we take all the people the model gives a 30% risk, about 30% of them should really have diabetes.
If only 10% of them have diabetes, the model **overestimates** risk. If 50% have diabetes, it
**underestimates** risk.

### Calibration is not the same as discrimination

- **Discrimination** (measured with ROC-AUC) asks: does the model give higher risks to people with
  diabetes than to people without? It is only about the *order* of the risks.
- **Calibration** asks: are the risk *values* themselves correct?

A model can rank people very well (high ROC-AUC) and still give risks that are much too high or too
low. For example, if every predicted risk is doubled, the order stays the same, so ROC-AUC does not
change, but every risk is now wrong.

This matters for screening because a health service uses the actual number (for example "invite
everyone above 20% risk"), not only the order.

### Why this project uses calibration

1. **The risk number is what people act on.** In screening, a health service decides whom to invite
   for a blood test based on the predicted risk, for example "everyone above 20%". The Streamlit app
   will also tell a person "your risk is 23%". If the model overestimates risk, too many healthy people
   are sent for tests and people are worried for no reason. If it underestimates risk, people with
   diabetes are missed.

2. **ROC-AUC alone cannot show this problem.** As explained above, a model can have a good ROC-AUC
   and still give wrong risks. So a study that reports only accuracy or ROC-AUC does not tell us if
   the risk scores can be trusted.

3. **Imbalance corrections are known to damage calibration.** Many diabetes studies use oversampling
   or SMOTE because the diabetes class is small. Van den Goorbergh et al. and Carriero et al. showed
   that this makes models overestimate risk without improving ROC-AUC. Our own results in notebook 05
   show the same thing (see the example below).

4. **Earlier BRFSS diabetes studies did not check it.** The studies in the literature review
   (Muhammad et al., Li et al., Islam et al.) report accuracy, ROC-AUC, precision, recall or F1,
   but none of them report calibration. This is the research gap of the project.

5. **Reporting guidance asks for it.** TRIPOD+AI, the current guideline for reporting prediction
   models that use machine learning, says calibration must be reported together with discrimination.
   Collins et al. and Riley et al. add that prediction models more often fail on calibration than
   on discrimination.

6. **A model can be calibrated overall but not for some groups.** The risks may be right on average
   but too high for young people or too low for older men, for example. This is why calibration is
   also checked for each age band and sex (RQ4).

7. **SHAP explains the risk number.** If the number is wrong, the explanation gives the user false
   confidence in it. So SHAP is applied to the best calibrated model.

**Why two correction methods?** Platt scaling is simple and stable but can only fix a simple
S-shaped error. Isotonic regression is flexible and can fix more complex errors, but it can overfit
when there is little data. Comparing both shows which kind of correction these models need. The large
calibration set in this project (83,698 records) is big enough for isotonic regression to be used safely.

### Example from this project (notebook 05)

| Setting | Mean predicted risk on the test set | Real diabetes rate |
|---------|-------------------------------------|--------------------|
| No correction | about 14.4% | 14.3% |
| Class weighting | about 38-39% | 14.3% |
| Oversampling | about 37-39% | 14.3% |

ROC-AUC was almost the same for all three settings, but class weighting and oversampling make the
models predict about 2.7 times too much risk. This is exactly the kind of problem calibration
measures and that ROC-AUC cannot see.

### How calibration is measured (notebook 06)

| Measure | What it is | Good value |
|---------|------------|------------|
| **Brier score** | Mean of (predicted probability - real outcome)², where the real outcome is 1 or 0 | Lower is better |
| **Calibration intercept** | Shows if risks are too high or too low on average | 0. Negative = overestimates risk, positive = underestimates risk |
| **Calibration slope** | Shows if risks are too extreme or too modest | 1. Below 1 = too extreme (high risks too high, low risks too low), above 1 = too modest |
| **Calibration curve** | Plot of mean predicted risk (x) against the real diabetes rate (y) in 10 risk groups | Points on the diagonal line |

The intercept and slope come from a logistic regression of the real outcome on the logit of the
predicted probability, as recommended in the TRIPOD+AI and Riley et al. guidance.

### How calibration is corrected (notebook 07)

A model's probabilities can be corrected after training with a second, small model that maps the old
probabilities to new ones:

- **Platt scaling**: fits a logistic (S-shaped) curve. Simple, works well when the error has a
  simple shape.
- **Isotonic regression**: fits a flexible step function that can only go up. Can fix more complex
  errors but needs more data.

Both are fitted on the **calibration set**, which the models never saw during training. If they were
fitted on the training data they would learn the model's own over-confidence on data it already knows,
and the correction would be wrong. They are then checked on the test set.

Because both corrections keep the order of the risks (a higher risk stays higher), they mostly change
calibration and have little effect on ROC-AUC. Measuring how much calibration improves, and whether
ROC-AUC changes, is RQ3 of the project.
