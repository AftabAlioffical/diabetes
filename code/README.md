# Diabetes Risk Prediction: Calibrated and Explainable Machine Learning

This folder has all the code for the COM748 Masters Research Project:
**"Trustworthy Risk Scores, Not Just Accurate Ones: Calibrated and Explainable Machine Learning for Diabetes Risk Prediction from Health Survey Data"**.

The project uses the 2023 BRFSS Diabetes Health Indicators dataset (433,323 survey responses) to build
machine learning models that predict diabetes risk, check whether their predicted probabilities are correct
(calibration), and explain the predictions with SHAP.

For the meaning of every column and code (for example 1 = Excellent ... 5 = Poor for general health,
or 1 = male, 0 = female) see [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Folder structure

```
code/
├── README.md              <- this file, explains how to set up and run the project
├── DATA_DICTIONARY.md     <- what every column and code means, and what each notebook does
├── requirements.txt       <- Python libraries needed for the project
├── data/
│   ├── raw/               <- original dataset downloaded from Kaggle (created by notebook 01)
│   └── processed/         <- clean data (notebook 02) and train / calibration / test sets (notebook 04)
├── models/                <- the 12 trained models as .joblib files (created by notebook 05)
├── results/
│   ├── predictions/       <- predicted probabilities for the calibration and test sets (notebook 05)
│   ├── figures/           <- figures for the report, PNG (600 dpi) and PDF (created by notebook 03)
│   └── tables/            <- summary tables as CSV (created by notebooks 03, 04 and 05)
└── src/
    └── notebooks/
        ├── 01_download_data.ipynb   <- downloads the dataset CSV into data/raw
        ├── 02_data_cleaning.ipynb   <- recodes the raw values and saves the clean CSV
        ├── 03_eda.ipynb             <- exploratory analysis, saves figures and tables
        ├── 04_split_data.ipynb      <- stratified train / calibration / test split
        └── 05_train_models.ipynb    <- trains and tunes the 12 model / imbalance combinations
```

## Setting up the project

### 1. Install Python

You need Python 3.10 or newer. Check your version with:

```bash
python3 --version
```

### 2. Create a virtual environment

Open a terminal inside the `code` folder and run:

```bash
python3 -m venv venv
```

Activate it:

- macOS / Linux:
  ```bash
  source venv/bin/activate
  ```
- Windows:
  ```bash
  venv\Scripts\activate
  ```

### 3. Install the libraries

Make sure the virtual environment is active (you should see `(venv)` in the terminal), then run:

```bash
python -m pip install -r requirements.txt
```

Using `python -m pip` makes sure the libraries go into the virtual environment and not into
another Python on the computer (for example Anaconda).

### 4. Run the notebooks

**In VS Code:** open a notebook, click **Select Kernel** at the top right, choose
**Python Environments** and pick `code/venv/bin/python`. The `.vscode/settings.json` file in the
main project folder already sets this as the default.

**In the browser:** run

```bash
jupyter notebook
```

Then open the notebooks in `src/notebooks` and run them in order.

### Common problem

`ModuleNotFoundError: No module named 'kagglehub'` means the notebook is using a different Python
from the one the libraries were installed into. Select the `code/venv` kernel as described above
and restart the kernel.

## Notebooks

Run the notebooks in this order:

| No. | Notebook | What it does |
|-----|----------|--------------|
| 01 | `01_download_data.ipynb` | Downloads the dataset from Kaggle and saves it as `data/raw/Diabetes Health Indicators.csv` |
| 02 | `02_data_cleaning.ipynb` | Creates the diabetes target, recodes the BRFSS codes into readable columns and saves `data/processed/diabetes_clean.csv` |
| 03 | `03_eda.ipynb` | Class balance, missing values, diabetes rate by feature, age and sex, correlation. Saves figures to `results/figures` and tables to `results/tables` |
| 04 | `04_split_data.ipynb` | Splits the clean data 60 / 20 / 20 into `train.csv`, `calibration.csv` and `test.csv` (stratified on diabetes, `random_state=42`) |
| 05 | `05_train_models.ipynb` | Trains 4 models with 3 imbalance settings, tunes them with 5-fold cross-validation and saves the models and predicted probabilities. Takes about 45 minutes (Random Forest is the slowest) |

### Cleaning decisions (notebook 02)

- **Target:** `DIABETE4` 1 (yes) becomes 1 and 3 (no) becomes 0. Diabetes only during pregnancy (2),
  prediabetes (4), don't know (7), refused (9) and missing answers are removed (14,831 rows).
- **`DIABTYPE` is dropped** because it is only asked to people who already have diabetes.
- Don't know / refused codes (7, 9, 77, 99, age group 14, education and income 9) are set to missing.
- `MENTHLTH` and `PHYSHLTH`: 88 means zero days. `_BMI5` is divided by 100 to give the real BMI.
- Missing values are kept and will be filled in later inside the model pipeline, using the training data only.
- Duplicate rows are kept, because different people can give the same answers to 20 coded questions.

The clean file has 418,492 rows and 19 columns, and 14.3% of people have diabetes.

### Figures (notebook 03)

All figures are sized for an IEEE two column paper: single column figures are 3.5 in wide and
double column figures are 7.16 in wide. Text is Times New Roman at 7-8 pt. Each figure is saved
as a 600 dpi PNG (for Word) and as a PDF (for LaTeX).

| File | Width | Shows |
|------|-------|-------|
| `fig_class_balance` | single | Number of people with and without diabetes |
| `fig_missing_values` | single | Percentage of missing values for each feature |
| `fig_bmi_distribution` | single | BMI distribution for both groups, dashed lines are the medians |
| `fig_binary_feature_rates` | single | Diabetes rate for yes/no answers of each binary feature |
| `fig_rate_age_sex` | single | Diabetes rate by age group for men and women |
| `fig_rate_health_bmi` | double | Diabetes rate by general health and by BMI category |
| `fig_rate_income_education` | double | Diabetes rate by income and by education |
| `fig_correlation` | double | Spearman correlation between all features and the target |

### Data split (notebook 04)

| Set | Records | Diabetes cases | Diabetes rate |
|-----|---------|----------------|---------------|
| Training (60%) | 251,095 | 35,872 | 14.29% |
| Calibration (20%) | 83,698 | 11,957 | 14.29% |
| Test (20%) | 83,699 | 11,957 | 14.29% |

The training set is used to train and tune the models, the calibration set to fit Platt scaling
and isotonic regression, and the test set only once for the final results. Missing values are
not filled in here, this is done inside the model pipeline using the training set only.
In the test set the smallest age band and sex group (men aged 18-34) has 137 diabetes cases.

### Model training (notebook 05)

- **Models:** Logistic Regression (`lr`), Decision Tree (`dt`), Random Forest (`rf`) and XGBoost (`xgb`).
- **Imbalance settings:** `none` (no correction), `weighted` (class weighting, `scale_pos_weight` for XGBoost)
  and `oversampled` (random oversampling of the diabetes class).
- **Pipeline:** median imputation with missing-value indicator columns, scaling for Logistic Regression only,
  then oversampling (if used) and the model. Everything is fitted inside the cross-validation folds,
  so nothing from the validation folds leaks into training.
- **Tuning:** randomized search with stratified 5-fold cross-validation on the training set, ROC-AUC
  is used to pick the best settings and log loss is also recorded.
- **Outputs:** `models/<model>_<setting>.joblib`, `results/predictions/calibration_predictions.csv`,
  `results/predictions/test_predictions.csv` and `results/tables/table_cv_results.csv`.
- Models are saved with `joblib` compression, so `models/` is about 130 MB and the largest file
  (Random Forest with oversampling) is 58 MB, under the GitHub 100 MB file limit.

## Dataset

- **Name:** Diabetes Health Indicators (2023 BRFSS)
- **Source:** https://www.kaggle.com/datasets/siamaktahmasbi/diabetes-health-indicators
- **Size:** 433,323 rows and 20 columns
- **Official documentation:** https://www.cdc.gov/brfss/annual_data/annual_2023.html

The dataset is public, so it can normally be downloaded without a Kaggle account.
If the download asks for login details, create an API token on your Kaggle account page
(Settings → API → Create New Token) and place the `kaggle.json` file in `~/.kaggle/`.
