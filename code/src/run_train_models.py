import os
import time
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

base_dir = Path(__file__).resolve().parent.parent
os.chdir(base_dir)

processed_data_folder = base_dir / "data" / "processed"
models_folder = base_dir / "models"
predictions_folder = base_dir / "results" / "predictions"
tables_folder = base_dir / "results" / "tables"

models_folder.mkdir(exist_ok=True, parents=True)
predictions_folder.mkdir(exist_ok=True, parents=True)
tables_folder.mkdir(exist_ok=True, parents=True)

train_data = pd.read_csv(processed_data_folder / "train.csv")
calibration_data = pd.read_csv(processed_data_folder / "calibration.csv")
test_data = pd.read_csv(processed_data_folder / "test.csv")

X_train = train_data.drop(columns=["diabetes"])
y_train = train_data["diabetes"]
X_calibration = calibration_data.drop(columns=["diabetes"])
X_test = test_data.drop(columns=["diabetes"])

print("Training:", X_train.shape)
print("Features:", list(X_train.columns))

random_seed = 42
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
xgboost_weight = negative_count / positive_count
print("Negative / positive ratio:", round(xgboost_weight, 2))

base_models = {
    "lr": LogisticRegression(max_iter=1000),
    "dt": DecisionTreeClassifier(random_state=random_seed),
    "rf": RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=random_seed),
    "xgb": XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=1, random_state=random_seed),
}

model_names = {
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}

settings = ["none", "weighted", "oversampled"]

param_grids = {
    "lr": {
        "model__C": [0.001, 0.01, 0.1, 1, 10],
    },
    "dt": {
        "model__max_depth": [4, 6, 8, 10, 12, 15],
        "model__min_samples_leaf": [50, 100, 200, 500, 1000],
    },
    "rf": {
        "model__max_depth": [8, 12, 16, 20],
        "model__min_samples_leaf": [20, 50, 100],
        "model__max_features": ["sqrt", 0.5],
    },
    "xgb": {
        "model__n_estimators": [200, 400, 600],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [3, 4, 5, 6],
        "model__min_child_weight": [1, 5, 10],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },
}

search_iterations = {"lr": 5, "dt": 10, "rf": 10, "xgb": 12}


def make_pipeline(model_key, setting):
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]

    if model_key == "lr":
        steps.append(("scaler", StandardScaler()))

    if setting == "oversampled":
        steps.append(("oversampler", RandomOverSampler(random_state=random_seed)))

    model = clone(base_models[model_key])
    if setting == "weighted":
        if model_key == "xgb":
            model.set_params(scale_pos_weight=xgboost_weight)
        else:
            model.set_params(class_weight="balanced")

    steps.append(("model", model))
    return Pipeline(steps)


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
calibration_predictions = calibration_data[["diabetes", "age_group", "sex"]].copy()
test_predictions = test_data[["diabetes", "age_group", "sex"]].copy()
results = []

for model_key in base_models:
    for setting in settings:
        run_name = model_key + "_" + setting
        print("Training", run_name, "...")
        start_time = time.time()

        search = RandomizedSearchCV(
            make_pipeline(model_key, setting),
            param_grids[model_key],
            n_iter=search_iterations[model_key],
            scoring={"roc_auc": "roc_auc", "log_loss": "neg_log_loss"},
            refit="roc_auc",
            cv=folds,
            n_jobs=-1,
            random_state=random_seed,
        )
        search.fit(X_train, y_train)
        minutes = (time.time() - start_time) / 60

        best = search.best_index_
        best_params = {}
        for name, value in search.best_params_.items():
            best_params[name.replace("model__", "")] = value

        results.append({
            "run": run_name,
            "model": model_names[model_key],
            "setting": setting,
            "cv_roc_auc": search.cv_results_["mean_test_roc_auc"][best],
            "cv_roc_auc_std": search.cv_results_["std_test_roc_auc"][best],
            "cv_log_loss": -search.cv_results_["mean_test_log_loss"][best],
            "best_params": best_params,
            "minutes": round(minutes, 1),
        })

        best_model = search.best_estimator_
        joblib.dump(best_model, models_folder / f"{run_name}.joblib", compress=3)

        calibration_predictions[run_name] = best_model.predict_proba(X_calibration)[:, 1]
        test_predictions[run_name] = best_model.predict_proba(X_test)[:, 1]

        print("   ROC-AUC:", round(results[-1]["cv_roc_auc"], 4), " time:", round(minutes, 1), "min")

cv_results = pd.DataFrame(results)
cv_results["cv_roc_auc"] = cv_results["cv_roc_auc"].round(4)
cv_results["cv_roc_auc_std"] = cv_results["cv_roc_auc_std"].round(4)
cv_results["cv_log_loss"] = cv_results["cv_log_loss"].round(4)
cv_results.to_csv(tables_folder / "table_cv_results.csv", index=False)
print(cv_results[["model", "setting", "cv_roc_auc", "cv_roc_auc_std", "cv_log_loss", "minutes"]])

mean_predicted = test_predictions.drop(columns=["diabetes", "age_group", "sex"]).mean() * 100
print("Real diabetes rate in the test set:", round(test_predictions["diabetes"].mean() * 100, 2), "%")
print(mean_predicted.round(2))

calibration_predictions.to_csv(predictions_folder / "calibration_predictions.csv", index=False)
test_predictions.to_csv(predictions_folder / "test_predictions.csv", index=False)
print("Saved", calibration_predictions.shape, test_predictions.shape)
