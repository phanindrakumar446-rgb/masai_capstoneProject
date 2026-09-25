"""
Module 2 / Part B - Predictive modeling, continuing from the same data 01_eda.py saved.

    python analytics/02_modeling.py      (run 01_eda.py first - it produces titanic.csv)

No sns.load_dataset call here: the data comes only from the committed titanic.csv.
All preprocessing lives inside sklearn/imblearn Pipelines, so it is fit on the training split only.
"""

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from cleaning import (
    CHART_DIR, CSV_PATH, DIED_COLOR, HERE, INK_2, MUTED, SURVIVED_COLOR, apply_style, row_level_clean,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
CHART_DIR.mkdir(exist_ok=True)
apply_style()
SEED = 42
MODEL_PATH = HERE / "best_model_pipeline.joblib"
MODEL_COLORS = {"Logistic Regression": SURVIVED_COLOR, "Decision Tree": DIED_COLOR,
                "Random Forest": "#1baf7a", "Random Forest (tuned)": "#4a3aa7"}


def section(title):
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


def save(fig, name):
    fig.savefig(CHART_DIR / name)
    plt.close(fig)
    print(f"[chart saved] charts/{name}")


# ======================================================================================
# Load the SAME data 01_eda.py saved, apply the same row-level cleaning decisions
# ======================================================================================
section("Load titanic.csv (produced by 01_eda.py)")
raw = pd.read_csv(CSV_PATH)
df = row_level_clean(raw)   # drops the 2 rows missing embarked; age stays NaN -> imputed inside the pipeline
print(f"raw {raw.shape} -> after row-level cleaning {df.shape}; age still missing in {df['age'].isna().sum()} rows "
      "(imputed later, train-only)")

NUMERIC = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL = ["sex", "embarked"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "survived"
X, y = df[FEATURES], df[TARGET]

# ======================================================================================
# Task 7 - Stratified split BEFORE any preprocessing
# ======================================================================================
section("TASK 7 - Stratified train/test split")
print("Class balance (full):", y.value_counts(normalize=True).round(3).to_dict())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"train {X_train.shape}, test {X_test.shape}")
print("Class balance (train):", y_train.value_counts(normalize=True).round(3).to_dict())
print("Class balance (test): ", y_test.value_counts(normalize=True).round(3).to_dict())


# ======================================================================================
# Task 8 - Preprocessing (ColumnTransformer inside a Pipeline -> fit on train only)
# ======================================================================================
def make_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def make_pipeline(estimator):
    return Pipeline([("prep", make_preprocessor()), ("clf", estimator)])


# ======================================================================================
# Task 9 & 10 - Train three classifiers on the identical split, evaluate
# ======================================================================================
section("TASK 9/10 - Train & evaluate LR, DT, RF")


def evaluate(pipe, name):
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred)
    row = {
        "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred), "F1": f1_score(y_test, pred), "AUC": roc_auc_score(y_test, proba),
    }
    print(f"\n{name}\n  confusion matrix [[TN FP] [FN TP]] = {cm.tolist()}\n  "
          + "  ".join(f"{k}={v:.3f}" for k, v in row.items()))
    return row, cm, proba


models = {
    "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, random_state=SEED)),
    "Decision Tree": make_pipeline(DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=SEED)),
    "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=200, random_state=SEED)),
}
results, cms, probas = {}, {}, {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)                   # preprocessing fit on X_train only
    results[name], cms[name], probas[name] = evaluate(pipe, name)

# Decision tree visualisation
dt = models["Decision Tree"]
feature_names = dt.named_steps["prep"].get_feature_names_out()
print("\nEncoded feature names:", list(feature_names))
fig, ax = plt.subplots(figsize=(24, 11))
plot_tree(dt.named_steps["clf"], feature_names=feature_names, class_names=["Died", "Survived"],
          filled=True, rounded=True, fontsize=8, impurity=True, ax=ax)
ax.set_title("Decision Tree (max_depth=4) - features are post-preprocessing (scaled numeric, one-hot categorical)")
save(fig, "09_decision_tree.png")


# ======================================================================================
# Task 11 - Imbalance handling comparison (Logistic Regression)
# ======================================================================================
section("TASK 11 - Imbalance handling (Logistic Regression)")
counts = y_train.value_counts()
print(f"Training class counts: not survived={counts[0]}, survived={counts[1]} "
      f"(ratio {counts[0] / counts[1]:.2f}:1)")

variants = {
    "(a) baseline": make_pipeline(LogisticRegression(max_iter=1000, random_state=SEED)),
    "(b) class_weight='balanced'": make_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    # imblearn Pipeline: SMOTE runs only inside .fit() -> only on the (preprocessed) training fold;
    # .predict() on the test split skips the sampler entirely.
    "(c) SMOTE (train fold only)": ImbPipeline([("prep", make_preprocessor()), ("smote", SMOTE(random_state=SEED)),
                                                ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
}
imb_rows = {}
for name, pipe in variants.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    imb_rows[name] = {"Precision": precision_score(y_test, pred), "Recall": recall_score(y_test, pred),
                      "F1": f1_score(y_test, pred), "Accuracy": accuracy_score(y_test, pred)}
smote_pipe = variants["(c) SMOTE (train fold only)"]
Xt = smote_pipe.named_steps["prep"].transform(X_train)
_, y_res = smote_pipe.named_steps["smote"].fit_resample(Xt, y_train)
print(f"After SMOTE, training fold class counts: {pd.Series(y_res).value_counts().to_dict()} "
      f"(test set untouched: {y_test.value_counts().to_dict()})")
imb_table = pd.DataFrame(imb_rows).T.round(3)
print(imb_table.to_string())

# ======================================================================================
# Task 12 - GridSearchCV on Random Forest (oob_score=True)
# ======================================================================================
section("TASK 12 - GridSearchCV (Random Forest, oob_score=True)")
rf_search = GridSearchCV(
    make_pipeline(RandomForestClassifier(oob_score=True, bootstrap=True, random_state=SEED, n_jobs=-1)),
    param_grid={
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [4, 6, 8, None],
        "clf__max_features": ["sqrt", "log2", None],
    },
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring="roc_auc", n_jobs=-1,
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
print("Best params:", rf_search.best_params_)
print(f"Best mean CV ROC-AUC: {rf_search.best_score_:.4f}")
print(f"OOB score of best RandomForestClassifier (refit on full training split): "
      f"{best_rf.named_steps['clf'].oob_score_:.4f}")
results["Random Forest (tuned)"], cms["Random Forest (tuned)"], probas["Random Forest (tuned)"] = evaluate(
    best_rf, "Random Forest (tuned)")

# Confusion matrices + ROC curves
fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for ax, (name, cm) in zip(axes, cms.items()):
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec", SURVIVED_COLOR]))
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=13,
                color="white" if v > cm.max() * 0.6 else "#0b0b0b")
    ax.set(title=name, xticks=[0, 1], yticks=[0, 1], xticklabels=["Died", "Survived"],
           yticklabels=["Died", "Survived"], xlabel="predicted", ylabel="actual")
    ax.grid(False)
fig.suptitle("Confusion matrices (test split)", fontweight="bold")
fig.tight_layout()
save(fig, "10_confusion_matrices.png")

fig, ax = plt.subplots(figsize=(6.5, 6))
for name, proba in probas.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, linewidth=2, color=MODEL_COLORS[name], label=f"{name} (AUC {results[name]['AUC']:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="chance")
ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curves (test split)")
ax.legend(loc="lower right")
save(fig, "11_roc_curves.png")

clf_table = pd.DataFrame(results).T.round(3)
print("\nClassifier comparison (test split):\n", clf_table.to_string())

# ======================================================================================
# Task 13 - Regression side-task: predict fare
# ======================================================================================
section("TASK 13 - Linear regression: predict fare")
REG_NUMERIC = ["pclass", "age", "sibsp", "parch", "survived"]
REG_CATEGORICAL = ["sex", "embarked"]
Xr, yr = df[REG_NUMERIC + REG_CATEGORICAL], df["fare"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=SEED)
reg = Pipeline([
    ("prep", ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), REG_NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))]), REG_CATEGORICAL),
    ])),
    ("reg", LinearRegression()),
])
reg.fit(Xr_train, yr_train)
yr_pred = reg.predict(Xr_test)
n, p = len(yr_test), len(reg.named_steps["prep"].get_feature_names_out())
r2 = r2_score(yr_test, yr_pred)
reg_metrics = {
    "MAE": mean_absolute_error(yr_test, yr_pred),
    "RMSE": np.sqrt(mean_squared_error(yr_test, yr_pred)),
    "R2": r2,
    "Adj R2": 1 - (1 - r2) * (n - 1) / (n - p - 1),
}
print(f"n_test={n}, p (encoded predictors)={p}")
print("  ".join(f"{k}={v:.3f}" for k, v in reg_metrics.items()))
print("Coefficients:", {f: round(float(c), 2) for f, c in zip(reg.named_steps["prep"].get_feature_names_out(),
                                                                reg.named_steps["reg"].coef_)})

residuals = yr_test - yr_pred
# Simple heteroscedasticity diagnostics: does the residual spread grow with the fitted value?
spread_corr = np.corrcoef(yr_pred, np.abs(residuals))[0, 1]
terciles = pd.qcut(yr_pred, 3, labels=["low fitted", "mid fitted", "high fitted"])
spread_by_tercile = pd.Series(residuals.values).groupby(terciles, observed=True).std()
print(f"corr(fitted, |residual|) = {spread_corr:.3f}")
print("Residual std by fitted-value tercile:\n", spread_by_tercile.round(2).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(yr_pred, residuals, s=22, color=SURVIVED_COLOR, alpha=0.7, edgecolor="#fcfcfb", linewidth=0.6)
ax.axhline(0, color=INK_2, linewidth=1)
ax.set(xlabel="fitted fare", ylabel="residual (actual - fitted)",
       title="Residuals vs fitted - linear regression on fare (test split)")
save(fig, "12_regression_residuals.png")

# ======================================================================================
# Task 14 - Combined comparison table (two separate metric groups)
# ======================================================================================
section("TASK 14 - Model comparison table")
rows = list(results) + ["Linear Regression (fare)"]
columns = pd.MultiIndex.from_tuples(
    [("Classification (target: survived)", m) for m in ["Accuracy", "Precision", "Recall", "F1", "AUC"]]
    + [("Regression (target: fare)", m) for m in ["MAE", "RMSE", "R2", "Adj R2"]])
comparison = pd.DataFrame(index=rows, columns=columns, dtype=object)
for name, r in results.items():
    for m, v in r.items():
        comparison.loc[name, ("Classification (target: survived)", m)] = f"{v:.3f}"
for m, v in reg_metrics.items():
    comparison.loc["Linear Regression (fare)", ("Regression (target: fare)", m)] = f"{v:.3f}"
comparison = comparison.fillna("-")
print(comparison.to_string())

md = ["| Model | Accuracy | Precision | Recall | F1 | AUC | MAE | RMSE | R² | Adj R² |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for name in rows:
    md.append("| " + name + " | " + " | ".join(comparison.loc[name].tolist()) + " |")
(HERE / "outputs").mkdir(exist_ok=True)
(HERE / "outputs" / "model_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

# ======================================================================================
# Task 15 - Save the best complete pipeline, reload, predict on raw data
# ======================================================================================
section("TASK 15 - Save / reload best pipeline")
fitted = {**models, "Random Forest (tuned)": best_rf}
best_name = max(results, key=lambda k: (results[k]["AUC"], results[k]["F1"]))
best_pipe = fitted[best_name]
print(f"Best classifier by test AUC (tie-break F1): {best_name}")
joblib.dump(best_pipe, MODEL_PATH)
print(f"Saved full pipeline (ColumnTransformer + estimator) -> {MODEL_PATH.name}")

reloaded = joblib.load(MODEL_PATH)
same = np.array_equal(reloaded.predict(X_test), best_pipe.predict(X_test))
print(f"Reloaded pipeline reproduces in-memory test predictions exactly: {same}")

new_passengers = pd.DataFrame([
    {"pclass": 1, "age": 29, "sibsp": 0, "parch": 0, "fare": 100.0, "sex": "female", "embarked": "C"},
    {"pclass": 3, "age": 25, "sibsp": 0, "parch": 0, "fare": 7.25, "sex": "male", "embarked": "S"},
    {"pclass": 2, "age": np.nan, "sibsp": 1, "parch": 2, "fare": 30.0, "sex": "female", "embarked": "S"},
])  # raw, unscaled, un-encoded - and one missing age
new_passengers["pred_survived"] = reloaded.predict(new_passengers[FEATURES])
new_passengers["p_survived"] = reloaded.predict_proba(new_passengers[FEATURES])[:, 1].round(3)
print("Predictions on raw new passengers:\n", new_passengers.to_string(index=False))

print("\nModeling complete.")
