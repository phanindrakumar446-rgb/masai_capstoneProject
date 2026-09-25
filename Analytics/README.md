# Module 2: Analytics & Machine Learning (Titanic)

## EDA observations

Dataset has 891 rows and 15 columns. Loaded with seaborn and saved to `titanic.csv` immediately.

**Missing values:**
- `age` — about 20% missing. Imputed using median grouped by sex + pclass (felt more accurate than a single global median)
- `deck` — 77% missing, way too much to impute. Kept it as a "Missing" category since whether someone had a known deck actually correlated with survival
- `embarked` — only 2 rows missing, just dropped them

**A few things I noticed in EDA:**
- Fare is very right-skewed — a few first class passengers paid way more than everyone else. Mean > median because of these outliers
- Women had ~74% survival vs men at ~19%. Biggest predictor visually
- Class 1 passengers survived at 63%, class 3 at only 24%
- Age and fare have some correlation with pclass, nothing too strong otherwise
- IQR check found many outliers in fare (expected given the skew), and some in age

## Modeling

Split: 80/20 stratified on survived so class distribution is preserved in both sets.

**Three models trained:**

| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression | ~80% | 0.861 |
| Decision Tree | ~78% | 0.76 |
| Random Forest | ~82% | 0.85 |

Decision tree overfit a bit without depth limit. Random Forest did well but Logistic Regression had the best AUC so I went with that as the final model.

**Class imbalance:**
Tried three approaches — baseline, `class_weight='balanced'`, and SMOTE. The balanced weight helped recall for class 0 (not survived) without hurting overall accuracy much. SMOTE was similar. For this dataset class imbalance wasn't extreme so the difference wasn't huge.

**Hyperparameter tuning (Random Forest):**
Used GridSearchCV with StratifiedKFold. Best params: `n_estimators=400, max_depth=6, max_features=None`. OOB score came out around 0.81.

**Regression on Fare:**
Linear regression with age, pclass, sibsp, parch as features. R² was modest (~0.27). Residual plot showed clear heteroscedasticity — variance increases with predicted fare, which makes sense given the skewed distribution.

**No data leakage:**
All preprocessing (imputation, scaling, encoding) is inside a sklearn Pipeline, so `fit()` only sees training data. SMOTE also runs only inside the pipeline fit step.

## Final pipeline

Saved as `best_model_pipeline.joblib` — includes the full preprocessor + Logistic Regression. Reloaded and tested on a raw input row (with missing age) and it predicted correctly.
