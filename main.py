import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.linear_model import LogisticRegression

import joblib #save the fitted pipeline

# 1. Load data
df = pd.read_csv("dataset/Churn_Modelling.csv")

# Quick checks
print(df.head())       # show first 5 rows
print(df.dtypes)       # show column types

# 2. Drop ID-like columns
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# 3. Separate target and features
y = df["Exited"]
X = df.drop(columns=["Exited"])

# 4. Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# 5. Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Base pipeline (preprocessing + logistic regression)
base_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # handle class imbalance
        n_jobs=-1                  # use all CPU cores
    )),
])

# 8. Hyperparameter grid for GridSearchCV
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__penalty": ["l2"],
    "classifier__solver": ["lbfgs", "liblinear"],
}

grid = GridSearchCV(
    estimator=base_pipe,
    param_grid=param_grid,
    scoring="f1",    # focus on F1 for churn class
    cv=5,
    n_jobs=-1,
    verbose=1
)

# 9. Fit grid search on training data
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
best_model = grid.best_estimator_

# Save the trained model to a file
joblib.dump(best_model, "churn_model.joblib")
print("Model saved to churn_model.joblib")

# 10. Evaluate on test set
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

threshold = 0.6  # for example
y_pred_custom = (y_proba >= threshold).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


acc2 = accuracy_score(y_test, y_pred_custom)
prec2 = precision_score(y_test, y_pred_custom)
rec2 = recall_score(y_test, y_pred_custom)
f12 = f1_score(y_test, y_pred_custom)


print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)


print("Custom threshold metrics:")
print("Accuracy:", acc2)
print("Precision:", prec2)
print("Recall:", rec2)
print("F1 Score:", f12)


# 11. Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 12. ROC and Precision-Recall curves (optional but useful)
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title("ROC Curve")
plt.tight_layout()
plt.show()

PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test)
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.show()
