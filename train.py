#!/usr/bin/env python
"""
Train a heart disease prediction model.
Saves the best model + feature names to models/
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Paths
DATA_PATH = Path("data/heart_clean.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def train():
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}\n")

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")

    # -------------------- Logistic Regression --------------------
    print("=== Logistic Regression ===")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    y_proba_lr = lr_pipe.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba_lr):.4f}")
    print(classification_report(y_test, y_pred_lr))

    # -------------------- Random Forest + Tuning --------------------
    print("=== Random Forest (with GridSearchCV) ===")
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, None],
        "min_samples_leaf": [1, 2, 4]
    }
    grid = GridSearchCV(
        rf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

    best_rf = grid.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"Test ROC-AUC : {roc_auc_score(y_test, y_proba_rf):.4f}")
    print(classification_report(y_test, y_pred_rf))

    # -------------------- Select best model --------------------
    # Prefer Random Forest if it is better or equal
    if roc_auc_score(y_test, y_proba_rf) >= roc_auc_score(y_test, y_proba_lr):
        final_model = best_rf
        model_name = "random_forest"
        print("\n>>> Selected model: Random Forest")
    else:
        final_model = lr_pipe
        model_name = "logistic_regression"
        print("\n>>> Selected model: Logistic Regression")

    # Save model + feature names
    output = {
        "model": final_model,
        "features": feature_cols,
        "model_name": model_name
    }
    with open(MODEL_DIR / "model.bin", "wb") as f:
        pickle.dump(output, f)

    print(f"\nModel saved to {MODEL_DIR / 'model.bin'}")
    print("Training complete.")

if __name__ == "__main__":
    train()
