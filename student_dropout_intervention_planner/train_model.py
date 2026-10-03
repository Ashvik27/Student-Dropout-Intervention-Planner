"""Train baseline logistic regression and ANN (MLP), compare metrics, audit fairness."""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=UserWarning)
DATA = Path("data/students.csv")
ARTIFACTS = Path("artifacts")
NUMERIC_FEATURES = [
    "attendance_rate", "average_grade", "backlog_count",
    "lms_logins_per_week", "assignment_completion_rate", "fee_delay_days"
]
AUDIT_GROUP = "first_generation_group"
TARGET = "dropout"

def evaluate(model, X, y, threshold=0.5):
    p = model.predict_proba(X)[:, 1]
    pred = (p >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y, pred, labels=[0,1]).ravel().tolist(),
    }

def fairness_audit(y_true, scores, groups, threshold=0.5):
    pred = (scores >= threshold).astype(int)
    df = pd.DataFrame({"y": np.asarray(y_true), "pred": pred, "group": np.asarray(groups)})
    rows = []
    for group, part in df.groupby("group", dropna=False):
        positives = int((part.y == 1).sum())
        rows.append({
            "group": str(group),
            "n": int(len(part)),
            "positive_label_rate": float(part.y.mean()),
            "predicted_high_risk_rate": float(part.pred.mean()),
            "recall": float(recall_score(part.y, part.pred, zero_division=0)) if positives else None,
            "precision": float(precision_score(part.y, part.pred, zero_division=0)),
            "actual_positive_count": positives,
        })
    return rows

def main():
    if not DATA.exists():
        from generate_dataset import generate
        generate(DATA)
    df = pd.read_csv(DATA)
    required = set(NUMERIC_FEATURES + [AUDIT_GROUP, TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    X = df[NUMERIC_FEATURES].copy()  # Audit group is intentionally excluded.
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    models = {
        "Logistic Regression baseline": Pipeline([
            ("prep", preprocessor),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ]),
        "ANN (MLP)": Pipeline([
            ("prep", preprocessor),
            ("model", MLPClassifier(
                hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                alpha=0.001, learning_rate_init=0.001, max_iter=500,
                early_stopping=True, random_state=42
            )),
        ]),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        fitted[name] = model

    # Select by PR-AUC, a useful ranking metric under class imbalance; this is a
    # demonstration rule, not a universal model-selection prescription.
    chosen_name = max(results, key=lambda k: results[k]["pr_auc"])
    chosen = fitted[chosen_name]
    scores = chosen.predict_proba(X_test)[:, 1]
    audit = fairness_audit(y_test, scores, df.loc[idx_test, AUDIT_GROUP].values)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": chosen,
        "model_name": chosen_name,
        "features": NUMERIC_FEATURES,
        "threshold": 0.5,
        "audit_group": AUDIT_GROUP,
        "synthetic_demo": True,
    }, ARTIFACTS / "model_bundle.joblib")
    report = {
        "dataset_rows": int(len(df)),
        "positive_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features_used": NUMERIC_FEATURES,
        "excluded_audit_fields": [AUDIT_GROUP, "program_track", "student_id"],
        "model_metrics": results,
        "selected_model": chosen_name,
        "fairness_audit": audit,
        "notes": [
            "All supplied data are synthetic and for software demonstration only.",
            "Group fields are excluded from model inputs and used only for auditing.",
            "Metrics and subgroup results are illustrative; do not use for real student decisions.",
            "Threshold 0.5 is a demo default and should be chosen with stakeholders.",
        ],
    }
    (ARTIFACTS / "model_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "selected_model": chosen_name,
        "metrics": results,
        "fairness_audit": audit,
        "saved": str(ARTIFACTS / "model_bundle.joblib"),
    }, indent=2))

if __name__ == "__main__":
    main()
