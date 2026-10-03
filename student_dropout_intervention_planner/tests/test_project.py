from pathlib import Path
import subprocess
import sys
import pandas as pd

def test_dataset_has_required_columns():
    if not Path("data/students.csv").exists():
        subprocess.run([sys.executable, "generate_dataset.py"], check=True)
    df = pd.read_csv("data/students.csv")
    expected = {
        "student_id", "attendance_rate", "average_grade", "backlog_count",
        "lms_logins_per_week", "assignment_completion_rate", "fee_delay_days",
        "first_generation_group", "program_track", "dropout"
    }
    assert expected.issubset(df.columns)
    assert len(df) >= 100
    assert set(df["dropout"].dropna().unique()).issubset({0, 1})

def test_five_live_cases_after_training():
    if not Path("artifacts/model_bundle.joblib").exists():
        subprocess.run([sys.executable, "train_model.py"], check=True)
    import joblib
    bundle = joblib.load("artifacts/model_bundle.joblib")
    features = bundle["features"]
    cases = pd.DataFrame([
        [95, 90, 0, 12, 95, 0],
        [80, 70, 1, 6, 80, 5],
        [60, 50, 3, 2, 45, 45],
        [72, 58, 2, 3, 55, 15],
        [45, 35, 5, 1, 25, 70],
    ], columns=features)
    probabilities = bundle["model"].predict_proba(cases)[:, 1]
    assert len(probabilities) == 5
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
