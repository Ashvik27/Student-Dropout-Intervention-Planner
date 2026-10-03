"""Generate a fully synthetic student-retention dataset for demonstration only."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N = 1800

def generate(path="data/students.csv", n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    attendance = np.clip(rng.normal(78, 15, n), 20, 100)
    avg_grade = np.clip(rng.normal(66, 14, n), 10, 100)
    backlogs = np.clip(rng.poisson(1.0, n), 0, 8)
    lms = np.clip(rng.gamma(2.3, 2.0, n), 0, 20)
    assignment = np.clip(rng.normal(76, 18, n), 0, 100)
    fee_delay = np.clip(rng.gamma(1.4, 12, n), 0, 120)
    first_gen = rng.choice(["First-generation", "Continuing-generation"], n, p=[0.38, 0.62])
    program = rng.choice(["STEM", "Business", "Arts"], n, p=[0.42, 0.33, 0.25])

    # Synthetic association for an educational demo; not a causal or real-world model.
    logit = (
        2.4
        - 0.045 * (attendance - 70)
        - 0.055 * (avg_grade - 55)
        + 0.48 * backlogs
        - 0.13 * (lms - 5)
        - 0.025 * (assignment - 65)
        + 0.012 * (fee_delay - 15)
        + (first_gen == "First-generation") * 0.28
        + (program == "STEM") * 0.08
    )
    prob = 1 / (1 + np.exp(-logit))
    dropout = rng.binomial(1, prob)

    df = pd.DataFrame({
        "student_id": [f"S{i:05d}" for i in range(1, n + 1)],
        "attendance_rate": attendance.round(1),
        "average_grade": avg_grade.round(1),
        "backlog_count": backlogs.astype(int),
        "lms_logins_per_week": lms.round(1),
        "assignment_completion_rate": assignment.round(1),
        "fee_delay_days": fee_delay.round(1),
        "first_generation_group": first_gen,
        "program_track": program,
        "dropout": dropout.astype(int),
    })

    # Add missingness to numeric predictors to demonstrate robust preprocessing.
    for col, rate in {
        "attendance_rate": 0.035,
        "average_grade": 0.025,
        "lms_logins_per_week": 0.05,
        "assignment_completion_rate": 0.04,
        "fee_delay_days": 0.03,
    }.items():
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    df = generate()
    print(f"Created data/students.csv with {len(df)} synthetic rows.")
    print(df["dropout"].value_counts(normalize=True).rename("share").round(3))
