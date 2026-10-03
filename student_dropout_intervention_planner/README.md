# Student Dropout Intervention Planner
**Track:** ANN & Predictive Analytics  
**Domain:** Academic Analytics & Student Retention

> Safety note: this repository ships with a fully synthetic dataset. It is an educational prototype, not a validated student-risk system. Never use its scores to make consequential decisions about real students.

## 1. Project objective
Build a prototype that helps student-support teams prioritize voluntary, supportive outreach using academic and learning-engagement indicators. The app provides a single-record score, a capacity-limited ranked queue, model comparison, and a subgroup fairness audit.

## 2. Included deliverables
- `data/students.csv` — generated synthetic tabular dataset with missing values.
- `generate_dataset.py` — reproducible data generator.
- `train_model.py` — EDA-ready preprocessing pipeline, baseline and ANN training, holdout metrics, fairness audit, and saved model.
- `app.py` — Streamlit user interface with live predictions, contributing indicator descriptions, queue capacity control, CSV export, model metrics, and fairness audit.
- `tests/test_project.py` — dataset checks and five live-case prediction test.
- `artifacts/model_report.json` and `artifacts/model_bundle.joblib` — generated after training.

## 3. Target definition and prediction design
- **Prediction point:** end of a teaching week, using only information available by that point.
- **Observation window:** the most recent 4–8 teaching weeks, depending on institutional data availability. The synthetic demo stores aggregate values and does not encode a true temporal window.
- **Target label:** `dropout = 1` means a synthetic record is labelled as dropout; `0` means not labelled as dropout.
- **Features:** attendance rate, average grade, backlog count, LMS logins/week, assignment completion rate, fee delay days.
- **Excluded fields:** `student_id`, `first_generation_group`, and `program_track`. Group fields are used only for audit/display and never passed to the model.
- **Leakage prevention:** in a real deployment, define a strict cutoff date and exclude withdrawal status, future attendance/grades, post-intervention activity, final outcomes, and any field created after the prediction point. Split by time or student when appropriate. The synthetic dataset is not a substitute for a time-aware validation design.

## 4. Exploratory analysis and missing values
The generator creates 1,800 synthetic records, with some missing numeric predictors. Run:
```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/students.csv")
print(df.head())
print(df.shape)
print(df.isna().mean().sort_values(ascending=False))
print(df.describe(include="all").T)
print(df["dropout"].value_counts(normalize=True))
PY
```
Numeric missing values are imputed using the median inside a scikit-learn pipeline. The imputer is fitted on training data only, avoiding leakage from the test set.

## 5. Models and evaluation
- **Baseline:** class-weighted logistic regression.
- **ANN:** feed-forward multilayer perceptron (32 and 16 hidden units).
- **Imbalance:** class weighting for the logistic-regression baseline; report precision, recall, F1, PR-AUC, ROC-AUC, accuracy, and confusion matrix. The ANN may need resampling/class weighting in a future extension if imbalance is severe.
- **Model selection:** demo code selects by holdout PR-AUC. In a real study, tune hyperparameters and thresholds on validation data, then report a separate untouched test set. Do not repeatedly select models against the final test set.
- **Threshold:** 0.5 is a demonstration default, not a universal operating point. Select a threshold with counsellor capacity, missed-need costs, and false-positive burden in mind.

## 6. Fairness audit
The holdout test results are grouped by `first_generation_group`, which is not an input. The report compares group size, observed label prevalence, predicted high-risk rate, recall, and precision. This is a basic audit, not proof of fairness. Review uncertainty, subgroup sizes, label quality, access to support, and institutional context. Document the concern, evidence, proposed mitigation, and owner before any real-world pilot. Do not automatically alter service access by demographic group.

## 7. Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```
Open the local URL printed by Streamlit.

## 8. Run tests
```bash
pytest -q
```
The test suite checks the dataset schema and runs five sample records through the trained model. It is a smoke test, not evidence of real-world predictive validity.

## 9. Five test cases
| Case | Attendance | Grade | Backlogs | LMS/week | Assignments | Fee delay | Test intent |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 95 | 90 | 0 | 12 | 95 | 0 | Strong engagement |
| B | 80 | 70 | 1 | 6 | 80 | 5 | Moderate profile |
| C | 60 | 50 | 3 | 2 | 45 | 45 | Multiple support indicators |
| D | 72 | 58 | 2 | 3 | 55 | 15 | Mixed indicators |
| E | 45 | 35 | 5 | 1 | 25 | 70 | Several support indicators |

Expected: the app returns five valid probabilities between 0 and 1. Do not hard-code expected rankings; fitted models can behave differently.

## 10. Responsible-use checklist
- Verify data definitions, freshness, consent/authority, and missingness before use.
- Validate prospectively and across terms/programs before considering a pilot.
- Show reasons as review prompts, not causal explanations.
- Give students a chance to correct data and describe circumstances.
- Use the ranked queue only to organize supportive human review, not to deny admission, scholarships, progression, or services.
- Set a weekly capacity with counsellors; document why that capacity is appropriate.
- Monitor precision/recall, subgroup errors, drift, and downstream outcomes.
- Provide a non-model route to support and an appeal/correction process.
