"""Streamlit UI for the Student Dropout Intervention Planner demo."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Dropout Intervention Planner", page_icon="🎓", layout="wide")
st.title("🎓 Student Dropout Intervention Planner")
st.caption("Academic analytics demo • Human-reviewed support prioritization • Synthetic data only")

BUNDLE_PATH = Path("artifacts/model_bundle.joblib")
REPORT_PATH = Path("artifacts/model_report.json")
DATA_PATH = Path("data/students.csv")
FEATURES = [
    "attendance_rate", "average_grade", "backlog_count",
    "lms_logins_per_week", "assignment_completion_rate", "fee_delay_days"
]
LABELS = {
    "attendance_rate": "Attendance rate (%)",
    "average_grade": "Average grade (%)",
    "backlog_count": "Backlog count",
    "lms_logins_per_week": "LMS logins per week",
    "assignment_completion_rate": "Assignment completion (%)",
    "fee_delay_days": "Fee delay (days)",
}
HELP = {
    "attendance_rate": "Recent attendance percentage.",
    "average_grade": "Average academic grade percentage.",
    "backlog_count": "Number of unresolved backlogs.",
    "lms_logins_per_week": "Learning-platform logins per week.",
    "assignment_completion_rate": "Percentage of assignments completed.",
    "fee_delay_days": "Days of fee delay; may be missing or not applicable.",
}

@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)

@st.cache_data
def load_report():
    if REPORT_PATH.exists():
        return json.loads(REPORT_PATH.read_text())
    return None

bundle = load_bundle()
report = load_report()
if bundle is None:
    st.warning("Model artifacts are missing. In the project folder, run `python generate_dataset.py` and `python train_model.py`, then restart the app.")
    st.stop()

model = bundle["model"]
st.info("Important: this is a synthetic demonstration. Predictions must not be used to make real student decisions. Use outputs only to prompt supportive human review.")
tab1, tab2, tab3, tab4 = st.tabs(["Single student", "Ranked support queue", "Model evaluation", "Fairness audit"])

def risk_band(score):
    if score >= 0.70:
        return "Higher model score"
    if score >= 0.40:
        return "Moderate model score"
    return "Lower model score"

def factors(row):
    # Transparent indicator-based descriptions, not causal explanations.
    notes = []
    if row["attendance_rate"] < 70: notes.append("Attendance is below 70%.")
    if row["average_grade"] < 55: notes.append("Average grade is below 55%.")
    if row["backlog_count"] >= 2: notes.append("There are at least two backlogs.")
    if row["lms_logins_per_week"] < 3: notes.append("Learning-platform activity is low.")
    if row["assignment_completion_rate"] < 60: notes.append("Assignment completion is below 60%.")
    if row["fee_delay_days"] > 30: notes.append("Recorded fee delay exceeds 30 days; verify context sensitively.")
    return notes or ["No listed indicator crossed the demo thresholds. Review the full context before deciding on support."]

with tab1:
    st.subheader("Assess a student record")
    st.write("Enter academic/engagement variables. Demographic and program grouping fields are not used as model inputs.")
    with st.form("student_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            attendance = st.number_input(LABELS["attendance_rate"], 0.0, 100.0, 75.0, 1.0, help=HELP["attendance_rate"])
            grade = st.number_input(LABELS["average_grade"], 0.0, 100.0, 65.0, 1.0, help=HELP["average_grade"])
        with c2:
            backlogs = st.number_input(LABELS["backlog_count"], 0, 20, 1, 1, help=HELP["backlog_count"])
            lms = st.number_input(LABELS["lms_logins_per_week"], 0.0, 100.0, 5.0, 0.5, help=HELP["lms_logins_per_week"])
        with c3:
            assignments = st.number_input(LABELS["assignment_completion_rate"], 0.0, 100.0, 75.0, 1.0, help=HELP["assignment_completion_rate"])
            fee_delay = st.number_input(LABELS["fee_delay_days"], 0.0, 365.0, 0.0, 1.0, help=HELP["fee_delay_days"])
        submitted = st.form_submit_button("Estimate model score")
    if submitted:
        row = pd.DataFrame([{
            "attendance_rate": attendance, "average_grade": grade, "backlog_count": backlogs,
            "lms_logins_per_week": lms, "assignment_completion_rate": assignments, "fee_delay_days": fee_delay,
        }], columns=FEATURES)
        score = float(model.predict_proba(row)[0, 1])
        st.metric("Model-estimated dropout risk score", f"{score:.1%}")
        st.write(f"**Score band:** {risk_band(score)}")
        st.caption("This is a model score, not certainty, diagnosis, or a judgment about a student.")
        st.write("**Potential factors to review**")
        for note in factors(row.iloc[0].to_dict()):
            st.write(f"- {note}")
        st.warning("Suggested next step: a trained staff member should verify data quality and ask the student what support, if any, would be useful. Do not automatically deny opportunities or apply punitive action.")

with tab2:
    st.subheader("Capacity-limited support queue")
    if not DATA_PATH.exists():
        st.warning("Dataset missing. Run `python generate_dataset.py`.")
    else:
        data = pd.read_csv(DATA_PATH)
        default_n = min(25, len(data))
        capacity = st.slider("Counsellor capacity (students per week)", 1, min(200, len(data)), default_n)
        # Group fields are kept for audit/display only and never passed to predict_proba.
        scores = model.predict_proba(data[FEATURES])[:, 1]
        queue = data[["student_id"] + FEATURES + ["first_generation_group", "program_track"]].copy()
        queue["model_score"] = scores
        queue["score_band"] = queue["model_score"].map(risk_band)
        queue = queue.sort_values("model_score", ascending=False).head(capacity).reset_index(drop=True)
        queue.insert(0, "queue_rank", np.arange(1, len(queue) + 1))
        st.write(f"Showing the top **{len(queue)}** records by model score, matching the selected weekly capacity.")
        st.dataframe(queue[["queue_rank", "student_id", "model_score", "score_band", "attendance_rate", "average_grade", "backlog_count", "lms_logins_per_week", "assignment_completion_rate"]].style.format({"model_score": "{:.1%}"}), use_container_width=True)
        st.download_button("Download queue CSV", queue.to_csv(index=False).encode("utf-8"), "ranked_support_queue.csv", "text/csv")
        st.caption("A queue rank is only a triage aid. Confirm need, urgency, student preferences, and data accuracy before outreach.")

with tab3:
    st.subheader("Model comparison")
    if report:
        st.write(f"**Selected demo model:** {report['selected_model']}")
        metric_rows = []
        for name, metrics in report["model_metrics"].items():
            metric_rows.append({"Model": name, **{k: metrics[k] for k in ["precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy"]}})
        st.dataframe(pd.DataFrame(metric_rows).set_index("Model").style.format("{:.3f}"), use_container_width=True)
        st.write(f"Dataset: {report['dataset_rows']} synthetic records; holdout test set: {report['test_rows']} records.")
        st.json({"positive_rate": report["positive_rate"], "excluded_audit_fields": report["excluded_audit_fields"]})
    else:
        st.warning("Run `python train_model.py` to generate the report.")

with tab4:
    st.subheader("Fairness audit (test set)")
    if report:
        audit = pd.DataFrame(report["fairness_audit"])
        st.write("The grouping variable is excluded from model inputs and used only to compare observed test-set outcomes.")
        st.dataframe(audit, use_container_width=True)
        st.markdown("""
**How to interpret this audit**
- Compare subgroup recall, precision, and predicted high-risk rates; gaps can reflect different error rates or label prevalence.
- Small subgroup counts make estimates unstable. Inspect counts and uncertainty before drawing conclusions.
- Differences do not by themselves prove discrimination or explain its cause. Review data collection, access to support, label quality, and context with relevant stakeholders.
- Do not change treatment automatically based on group membership. Document concerns and mitigation steps.
""")
    else:
        st.warning("Run `python train_model.py` to generate the audit.")
