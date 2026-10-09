from pathlib import Path
import textwrap, zipfile

root = Path("/mnt/data/student-performance-predictor")
(root / "database").mkdir(parents=True, exist_ok=True)
(root / "models").mkdir(parents=True, exist_ok=True)
(root / "src").mkdir(parents=True, exist_ok=True)
(root / "notebooks").mkdir(parents=True, exist_ok=True)

files = {
"requirements.txt": r"""
streamlit
mysql-connector-python
pandas
numpy
scikit-learn
matplotlib
plotly
joblib
""".strip(),

"database/schema.sql": r"""
CREATE DATABASE IF NOT EXISTS student_performance;
USE student_performance;

CREATE TABLE IF NOT EXISTS students (
    student_id VARCHAR(20) PRIMARY KEY,
    student_name VARCHAR(100) NOT NULL,
    class_name VARCHAR(20) NOT NULL,
    section VARCHAR(10) NOT NULL,
    study_hours FLOAT NOT NULL,
    attendance FLOAT NOT NULL,
    previous_marks FLOAT NOT NULL,
    sleep_hours FLOAT NOT NULL,
    assignment_completion FLOAT NOT NULL,
    actual_final_score FLOAT NULL,
    predicted_score FLOAT NULL,
    performance_category VARCHAR(30) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample data. Run only if your table is empty.
INSERT IGNORE INTO students
(student_id, student_name, class_name, section, study_hours, attendance,
 previous_marks, sleep_hours, assignment_completion, actual_final_score)
VALUES
('STU001','Rahul','10','A',6.5,92,78,7,90,84),
('STU002','Priya','10','A',8,95,85,7.5,95,91),
('STU003','Amit','10','A',3,75,60,6,65,58),
('STU004','Neha','10','B',7,90,80,7,88,85),
('STU005','Ravi','10','B',4,80,65,6.5,70,63),
('STU006','Anjali','10','B',9,98,92,8,98,95),
('STU007','Karan','9','A',5,85,70,7,80,72),
('STU008','Pooja','9','A',6,88,75,7.5,85,78),
('STU009','Vikas','9','B',2.5,68,55,6,60,48),
('STU010','Simran','9','B',8.5,96,90,8,95,93),
('STU011','Arjun','9','B',5.5,82,68,7,75,70),
('STU012','Meena','9','A',7.5,91,82,7.5,90,86);
""".strip(),

"database/db_connection.py": r"""
import os
import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "student_performance"),
}

def get_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        raise RuntimeError(
            f"MySQL connection failed: {e}. "
            "Check XAMPP MySQL, database name, username and password."
        )

def fetch_students():
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM students ORDER BY created_at DESC, student_id")
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

def fetch_training_data():
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT student_id, student_name, class_name, section,
                   study_hours, attendance, previous_marks, sleep_hours,
                   assignment_completion, actual_final_score
            FROM students
            WHERE actual_final_score IS NOT NULL
        """)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

def insert_student(data):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        query = """
        INSERT INTO students
        (student_id, student_name, class_name, section, study_hours,
         attendance, previous_marks, sleep_hours, assignment_completion,
         actual_final_score, predicted_score, performance_category)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        cursor.execute(query, tuple(data))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def update_student(student_id, data):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        query = """
        UPDATE students
        SET student_name=%s, class_name=%s, section=%s, study_hours=%s,
            attendance=%s, previous_marks=%s, sleep_hours=%s,
            assignment_completion=%s, actual_final_score=%s
        WHERE student_id=%s
        """
        cursor.execute(query, tuple(data) + (student_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def delete_student(student_id):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE student_id=%s", (student_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()
""".strip(),

"src/data_preprocessing.py": r"""
import pandas as pd

FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks",
    "sleep_hours",
    "assignment_completion",
]
TARGET = "actual_final_score"

def clean_training_data(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = FEATURES + [TARGET]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()

    # Keep only valid project ranges.
    df = df[
        df["study_hours"].between(0, 24)
        & df["sleep_hours"].between(0, 24)
        & df["attendance"].between(0, 100)
        & df["previous_marks"].between(0, 100)
        & df["assignment_completion"].between(0, 100)
        & df[TARGET].between(0, 100)
    ]

    return df
""".strip(),

"src/analytics.py": r"""
import pandas as pd

def category(score):
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 50:
        return "Average"
    return "Needs Improvement"

def add_category(df, score_column="actual_final_score"):
    result = df.copy()
    if score_column in result.columns:
        result["performance_category"] = result[score_column].apply(category)
    return result

def summary(df):
    if df.empty:
        return {
            "total": 0, "classes": 0, "average": 0,
            "highest": 0, "lowest": 0, "pass_pct": 0
        }

    scores = pd.to_numeric(df["actual_final_score"], errors="coerce").dropna()
    return {
        "total": len(df),
        "classes": df["class_name"].nunique(),
        "average": scores.mean() if len(scores) else 0,
        "highest": scores.max() if len(scores) else 0,
        "lowest": scores.min() if len(scores) else 0,
        "pass_pct": ((scores >= 50).mean() * 100) if len(scores) else 0,
    }
""".strip(),

"src/model_training.py": r"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from database.db_connection import fetch_training_data
from src.data_preprocessing import clean_training_data, FEATURES, TARGET

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
LINEAR_PATH = MODEL_DIR / "linear_model.pkl"
RF_PATH = MODEL_DIR / "random_forest_model.pkl"
SELECTED_PATH = MODEL_DIR / "selected_model.pkl"

def train_models():
    rows = fetch_training_data()
    df = clean_training_data(rows)

    if len(df) < 6:
        raise ValueError("At least 6 valid historical records are recommended for training.")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    linear = LinearRegression()
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=6,
        min_samples_leaf=1
    )

    linear.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    results = {}
    for name, model in [("Linear Regression", linear), ("Random Forest", rf)]:
        pred = model.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred),
        }

    # Select by lowest MAE, then lowest RMSE, then highest R2.
    selected_name = min(
        results,
        key=lambda n: (results[n]["MAE"], results[n]["RMSE"], -results[n]["R2"])
    )
    selected_model = linear if selected_name == "Linear Regression" else rf

    joblib.dump(linear, LINEAR_PATH)
    joblib.dump(rf, RF_PATH)
    joblib.dump(selected_model, SELECTED_PATH)

    return results, selected_name, X_test, y_test, linear, rf

def load_selected_model():
    if not SELECTED_PATH.exists():
        return None
    return joblib.load(SELECTED_PATH)
""".strip(),

"app.py": r"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from database.db_connection import (
    fetch_students, insert_student, update_student, delete_student
)
from src.analytics import category, add_category, summary
from src.data_preprocessing import FEATURES
from src.model_training import train_models, load_selected_model

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor & Class Analytics")
st.caption("Streamlit + MySQL + Machine Learning")

# ---------- Helpers ----------
def validate_student(sid, name, class_name, section, study, attendance,
                     previous, sleep, assignments, final_score):
    errors = []
    if not sid.strip():
        errors.append("Student ID is required.")
    if not name.strip():
        errors.append("Student name is required.")
    if not class_name.strip():
        errors.append("Class is required.")
    if not section.strip():
        errors.append("Section is required.")
    if not 0 <= study <= 24:
        errors.append("Study hours must be between 0 and 24.")
    if not 0 <= sleep <= 24:
        errors.append("Sleep hours must be between 0 and 24.")
    for label, value in [
        ("Attendance", attendance),
        ("Previous marks", previous),
        ("Assignment completion", assignments),
        ("Final score", final_score),
    ]:
        if not 0 <= value <= 100:
            errors.append(f"{label} must be between 0 and 100.")
    return errors

def load_df():
    rows = fetch_students()
    df = pd.DataFrame(rows)
    if not df.empty:
        numeric = [
            "study_hours", "attendance", "previous_marks",
            "sleep_hours", "assignment_completion",
            "actual_final_score", "predicted_score"
        ]
        for col in numeric:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = add_category(df)
    return df

# ---------- Sidebar ----------
st.sidebar.header("📚 Navigation")
page = st.sidebar.radio(
    "Select page",
    [
        "Dashboard",
        "Student Prediction",
        "Student Management",
        "Student List",
        "Class Analytics",
        "Class Comparison",
        "ML Model Evaluation",
    ]
)

try:
    df = load_df()
except Exception as e:
    st.error(str(e))
    st.info("Start XAMPP MySQL and create the student_performance database/table using database/schema.sql.")
    st.stop()

# ---------- Dashboard ----------
if page == "Dashboard":
    st.header("🏠 Overview Dashboard")
    s = summary(df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Students", s["total"])
    c2.metric("Total Classes", s["classes"])
    c3.metric("Average Score", f"{s['average']:.2f}")
    c4.metric("Highest", f"{s['highest']:.2f}")
    c5.metric("Lowest", f"{s['lowest']:.2f}")
    c6.metric("Pass %", f"{s['pass_pct']:.1f}%")

    if not df.empty:
        left, right = st.columns(2)
        with left:
            cat_counts = df["performance_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Students"]
            st.plotly_chart(
                px.pie(cat_counts, names="Category", values="Students",
                       title="Performance Categories"),
                use_container_width=True
            )
        with right:
            st.plotly_chart(
                px.histogram(df, x="actual_final_score", nbins=10,
                             title="Final Score Distribution"),
                use_container_width=True
            )

        st.subheader("🏆 Top Students")
        st.dataframe(
            df[["student_id", "student_name", "class_name", "section",
                "actual_final_score", "performance_category"]]
            .sort_values("actual_final_score", ascending=False)
            .head(10),
            use_container_width=True
        )

# ---------- Prediction ----------
elif page == "Student Prediction":
    st.header("🤖 Student Score Prediction")

    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            study = st.number_input("Study Hours / Day", 0.0, 24.0, 5.0, 0.5)
            attendance = st.number_input("Attendance %", 0.0, 100.0, 85.0, 1.0)
        with c2:
            previous = st.number_input("Previous Marks", 0.0, 100.0, 70.0, 1.0)
            sleep = st.number_input("Sleep Hours / Day", 0.0, 24.0, 7.0, 0.5)
        with c3:
            assignments = st.number_input("Assignment Completion %", 0.0, 100.0, 80.0, 1.0)
            student_id = st.text_input("Student ID", placeholder="STU013")
            student_name = st.text_input("Student Name", placeholder="New Student")

        predict = st.form_submit_button("🔮 Predict Score")

    if predict:
        model = load_selected_model()
        if model is None:
            st.warning("No trained model found. Train the model from ML Model Evaluation first.")
        else:
            input_df = pd.DataFrame([{
                "study_hours": study,
                "attendance": attendance,
                "previous_marks": previous,
                "sleep_hours": sleep,
                "assignment_completion": assignments
            }])
            prediction = float(model.predict(input_df)[0])
            prediction = max(0.0, min(100.0, prediction))
            st.session_state.prediction = prediction

    if st.session_state.prediction is not None:
        prediction = st.session_state.prediction
        cat = category(prediction)
        st.success(f"Predicted Final Score: {prediction:.2f}/100")
        st.info(f"Performance Category: {cat}")

        if student_id.strip() and student_name.strip():
            if st.button("💾 Save Prediction to MySQL"):
                try:
                    data = [
                        student_id.strip(), student_name.strip(), "Prediction", "A",
                        study, attendance, previous, sleep, assignments,
                        None, prediction, cat
                    ]
                    insert_student(data)
                    st.success("Prediction record saved to MySQL.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not save record: {e}")

# ---------- Student Management ----------
elif page == "Student Management":
    st.header("👨‍🎓 Student Management")
    tab1, tab2, tab3 = st.tabs(["Add", "Update", "Delete"])

    with tab1:
        with st.form("add_student"):
            c1, c2 = st.columns(2)
            with c1:
                sid = st.text_input("Student ID")
                name = st.text_input("Student Name")
                class_name = st.text_input("Class")
                section = st.text_input("Section")
                study = st.number_input("Study Hours", 0.0, 24.0, 5.0, 0.5)
                attendance = st.number_input("Attendance %", 0.0, 100.0, 85.0)
            with c2:
                previous = st.number_input("Previous Marks", 0.0, 100.0, 70.0)
                sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0, 0.5)
                assignments = st.number_input("Assignment Completion %", 0.0, 100.0, 80.0)
                final_score = st.number_input("Actual Final Score", 0.0, 100.0, 70.0)
            submitted = st.form_submit_button("➕ Add Student")

        if submitted:
            errors = validate_student(
                sid, name, class_name, section, study, attendance,
                previous, sleep, assignments, final_score
            )
            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    data = [
                        sid.strip(), name.strip(), class_name.strip(), section.strip(),
                        study, attendance, previous, sleep, assignments,
                        final_score, None, category(final_score)
                    ]
                    insert_student(data)
                    st.success("Student added successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Insert failed: {e}")

    with tab2:
        if df.empty:
            st.info("No students available.")
        else:
            selected = st.selectbox("Select Student ID", df["student_id"].tolist())
            row = df[df["student_id"] == selected].iloc[0]

            with st.form("update_student"):
                name = st.text_input("Student Name", value=str(row["student_name"]))
                class_name = st.text_input("Class", value=str(row["class_name"]))
                section = st.text_input("Section", value=str(row["section"]))
                study = st.number_input("Study Hours", 0.0, 24.0, float(row["study_hours"]), 0.5)
                attendance = st.number_input("Attendance %", 0.0, 100.0, float(row["attendance"]))
                previous = st.number_input("Previous Marks", 0.0, 100.0, float(row["previous_marks"]))
                sleep = st.number_input("Sleep Hours", 0.0, 24.0, float(row["sleep_hours"]), 0.5)
                assignments = st.number_input("Assignment Completion %", 0.0, 100.0, float(row["assignment_completion"]))
                final_score = st.number_input("Actual Final Score", 0.0, 100.0,
                                              float(row["actual_final_score"] or 0))
                save = st.form_submit_button("✏️ Update Student")

            if save:
                errors = validate_student(
                    selected, name, class_name, section, study, attendance,
                    previous, sleep, assignments, final_score
                )
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    try:
                        update_student(selected, [
                            name.strip(), class_name.strip(), section.strip(),
                            study, attendance, previous, sleep, assignments,
                            final_score
                        ])
                        st.success("Student updated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Update failed: {e}")

    with tab3:
        if df.empty:
            st.info("No students available.")
        else:
            selected = st.selectbox("Student to delete", df["student_id"].tolist(), key="delete_id")
            st.warning("Deleting a student permanently removes the record.")
            if st.button("🗑️ Delete Student"):
                try:
                    delete_student(selected)
                    st.success("Student deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")

# ---------- Student List ----------
elif page == "Student List":
    st.header("📋 Student List")

    if df.empty:
        st.info("No student records.")
    else:
        c1, c2, c3 = st.columns(3)
        search = c1.text_input("Search Name / ID")
        class_filter = c2.selectbox("Class", ["All"] + sorted(df["class_name"].astype(str).unique().tolist()))
        section_filter = c3.selectbox("Section", ["All"] + sorted(df["section"].astype(str).unique().tolist()))

        filtered = df.copy()
        if search:
            mask = (
                filtered["student_name"].astype(str).str.contains(search, case=False, na=False)
                | filtered["student_id"].astype(str).str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask]
        if class_filter != "All":
            filtered = filtered[filtered["class_name"].astype(str) == class_filter]
        if section_filter != "All":
            filtered = filtered[filtered["section"].astype(str) == section_filter]

        order = st.radio("Sort", ["Highest Score", "Lowest Score", "Student Name"], horizontal=True)
        if order == "Highest Score":
            filtered = filtered.sort_values("actual_final_score", ascending=False)
        elif order == "Lowest Score":
            filtered = filtered.sort_values("actual_final_score", ascending=True)
        else:
            filtered = filtered.sort_values("student_name")

        st.dataframe(filtered, use_container_width=True)

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "student_report.csv", "text/csv")

# ---------- Class Analytics ----------
elif page == "Class Analytics":
    st.header("📊 Class Analytics")

    if df.empty:
        st.info("No data.")
    else:
        classes = sorted(df["class_name"].astype(str).unique())
        selected_class = st.selectbox("Select Class", classes)
        class_df = df[df["class_name"].astype(str) == selected_class].copy()

        s = summary(class_df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Students", s["total"])
        c2.metric("Average", f"{s['average']:.2f}")
        c3.metric("Highest", f"{s['highest']:.2f}")
        c4.metric("Pass %", f"{s['pass_pct']:.1f}%")

        st.subheader("Section Comparison")
        section_stats = (
            class_df.groupby("section", as_index=False)
            .agg(
                Students=("student_id", "count"),
                Average=("actual_final_score", "mean"),
                Highest=("actual_final_score", "max"),
                Lowest=("actual_final_score", "min")
            )
        )
        st.dataframe(section_stats, use_container_width=True)

        st.plotly_chart(
            px.bar(class_df, x="student_name", y="actual_final_score",
                   color="section", title=f"Scores - Class {selected_class}"),
            use_container_width=True
        )

        st.subheader("Performance Categories")
        cat_counts = class_df["performance_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Students"]
        st.plotly_chart(
            px.pie(cat_counts, names="Category", values="Students"),
            use_container_width=True
        )

# ---------- Class Comparison ----------
elif page == "Class Comparison":
    st.header("🏫 Class Comparison")

    if df.empty:
        st.info("No data.")
    else:
        comparison = (
            df.groupby("class_name", as_index=False)
            .agg(
                Students=("student_id", "count"),
                Average_Score=("actual_final_score", "mean"),
                Highest_Score=("actual_final_score", "max"),
                Lowest_Score=("actual_final_score", "min")
            )
        )

        pass_rates = (
            df.assign(passed=df["actual_final_score"] >= 50)
            .groupby("class_name")["passed"].mean()
            .mul(100)
            .reset_index(name="Pass_Percentage")
        )
        comparison = comparison.merge(pass_rates, on="class_name")

        st.dataframe(comparison, use_container_width=True)
        st.plotly_chart(
            px.bar(comparison, x="class_name", y="Average_Score",
                   title="Average Score by Class"),
            use_container_width=True
        )
        st.plotly_chart(
            px.bar(comparison, x="class_name", y="Pass_Percentage",
                   title="Pass Percentage by Class"),
            use_container_width=True
        )

# ---------- ML Evaluation ----------
elif page == "ML Model Evaluation":
    st.header("🧠 Machine Learning Model Evaluation")
    st.write("Models are trained only with records having an actual final score.")

    if st.button("🚀 Train / Retrain Models"):
        with st.spinner("Training models..."):
            try:
                results, selected, X_test, y_test, linear, rf = train_models()
                st.session_state["ml_results"] = results
                st.session_state["selected_model"] = selected
                st.session_state["test_data"] = (X_test, y_test, linear, rf)
                st.success(f"Training complete. Selected model: {selected}")
            except Exception as e:
                st.error(str(e))

    if "ml_results" in st.session_state:
        results = st.session_state["ml_results"]
        metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
        st.subheader("Evaluation Metrics")
        st.dataframe(metrics_df, use_container_width=True)

        selected = st.session_state["selected_model"]
        st.info(f"Selected model: {selected}")

        X_test, y_test, linear, rf = st.session_state["test_data"]
        model = linear if selected == "Linear Regression" else rf
        pred = model.predict(X_test)

        chart_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": pred
        })
        st.plotly_chart(
            px.scatter(chart_df, x="Actual", y="Predicted",
                       title="Actual vs Predicted Scores",
                       trendline="ols"),
            use_container_width=True
        )

        st.subheader("Feature Importance")
        if selected == "Random Forest":
            importance = rf.feature_importances_
        else:
            importance = abs(linear.coef_)

        fi = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": importance
        }).sort_values("Importance", ascending=False)

        st.plotly_chart(
            px.bar(fi, x="Importance", y="Feature", orientation="h",
                   title=f"Feature Importance - {selected}"),
            use_container_width=True
        )
""".strip(),

"requirements.txt": r"""
"database/schema.sql": r"""
"README.md": r"""
# Student Performance Predictor & Class Analytics System

## Stack
- Python
- Streamlit
- MySQL / XAMPP
- phpMyAdmin
- Pandas / NumPy
- Scikit-learn
- Plotly
- Joblib

## 1. Start XAMPP
Start **MySQL** from XAMPP. Apache is only required if you want to use phpMyAdmin through the web interface.

## 2. Create the database
Open phpMyAdmin -> SQL and run:

```sql
CREATE DATABASE IF NOT EXISTS student_performance;