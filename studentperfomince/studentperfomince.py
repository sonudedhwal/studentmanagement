"""Student Performance Predictor & Class Analytics  (run:  streamlit run student_app.py)"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
sns.set_theme(style="whitegrid")

FEATURES = ["study_hours", "attendance", "previous_score", "sleep_hours", "assignments_done"]
TARGET = "final_score"


# ------------------------------------------------------------------ data
@st.cache_data
def make_data(n, seed=42):
    """Synthetic class data: final score depends on the features plus noise."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "student_id": [f"S{i:03d}" for i in range(1, n + 1)],
        "study_hours": rng.uniform(0, 10, n).round(1),
        "attendance": rng.uniform(50, 100, n).round(0),
        "previous_score": np.clip(rng.normal(65, 15, n), 20, 100).round(0),
        "sleep_hours": np.clip(rng.normal(7, 1, n), 4, 10).round(1),
        "assignments_done": rng.integers(0, 11, n),
    })
    df[TARGET] = np.clip(
        0.35 * df.previous_score + 3.0 * df.study_hours + 0.25 * df.attendance
        + 1.0 * df.assignments_done + rng.normal(0, 5, n), 0, 100).round(0)
    return df


def load_data():
    st.sidebar.header("📂 Data")
    source = st.sidebar.radio("Source", ["Sample data", "Upload my CSV"])
    if source == "Sample data":
        return make_data(st.sidebar.slider("Number of students", 50, 500, 200, step=50))
    st.sidebar.caption("CSV needs columns: " + ", ".join(FEATURES + [TARGET]))
    up = st.sidebar.file_uploader("CSV file", type=["csv"])
    if up is None:
        return None
    df = pd.read_csv(up)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return None
    if "student_id" not in df.columns:
        df.insert(0, "student_id", [f"S{i:03d}" for i in range(1, len(df) + 1)])
    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


df = load_data()
st.title("🎓 Student Performance Predictor & Class Analytics")
if df is None:
    st.info("Choose *Sample data* or upload a CSV in the sidebar to begin.")
    st.stop()

pass_mark = st.sidebar.slider("Pass mark", 30, 70, 50)
st.sidebar.download_button("⬇️ Download this data as CSV", df.to_csv(index=False).encode("utf-8"),
                           "students.csv", "text/csv")
df["passed"] = (df[TARGET] >= pass_mark).astype(int)
if df["passed"].nunique() < 2:
    st.warning("All students are on the same side of the pass mark. Move the pass mark slider.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Class analytics", "🤖 Train & compare models",
                                  "🔮 Predict a student", "⚠️ At-risk students"])

# ------------------------------------------------------- 1. class analytics
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", len(df))
    c2.metric("Average score", f"{df[TARGET].mean():.1f}")
    c3.metric("Pass rate", f"{df.passed.mean() * 100:.0f}%")
    c4.metric("Top score", f"{df[TARGET].max():.0f}")

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.histplot(df[TARGET], bins=15, kde=True, ax=ax)
        ax.axvline(pass_mark, color="red", linestyle="--", label=f"Pass mark ({pass_mark})")
        ax.set_title("Score distribution"); ax.legend()
        fig.tight_layout(); st.pyplot(fig)
    with b:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7})
        ax.set_title("What correlates with the final score?")
        fig.tight_layout(); st.pyplot(fig)

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.regplot(data=df, x="study_hours", y=TARGET, scatter_kws={"s": 20},
                    line_kws={"color": "red"}, ax=ax)
        ax.set_title("Study hours vs final score")
        fig.tight_layout(); st.pyplot(fig)
    with b:
        band = pd.cut(df.attendance, [0, 60, 70, 80, 90, 100], include_lowest=True).astype(str)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(x=band, y=df[TARGET], order=sorted(band.unique()), ax=ax)
        ax.set_xlabel("Attendance %"); ax.set_title("Score by attendance band")
        fig.tight_layout(); st.pyplot(fig)

    with st.expander("See the data"):
        st.dataframe(df)

# ------------------------------------------------------ 2. train & compare
with tab2:
    st.write("**Supervised learning:** the data has known answers, so models learn to predict them. "
             "*Regression* predicts the score; *classification* predicts pass / fail.")
    task = st.radio("What to predict", ["Final score (regression)", "Pass / Fail (classification)"],
                    horizontal=True)
    test_size = st.slider("Test set size (%)", 10, 50, 25, step=5) / 100
    X = df[FEATURES]

    if task.startswith("Final"):
        algo = st.selectbox("Algorithm", ["Linear Regression", "Decision Tree"])
        depth = st.slider("max_depth", 1, 12, 3) if algo == "Decision Tree" else None
        y = df[TARGET]
        model = (make_pipeline(StandardScaler(), LinearRegression()) if algo == "Linear Regression"
                 else DecisionTreeRegressor(max_depth=depth, random_state=42))
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
        model.fit(X_tr, y_tr)
        p_tr, p_te = model.predict(X_tr), model.predict(X_te)

        m1, m2, m3 = st.columns(3)
        m1.metric("Train R²", f"{r2_score(y_tr, p_tr):.3f}")
        m2.metric("Test R²", f"{r2_score(y_te, p_te):.3f}")
        m3.metric("Test error (MAE, marks)", f"{mean_absolute_error(y_te, p_te):.2f}")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(x=y_te, y=p_te, ax=ax)
        ax.plot([0, 100], [0, 100], "r--", label="Perfect")
        ax.set_xlabel("Actual score"); ax.set_ylabel("Predicted score"); ax.legend()
        left = st.columns(2)
        left[0].pyplot(fig)
    else:
        algo = st.selectbox("Algorithm", ["Logistic Regression", "KNN", "Decision Tree"])
        k = st.slider("K (neighbours)", 1, 30, 5) if algo == "KNN" else None
        depth = st.slider("max_depth", 1, 12, 3) if algo == "Decision Tree" else None
        y = df["passed"]
        if algo == "Logistic Regression":
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        elif algo == "KNN":
            model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        else:
            model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        model.fit(X_tr, y_tr)
        p_tr, p_te = model.predict(X_tr), model.predict(X_te)
        cv = cross_val_score(model, X_tr, y_tr, cv=5)

        m1, m2, m3 = st.columns(3)
        m1.metric("Train accuracy", f"{accuracy_score(y_tr, p_tr):.3f}")
        m2.metric("Test accuracy", f"{accuracy_score(y_te, p_te):.3f}")
        m3.metric("5-fold CV", f"{cv.mean():.3f} ± {cv.std():.3f}")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_te, p_te), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test set)")
        left = st.columns(2)
        left[0].pyplot(fig)
        if accuracy_score(y_tr, p_tr) - accuracy_score(y_te, p_te) > 0.08:
            st.warning("Train accuracy is much higher than test accuracy: likely **overfitting**. "
                       "Try a smaller max_depth or a larger K.")

    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "coef_") or hasattr(est, "feature_importances_"):
        vals = np.ravel(est.coef_) if hasattr(est, "coef_") else est.feature_importances_
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=vals, y=FEATURES, ax=ax)
        ax.set_title("Which factors matter most?")
        left[1].pyplot(fig)

# ------------------------------------------------------ 3. predict a student
with tab3:
    st.write("Enter a student's details to predict their final score and pass chance.")
    reg = make_pipeline(StandardScaler(), LinearRegression()).fit(df[FEATURES], df[TARGET])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(df[FEATURES], df["passed"])

    c = st.columns(5)
    vals = {
        "study_hours": c[0].slider("Study hours / day", 0.0, 10.0, 5.0, 0.5),
        "attendance": c[1].slider("Attendance %", 40, 100, 80),
        "previous_score": c[2].slider("Previous score", 0, 100, 65),
        "sleep_hours": c[3].slider("Sleep hours", 3.0, 10.0, 7.0, 0.5),
        "assignments_done": c[4].slider("Assignments done (of 10)", 0, 10, 6),
    }
    row = pd.DataFrame([vals])[FEATURES]
    score = float(np.clip(reg.predict(row)[0], 0, 100))
    p_pass = float(clf.predict_proba(row)[0][1])

    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted final score", f"{score:.0f}")
    r2.metric("Chance of passing", f"{p_pass * 100:.0f}%")
    r3.metric("Verdict", "✅ Likely pass" if p_pass >= 0.5 else "⚠️ At risk")

    st.subheader("What-if: small changes")
    for label, change in [("+1 study hour", {"study_hours": 1}), ("+10% attendance", {"attendance": 10}),
                          ("+2 assignments", {"assignments_done": 2})]:
        alt = row.copy()
        for col, d in change.items():
            alt[col] = alt[col] + d
        gain = float(np.clip(reg.predict(alt)[0], 0, 100)) - score
        st.write(f"- **{label}** → about **{gain:+.1f}** marks")

# ----------------------------------------------------------- 4. at-risk list
with tab4:
    buffer = st.slider("Flag students predicted within this many marks of the pass mark", 0, 20, 10)
    reg_all = make_pipeline(StandardScaler(), LinearRegression()).fit(df[FEATURES], df[TARGET])
    out = df.copy()
    out["predicted_score"] = np.clip(reg_all.predict(df[FEATURES]), 0, 100).round(1)
    risk = out[out.predicted_score < pass_mark + buffer].sort_values("predicted_score")
    st.metric("Students at risk", f"{len(risk)} of {len(out)}")
    if risk.empty:
        st.success("No students flagged.")
    else:
        st.dataframe(risk[["student_id"] + FEATURES + ["predicted_score"]])
        st.download_button("⬇️ Download at-risk list", risk.to_csv(index=False).encode("utf-8"),
                           "at_risk_students.csv", "text/csv")