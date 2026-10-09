This is the Streamlit application.
import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression

from database import (
    add_student,
    get_students,
    search_students,
    delete_student,
    update_prediction
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Management System",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🎓 Student Management System")

st.write(
    "Python + Streamlit + MySQL + XAMPP + AI/ML"
)


# ==========================================
# SIDEBAR
# ==========================================

menu = st.sidebar.selectbox(
    "Select Module",
    [
        "Dashboard",
        "Add Student",
        "View Students",
        "Search Student",
        "Delete Student",
        "AI/ML Prediction"
    ]
)


# ==========================================
# DASHBOARD
# ==========================================

if menu == "Dashboard":

    st.header("📊 Student Dashboard")

    try:

        df = get_students()

        total_students = len(df)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Students",
            total_students
        )

        if len(df) > 0:

            average_marks = df["previous_marks"].mean()

            average_attendance = df["attendance"].mean()

            average_study = df["study_hours"].mean()

        else:

            average_marks = 0
            average_attendance = 0
            average_study = 0

        col2.metric(
            "Average Marks",
            f"{average_marks:.2f}"
        )

        col3.metric(
            "Average Attendance",
            f"{average_attendance:.2f}%"
        )

        col4.metric(
            "Average Study Hours",
            f"{average_study:.2f}"
        )

        st.subheader("Student Data")

        st.dataframe(
            df,
            use_container_width=True
        )

    except Exception as e:

        st.error(f"Database Error: {e}")


# ==========================================
# ADD STUDENT
# ==========================================

elif menu == "Add Student":

    st.header("➕ Add New Student")

    name = st.text_input("Student Name")

    email = st.text_input("Email")

    phone = st.text_input("Phone")

    class_name = st.text_input("Class")

    section = st.text_input("Section")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=18
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

    study_hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=2.0
    )

    previous_marks = st.number_input(
        "Previous Marks",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    assignment_completion = st.number_input(
        "Assignment Completion (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    if st.button("Save Student"):

        if name.strip() == "":

            st.warning("Please enter student name.")

        else:

            try:

                add_student(
                    name,
                    email,
                    phone,
                    class_name,
                    section,
                    gender,
                    age,
                    attendance,
                    study_hours,
                    previous_marks,
                    assignment_completion
                )

                st.success(
                    "Student added successfully!"
                )

            except Exception as e:

                st.error(
                    f"Database Error: {e}"
                )


# ==========================================
# VIEW STUDENTS
# ==========================================

elif menu == "View Students":

    st.header("👨🎓 All Students")

    try:

        df = get_students()

        if len(df) == 0:

            st.info("No students found.")

        else:

            st.dataframe(
                df,
                use_container_width=True
            )

            st.download_button(
                "Download Student Data",
                df.to_csv(index=False),
                "students.csv",
                "text/csv"
            )

    except Exception as e:

        st.error(f"Database Error: {e}")


# ==========================================
# SEARCH STUDENT
# ==========================================

elif menu == "Search Student":

    st.header("🔎 Search Student")

    name = st.text_input(
        "Enter Student Name"
    )

    if st.button("Search"):

        try:

            results = search_students(name)

            if results:

                st.dataframe(
                    pd.DataFrame(results),
                    use_container_width=True
                )

            else:

                st.warning(
                    "Student not found."
                )

        except Exception as e:

            st.error(
                f"Database Error: {e}"
            )


# ==========================================
# DELETE STUDENT
# ==========================================

elif menu == "Delete Student":

    st.header("🗑️ Delete Student")

    student_id = st.number_input(
        "Student ID",
        min_value=1,
        step=1
    )

    if st.button("Delete"):

        try:

            delete_student(student_id)

            st.success(
                "Student deleted successfully."
            )

        except Exception as e:

            st.error(
                f"Database Error: {e}"
            )


# ==========================================
# AI / ML PREDICTION
# ==========================================

elif menu == "AI/ML Prediction":

    st.header("🤖 AI / ML Student Performance Prediction")

    st.write(
        "Linear Regression predicts the student's final score."
    )

    try:

        df = get_students()

        if len(df) < 3:

            st.warning(
                "Add at least 3 students before training the ML model."
            )

        else:

            required_columns = [
                "attendance",
                "study_hours",
                "previous_marks",
                "assignment_completion"
            ]

            training_df = df.dropna(
                subset=required_columns
            )

            if len(training_df) < 3:

                st.warning(
                    "Not enough training data."
                )

            else:

                # -----------------------------
                # Training Data
                # -----------------------------

                X = training_df[
                    [
                        "attendance",
                        "study_hours",
                        "previous_marks",
                        "assignment_completion"
                    ]
                ]

                # If actual final score exists,
                # train using it.
                if (
                    "final_score" in training_df.columns
                    and training_df["final_score"].notna().sum() >= 3
                ):

                    train_data = training_df.dropna(
                        subset=["final_score"]
                    )

                    X = train_data[
                        [
                            "attendance",
                            "study_hours",
                            "previous_marks",
                            "assignment_completion"
                        ]
                    ]

                    y = train_data["final_score"]

                    model = LinearRegression()

                    model.fit(X, y)

                    st.success(
                        "ML model trained successfully."
                    )

                    st.subheader(
                        "Predict Student Score"
                    )

                    attendance = st.number_input(
                        "Attendance",
                        0.0,
                        100.0,
                        75.0
                    )

                    study_hours = st.number_input(
                        "Study Hours",
                        0.0,
                        24.0,
                        3.0
                    )

                    previous_marks = st.number_input(
                        "Previous Marks",
                        0.0,
                        100.0,
                        60.0
                    )

                    assignment_completion = st.number_input(
                        "Assignment Completion",
                        0.0,
                        100.0,
                        80.0
                    )

                    student_id = st.number_input(
                        "Student ID",
                        min_value=1,
                        step=1
                    )

                    if st.button("Predict Final Score"):

                        input_data = pd.DataFrame(
                            [[
                                attendance,
                                study_hours,
                                previous_marks,
                                assignment_completion
                            ]],
                            columns=[
                                "attendance",
                                "study_hours",
                                "previous_marks",
                                "assignment_completion"
                            ]
                        )

                        prediction = model.predict(
                            input_data
                        )[0]

                        prediction = max(
                            0,
                            min(100, prediction)
                        )

                        st.success(
                            f"Predicted Final Score: {prediction:.2f}"
                        )

                        update_prediction(
                            student_id,
                            prediction
                        )

                        st.info(
                            "Prediction saved to MySQL."
                        )

                else:

                    st.info(
                        "Enter actual final scores for at least 3 students "
                        "to train the supervised ML model."
                    )

    except Exception as e:

        st.error(
            f"ML / Database Error: {e}"
        )
