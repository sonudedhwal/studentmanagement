import os

import mysql.connector
import pandas as pd


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "student_management"),
    )


def add_student(name, email, phone, class_name, section, gender, age,
                attendance, study_hours, previous_marks, assignment_completion):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO students
        (student_name, email, phone, class_name, section, gender, age,
         attendance, study_hours, previous_marks, assignment_completion)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (name, email, phone, class_name, section, gender, age,
         attendance, study_hours, previous_marks, assignment_completion),
    )
    conn.commit()
    cursor.close()
    conn.close()


def get_students():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM students ORDER BY student_id DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows)


def search_students(name):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM students WHERE student_name LIKE %s", (f"%{name}%",)
    )
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data


def delete_student(student_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students WHERE student_id = %s", (student_id,))
    conn.commit()
    cursor.close()
    conn.close()


def update_final_score(student_id, score):
    """Save the REAL final score (used to train the model)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE students SET final_score = %s WHERE student_id = %s",
        (score, student_id),
    )
    conn.commit()
    cursor.close()
    conn.close()


def update_prediction(student_id, score):
    """Save the model's PREDICTED score in its own column."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE students SET predicted_score = %s WHERE student_id = %s",
        (score, student_id),
    )
    conn.commit()
    cursor.close()
    conn.close()