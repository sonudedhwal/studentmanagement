import mysql.connector
import pandas as pd

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",            # XAMPP default is empty
        database="student_analytics"
    )

def load_students():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM students", conn)
    conn.close()
    return df

def insert_student(data: tuple):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO students
        (student_id, student_name, class, section, study_hours, attendance,
         previous_marks, sleep_hours, assignment_completion,
         actual_final_score, predicted_score, performance_category)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, data)
    conn.commit()
    cur.close()
    conn.close()