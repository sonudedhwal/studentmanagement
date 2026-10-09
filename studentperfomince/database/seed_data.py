import numpy as np
from database.db_connection import get_connection

np.random.seed(42)
N = 200

def category(score):
    if score >= 90: return "Excellent"
    if score >= 75: return "Good"
    if score >= 50: return "Average"
    return "Needs Improvement"

names = ["Rahul","Priya","Amit","Sneha","Karan","Anjali","Vikram","Neha",
         "Arjun","Pooja","Rohan","Divya","Manish","Kavya","Suresh","Isha"]

rows = []
for i in range(1, N + 1):
    study  = round(np.clip(np.random.normal(4.5, 1.8), 0.5, 10), 1)
    attend = round(np.clip(np.random.normal(85, 10), 40, 100), 1)
    prev   = round(np.clip(np.random.normal(65, 15), 20, 100), 1)
    sleep  = round(np.clip(np.random.normal(7, 1), 4, 10), 1)
    assign = round(np.clip(np.random.normal(80, 15), 20, 100), 1)

    # Hidden rule that generates the score, plus noise so the ML task is realistic
    score = (0.35 * prev + 0.25 * attend + 0.20 * assign
             + 2.5 * study + 0.8 * sleep + np.random.normal(0, 4) - 12)
    score = round(float(np.clip(score, 0, 100)), 1)

    rows.append((
        f"STU{i:03d}",
        f"{np.random.choice(names)} {chr(65 + i % 26)}.",
        str(np.random.choice(["8", "9", "10"])),
        str(np.random.choice(["A", "B", "C"])),
        study, attend, prev, sleep, assign,
        score, None, category(score)
    ))

conn = get_connection()
cur = conn.cursor()
cur.execute("DELETE FROM students")  # clears the 4 test rows
cur.executemany("""
    INSERT INTO students
    (student_id, student_name, class, section, study_hours, attendance,
     previous_marks, sleep_hours, assignment_completion,
     actual_final_score, predicted_score, performance_category)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
""", rows)
conn.commit()
print(f"Inserted {cur.rowcount} students")
cur.close(); conn.close()