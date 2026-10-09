import mysql.connector

try:
    conn = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="",
        database="student_management",
        connection_timeout=5,
    )
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    print("Tables:", cursor.fetchall())
    cursor.close()
    conn.close()
except mysql.connector.Error as e:
    print("Failed:", e)