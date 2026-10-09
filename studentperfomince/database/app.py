import streamlit as st
from database.db_connection import load_students

st.set_page_config(page_title="Student Analytics", layout="wide")
st.title("Student Performance Analytics")

df = load_students()

# Overview metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Students", len(df))
c2.metric("Classes", df["class"].nunique())
c3.metric("Average Score", f"{df['actual_final_score'].mean():.1f}")
c4.metric("Pass %", f"{(df['actual_final_score'] >= 50).mean() * 100:.1f}%")

# Filters
st.sidebar.header("Filters")
cls = st.sidebar.multiselect("Class", sorted(df["class"].unique()))
sec = st.sidebar.multiselect("Section", sorted(df["section"].unique()))
search = st.sidebar.text_input("Search name or ID")

if cls: df = df[df["class"].isin(cls)]
if sec: df = df[df["section"].isin(sec)]
if search:
    df = df[df["student_name"].str.contains(search, case=False)
            | df["student_id"].str.contains(search, case=False)]

st.dataframe(df.sort_values("actual_final_score", ascending=False),
             use_container_width=True)