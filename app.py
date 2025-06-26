import streamlit as st
import os
import pandas as pd
from your_module import (
    capture_images,
    recognize_and_mark_attendance,
    save_student_data,
    load_student_data,
)

# Page config
st.set_page_config(page_title="Smart Attendance System", layout="centered")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["Add New Student", "Capture Face", "Recognize & Mark Attendance", "View Students", "View Attendance Logs"]
)

student_details_path = "StudentDetails/StudentDetails.csv"
attendance_folder = "Attendance"

os.makedirs("StudentDetails", exist_ok=True)
os.makedirs("Attendance", exist_ok=True)

# --- PAGE: Add New Student ---
if page == "Add New Student":
    st.title("➕ Add New Student")
    student_id = st.text_input("Student ID")
    student_name = st.text_input("Student Name")

    if st.button("Add Student"):
        if student_id and student_name:
            if save_student_data(student_id, student_name):
                st.success(f"Student {student_name} added successfully!")
            else:
                st.warning("Student already exists!")
        else:
            st.error("Please provide both Student ID and Name.")

# --- PAGE: Capture Face ---
elif page == "Capture Face":
    st.title("📸 Capture Face for Enrollment")
    student_id = st.text_input("Student ID")
    student_name = st.text_input("Student Name")

    if st.button("Capture Face"):
        if student_id and student_name:
            success = capture_images(student_id, student_name)
            if success:
                st.success(f"Images captured for {student_name}.")
            else:
                st.error("Error capturing images.")
        else:
            st.error("Please provide both Student ID and Name.")

# --- PAGE: Recognize & Mark Attendance ---
elif page == "Recognize & Mark Attendance":
    st.title("📷 Recognize Faces & Mark Attendance")
    if st.button("Start Recognition"):
        filename = recognize_and_mark_attendance()
        if filename:
            st.success(f"Attendance marked! Saved in {filename}")
            df = pd.read_csv(filename)
            st.dataframe(df)
        else:
            st.warning("No faces recognized or attendance already marked.")

# --- PAGE: View Students ---
elif page == "View Students":
    st.title("👥 Registered Students")
    if os.path.exists(student_details_path):
        df = pd.read_csv(student_details_path)
        st.dataframe(df)
    else:
        st.warning("No student data available.")

# --- PAGE: View Attendance Logs ---
elif page == "View Attendance Logs":
    st.title("📄 Attendance Logs")

    files = sorted(os.listdir(attendance_folder), reverse=True)
    csv_files = [f for f in files if f.endswith(".csv")]

    if csv_files:
        selected_file = st.selectbox("Select an attendance file", csv_files)
        if selected_file:
            df = pd.read_csv(os.path.join(attendance_folder, selected_file))
            st.dataframe(df)
    else:
        st.warning("No attendance files found.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
