import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('student_models.pkl', 'rb') as f:
    models = pickle.load(f)

rf = models['RandomForest']
nb = models['GaussianNB']
le_target = models['LabelEncoder']

df = pd.read_csv("student_study_habits.csv")
df['Performance'] = pd.cut(df['final_grade'], bins=[0, 55, 100], labels=['Fail','Pass'])

st.title("Student Performance Prediction App")
st.write("Predict student performance using Random Forest or Gaussian Naive Bayes.")

st.header("Enter Student Details")
with st.form("student_form"):
    study_hours = st.number_input("Study hours per week", max_value=100.0, value=5.0, step=0.1)
    sleep_hours = st.number_input("Sleep hours per day", max_value=24.0, value=8.0, step=0.1)
    attendance = st.number_input("Attendance percentage", max_value=100.0, value=90.0, step=0.1)
    assignments = st.number_input("Assignments completed", max_value=50.0, value=10.0, step=0.1)
    participation_low = st.selectbox("Participation level Low", [0,1])
    participation_med = st.selectbox("Participation level Medium", [0,1])
    internet = st.selectbox("Internet access Yes", [0,1])
    edu_highschool = st.selectbox("Parental Education High School", [0,1])
    edu_masters = st.selectbox("Parental Education Master's", [0,1])
    edu_phd = st.selectbox("Parental Education PhD", [0,1])
    extracurricular = st.selectbox("Extracurricular Yes", [0,1])
    part_time = st.selectbox("Part-time job Yes", [0,1])
    model_choice = st.selectbox("Choose Model", ["Random Forest", "GaussianNB"])
    submit = st.form_submit_button("Predict Performance")

if submit:
    new_student = pd.DataFrame({
        'study_hours_per_week':[study_hours],
        'sleep_hours_per_day':[sleep_hours],
        'attendance_percentage':[attendance],
        'assignments_completed':[assignments],
        'participation_level_Low':[participation_low],
        'participation_level_Medium':[participation_med],
        'internet_access_Yes':[internet],
        'parental_education_High School':[edu_highschool],
        'parental_education_Master\'s':[edu_masters],
        'parental_education_PhD':[edu_phd],
        'extracurricular_Yes':[extracurricular],
        'part_time_job_Yes':[part_time]
    })
    
    if model_choice == "Random Forest":
        pred_class = rf.predict(new_student)
        pred_probs = rf.predict_proba(new_student)[0]
    else:
        num_features = ['study_hours_per_week','sleep_hours_per_day','attendance_percentage','assignments_completed']
        pred_class = nb.predict(new_student[num_features])
        pred_probs = nb.predict_proba(new_student[num_features])[0]

    pred_performance = le_target.inverse_transform(pred_class)[0]
    
    if pred_performance == "Fail":
        st.error(f"Predicted Performance: {pred_performance}\n\n"
                 "The student is likely to underperform. Consider improving study habits, attendance, and engagement.")
    else:
        st.success(f"Predicted Performance: {pred_performance}\n\n"
                   "Excellent! The student is performing well and maintaining good academic habits.")
