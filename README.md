# Student Performance Prediction App

## Project Overview
This project is designed to predict a student's academic performance based on study habits, attendance, parental education, extracurricular activities, and other related features. The prediction is categorized into three performance levels: **Low**, **Medium**, and **High**. The application uses machine learning models (Random Forest and Gaussian Naive Bayes) to provide both a performance classification and an estimated final grade. A pie chart visualization displays the model's prediction probabilities for all performance categories.

**Dataset link** : https://www.kaggle.com/datasets/prekshad2166/student-study-habits

**Streamlit Link** : https://kanika-07cs-task15-14-10-25--app-bzxr8v.streamlit.app/
## Dataset Description
The dataset (`student_study_habits.csv`) contains records of students with the following features:

| Feature                          | Description                                    |
|----------------------------------|------------------------------------------------|
| `study_hours_per_week`            | Number of hours a student studies per week    |
| `sleep_hours_per_day`             | Average sleep hours per day                    |
| `attendance_percentage`           | Attendance percentage in classes              |
| `assignments_completed`           | Number of assignments completed               |
| `participation_level_Low`         | Participation in class (Low level)            |
| `participation_level_Medium`      | Participation in class (Medium level)         |
| `internet_access_Yes`             | Availability of internet at home (Yes/No)    |
| `parental_education_High School` | Parental education: High School               |
| `parental_education_Master's`    | Parental education: Master's                   |
| `parental_education_PhD`         | Parental education: PhD                        |
| `extracurricular_Yes`             | Participation in extracurricular activities   |
| `part_time_job_Yes`               | Has a part-time job                             |
| `final_grade`                     | Numeric grade obtained by the student         |

---

## Data Preprocessing Steps
1. **Handling Outliers**: Outliers in numerical features such as `study_hours_per_week` and `attendance_percentage` were visualized using boxplots. 
2. **Skewness Visualization**: Distributions of numeric columns were analyzed using histograms with KDE plots.  
3. **Target Variable Creation**: The `final_grade` column was converted into categorical `Performance` using:
   ```python
   df['Performance'] = pd.cut(df['final_grade'], bins=[0, 59, 79, 100], labels=['Low', 'Medium', 'High'])
4.**Label Encoding**: Performance labels were encoded into numeric values using LabelEncoder.

## Model Development
Two machine learning models were trained:

1. Random Forest Classifier
- Parameters: n_estimators=1000, class_weight='balanced', random_state=42
- Trained on all features.
2. Gaussian Naive Bayes
Trained on numeric features: study_hours_per_week, sleep_hours_per_day, attendance_percentage, assignments_completed.

## Results & Insights
- Random Forest outperforms GaussianNB due to handling all feature types and interactions.
- Attendance, study hours, and assignment completion are the most influential features.

## App screenshot
<img width="580" height="580" alt="image" src="https://github.com/user-attachments/assets/b0db08e0-8cf3-4840-8a50-7777ff2bb6c9" />
<img width="580" height="580" alt="image" src="https://github.com/user-attachments/assets/67eadabe-1025-4a10-baf4-ba4a634e826d" />


## How to Run
1. Clone the repository
- git clone <repository_url>
- cd <repository_folder>
2. Run the Streamlit app
- streamlit run app.py

## Conclusion
This project demonstrates the application of machine learning models for predicting student performance. The combination of Random Forest and Gaussian Naive Bayes provides accurate predictions and interpretability.
