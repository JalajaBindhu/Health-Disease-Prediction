import streamlit as st
import pandas as pd
import pickle

# ---------------- Load Models ----------------
model_diabetes = pickle.load(open("models/diabetes_model.pkl", "rb"))
model_heart = pickle.load(open("models/heart_model.pkl", "rb"))

diabetes_columns = pickle.load(open("models/diabetes_columns.pkl", "rb"))
heart_columns = pickle.load(open("models/heart_columns.pkl", "rb"))

# ---------------- App Title ----------------
st.title("🩺 Health Prediction System")
st.subheader("Diabetes & Heart Disease Prediction")

# ---------------- Diabetes Input ----------------
st.header("🧪 Diabetes Details")

preg = st.number_input("Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose Level", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age_d = st.number_input("Age (Diabetes)", 1, 120, 30)

diabetes_input = [preg, glucose, bp, skin, insulin, bmi, dpf, age_d]

# ---------------- Heart Input ----------------
st.header("❤️ Heart Disease Details")

age = st.number_input("Age (Heart)", 1, 120, 50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 0, 300, 120)
chol = st.number_input("Cholesterol", 0, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
restecg = st.selectbox("Rest ECG", [0,1,2])
thalach = st.number_input("Max Heart Rate", 0, 300, 150)
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("CA", [0,1,2,3,4])
thal = st.selectbox("Thal", [0,1,2,3])

heart_input = [age, sex, cp, trestbps, chol, fbs, restecg,
               thalach, exang, oldpeak, slope, ca, thal]

# ---------------- Prediction Button ----------------
if st.button("🔍 Predict Disease"):

    # ---- Diabetes Prediction ----
    df_diabetes = pd.DataFrame([diabetes_input], columns=diabetes_columns)
    diabetes_result = model_diabetes.predict(df_diabetes)[0]

    # ---- Heart Prediction ----
    heart_dict = dict(zip(
        ['age','sex','cp','trestbps','chol','fbs','restecg',
         'thalach','exang','oldpeak','slope','ca','thal'],
        heart_input
    ))

    df_heart = pd.DataFrame([heart_dict])
    df_heart = pd.get_dummies(df_heart, drop_first=True)
    df_heart = df_heart.reindex(columns=heart_columns, fill_value=0)

    heart_result = model_heart.predict(df_heart)[0]

    # ---------------- Results ----------------
    st.subheader("📊 Prediction Results")

    if diabetes_result == 1:
        st.error("⚠️ Diabetes: Positive")
    else:
        st.success("✅ Diabetes: Negative")

    if heart_result == 1:
        st.error("⚠️ Heart Disease: Positive")
    else:
        st.success("✅ Heart Disease: Negative")
