import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Health Disease Prediction", page_icon="🏥", layout="centered")

st.title("🏥 Health Disease Prediction System")
st.write("Predict **Heart Disease** and **Diabetes** using Machine Learning API")

# -------------------------
# Disease Selection
# -------------------------

disease = st.selectbox("Select Disease", ["Heart Disease", "Diabetes"])

st.divider()

# -------------------------
# Heart Disease Form
# -------------------------

if disease == "Heart Disease":
    st.subheader("❤️ Heart Disease Input")

    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.number_input("Chest Pain Type (cp)", 0, 3, 0)
    trestbps = st.number_input("Resting Blood Pressure", 50, 250, 130)
    chol = st.number_input("Cholesterol", 50, 600, 250)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.number_input("Rest ECG", 0, 2, 1)
    thalach = st.number_input("Max Heart Rate", 50, 250, 160)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.2)
    slope = st.number_input("Slope", 0, 2, 2)
    ca = st.number_input("CA", 0, 4, 0)
    thal = st.number_input("Thal", 0, 3, 2)

    if st.button("Predict Heart Disease"):
        payload = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }

        try:
            response = requests.post(f"{API_URL}/predict/heart", json=payload)
            result = response.json()

            if result.get("prediction") == 1:
                st.error("⚠️ Heart Disease Detected")
            else:
                st.success("✅ No Heart Disease Detected")

        except Exception as e:
            st.error(f"API Error: {e}")

# -------------------------
# Diabetes Form
# -------------------------

else:
    st.subheader("🩸 Diabetes Input")

    pregnancies = st.number_input("Pregnancies", 0, 20, 2)
    glucose = st.number_input("Glucose", 50, 300, 120)
    bloodpressure = st.number_input("Blood Pressure", 30, 200, 70)
    skinthickness = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 85)
    bmi = st.number_input("BMI", 10.0, 80.0, 26.5)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.45)
    age = st.number_input("Age", 1, 120, 35)

    if st.button("Predict Diabetes"):
        payload = {
            "pregnancies": pregnancies,
            "glucose": glucose,
            "bloodpressure": bloodpressure,
            "skinthickness": skinthickness,
            "insulin": insulin,
            "bmi": bmi,
            "dpf": dpf,
            "age": age
        }

        try:
            response = requests.post(f"{API_URL}/predict/diabetes", json=payload)
            result = response.json()

            if result.get("prediction") == 1:
                st.error("⚠️ Diabetes Detected")
            else:
                st.success("✅ No Diabetes Detected")

        except Exception as e:
            st.error(f"API Error: {e}")

st.divider()
st.caption("ML API powered by FastAPI • Frontend built with Streamlit")
