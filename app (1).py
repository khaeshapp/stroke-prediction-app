import streamlit as st
import time
import joblib
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Stroke Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("stroke_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# ---------------- LOAD CSS ----------------
def local_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    local_css("style.css")
except FileNotFoundError:
    pass

# FIX: ensure labels are visible even if CSS breaks
st.markdown("""
<style>
label {
    color: #000000 !important;
    font-weight: 600 !important;
}

.stSelectbox label, .stNumberInput label {
    color: #000000 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PREDICTION FUNCTION ----------------
def predict_stroke_risk(data):
    if model is None:
        return False, 0

    input_data = pd.DataFrame([{
        'age': data['age'],
        'hypertension': 1 if data['hypertension'] == "Yes" else 0,
        'heart_disease': 1 if data['heart_disease'] == "Yes" else 0,
        'avg_glucose_level': data['glucose'],
        'bmi': data['bmi']
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return prediction == 1, round(probability * 100, 1)

# ---------------- UI ----------------
def main():

    # HEADER
    st.title("🧠 AI Stroke Prediction System")
    st.write("AI-powered health risk analysis for early stroke detection")

    col1, col2 = st.columns([2, 1], gap="large")

    # ---------------- INPUT SECTION ----------------
    with col1:
        st.subheader("Patient Health Profile")

        with st.form("patient_form"):
            age = st.number_input("Age", 1, 120, 45)
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
            glucose = st.number_input("Glucose Level", 50.0, 300.0, 100.0)

            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

            submitted = st.form_submit_button("Analyze Risk")

    # ---------------- RESULT SECTION ----------------
    with col2:
        st.subheader("Analysis Result")

        if submitted:

            patient_data = {
                "age": age,
                "bmi": bmi,
                "glucose": glucose,
                "hypertension": hypertension,
                "heart_disease": heart_disease
            }

            with st.spinner("Analyzing data..."):
                time.sleep(1.5)

            risk, confidence = predict_stroke_risk(patient_data)

            if risk:
                st.error(f"⚠️ High Stroke Risk\nConfidence: {confidence}%")
                st.write("Consult a medical professional immediately.")
            else:
                st.success(f"✅ Low Stroke Risk\nConfidence: {confidence}%")
                st.write("Maintain a healthy lifestyle.")

        else:
            st.info("Enter patient data and click Analyze Risk")

    # ---------------- FOOTER (FIXED) ----------------
    st.markdown("---")

    st.subheader("About the AI Model")
    st.write(
        "This machine learning model analyzes medical parameters "
        "to estimate stroke risk. It is for educational purposes only."
    )

    st.subheader("Technologies Used")
    st.write("Python • Streamlit • Scikit-learn • Machine Learning")

    st.caption("⚠️ Disclaimer: This is not a medical diagnosis tool.")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
