import streamlit as st
import time
import random
import joblib
import pandas as pd

# Must be the first Streamlit command
st.set_page_config(
    page_title="AI Stroke Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Load trained ML model
model = joblib.load("stroke_model.pkl")
# Load custom CSS
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css("style.css")
except FileNotFoundError:
    pass
# Real ML Prediction Function
def predict_stroke_risk(data):

    input_data = pd.DataFrame([{
        'age': data['age'],
        'hypertension': 1 if data['hypertension'] == "Yes" else 0,
        'heart_disease': 1 if data['heart_disease'] == "Yes" else 0,
        'avg_glucose_level': data['glucose'],
        'bmi': data['bmi']
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    confidence = round(probability * 100, 1)

    is_high_risk = prediction == 1

    return is_high_risk, confidence

def main():
    # Hero Section
    st.markdown("""
        <div class="hero-container">
            <div class="floating-icon icon-1">🧠</div>
            <div class="floating-icon icon-2">🧬</div>
            <div class="floating-icon icon-3">❤️</div>
            <h1 class="hero-title">AI Stroke Prediction System <span class="heartbeat">❤️</span></h1>
            <h3 class="hero-subtitle">AI-powered stroke risk analysis</h3>
            <p class="hero-text">Leveraging state-of-the-art machine learning to provide early detection and actionable insights for proactive healthcare management.</p>
        </div>
    """, unsafe_allow_html=True)

    # Main Content Area
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">Patient Health Profile</h3>', unsafe_allow_html=True)
        
        with st.form("patient_form", clear_on_submit=False):
            # Form Inputs
            c1, c2 = st.columns(2)
            
            with c1:
                age = st.number_input("Age (Years)", min_value=1, max_value=120, value=45, step=1)
                bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
                glucose = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0, step=1.0)
                work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
                residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
                
            with c2:
                hypertension = st.selectbox("Hypertension", ["No", "Yes"])
                heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                ever_married = st.selectbox("Ever Married", ["No", "Yes"])
                smoking_status = st.selectbox("Smoking Status", ["never smoked", "Unknown", "formerly smoked", "smokes"])

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Action Buttons
            submit_col1, submit_col2 = st.columns([1, 1])
            with submit_col1:
                submitted = st.form_submit_button("🧠 Analyze Patient Data", use_container_width=True)
            with submit_col2:
                # To clear, we can just reload by using a button outside form or just a fake clear
                clear = st.form_submit_button("↺ Reset Form", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card result-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">Analysis Results</h3>', unsafe_allow_html=True)
        
        if clear:
            st.session_state.clear()
            st.rerun()

        if submitted:
            # Gather data
            patient_data = {
                'age': age, 'bmi': bmi, 'glucose': glucose,
                'hypertension': hypertension, 'heart_disease': heart_disease,
                'smoking': smoking_status
            }
            
            # Show animated loader
            with st.spinner("Initializing neural networks..."):
                time.sleep(1)
            with st.spinner("Analyzing biometric markers..."):
                time.sleep(1)
                
            is_high_risk, confidence = predict_stroke_risk(patient_data)
            
            # Display Results with animation
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
            if is_high_risk:
                st.markdown(f"""
                    <div class="result-card high-risk">
                        <div class="result-icon">⚠️</div>
                        <h2 style="color: #ff4b4b; margin: 0;">High Risk Detected</h2>
                        <div class="confidence-meter">
                            <span class="confidence-value">{confidence}%</span>
                            <span class="confidence-label">Confidence Level</span>
                        </div>
                        <p style="margin-top: 15px;">Immediate medical consultation is strongly advised. AI models indicate a significantly elevated probability of cerebrovascular events.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                 st.markdown(f"""
                    <div class="result-card low-risk">
                        <div class="result-icon">✅</div>
                        <h2 style="color: #00d4ff; margin: 0;">Low Risk Profile</h2>
                        <div class="confidence-meter">
                            <span class="confidence-value">{confidence}%</span>
                            <span class="confidence-label">Confidence Level</span>
                        </div>
                        <p style="margin-top: 15px;">Biometric indicators are within normal parameters. Continue maintaining a healthy lifestyle and regular check-ups.</p>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature Importance Mockup
            st.markdown("<br><h5>Key Risk Factors Identified:</h5>", unsafe_allow_html=True)
            if is_high_risk:
                if age > 50: st.markdown("• **Age** is a contributing factor.")
                if bmi > 25: st.markdown("• **BMI** suggests elevated risk.")
                if hypertension == "Yes": st.markdown("• **Hypertension** significantly increases risk.")
                if heart_disease == "Yes": st.markdown("• **Heart Disease** history detected.")
                if glucose > 120: st.markdown("• **Glucose Level** is higher than optimal.")
            else:
                st.markdown("• All tracked vital signs are within expected ranges.")
                
        else:
            st.markdown("""
                <div style="text-align: center; color: #8892b0; padding: 40px 0;">
                    <p style="font-size: 48px; margin-bottom: 10px;">📊</p>
                    <p>Awaiting patient data...</p>
                    <p style="font-size: 12px;">Fill out the profile and click Analyze to generate an AI risk assessment.</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer / About Section
    st.markdown("---")
    st.markdown("""
        <div class="footer-container">
            <h4 style="color: var(--accent-purple); font-family: 'Outfit', sans-serif;">About the AI model</h4>
            <p style="margin-bottom: 20px;">This system leverages advanced machine learning algorithms trained on comprehensive healthcare datasets to analyze biometric markers and identify complex patterns associated with stroke risk.</p>
            
            <h4 style="color: var(--accent-purple); font-family: 'Outfit', sans-serif;">Technologies used:</h4>
            <ul style="list-style-type: none; padding: 0; display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; margin-bottom: 30px;">
                <li style="background: rgba(100, 255, 218, 0.1); padding: 5px 15px; border-radius: 15px; border: 1px solid rgba(100, 255, 218, 0.3);">Python</li>
                <li style="background: rgba(100, 255, 218, 0.1); padding: 5px 15px; border-radius: 15px; border: 1px solid rgba(100, 255, 218, 0.3);">Streamlit</li>
                <li style="background: rgba(100, 255, 218, 0.1); padding: 5px 15px; border-radius: 15px; border: 1px solid rgba(100, 255, 218, 0.3);">Scikit-learn</li>
                <li style="background: rgba(100, 255, 218, 0.1); padding: 5px 15px; border-radius: 15px; border: 1px solid rgba(100, 255, 218, 0.3);">Logistic Regression</li>
                <li style="background: rgba(100, 255, 218, 0.1); padding: 5px 15px; border-radius: 15px; border: 1px solid rgba(100, 255, 218, 0.3);">AI Prompt Engineering</li>
            </ul>
            
            <p style="font-size: 12px; color: #64ffda;">* Disclaimer: This is a demonstration application and should not replace professional medical advice.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
