import streamlit as st
import numpy as np
import joblib

# ===== Load model & scaler =====
model = joblib.load("SVM_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🤖 Personality Prediction App 🎭")
st.markdown("Enter the information below to get a prediction of your personality type (Introvert or Extrovert).")

# ===== Input fields (sesuai fitur training) =====
time_spent_alone = st.number_input("👉 How much time do you usually spend alone? (0–11) 🏠⏳", min_value=0.0, max_value=11.0, step=1.0)
st.caption("Scale 0–11 → 0: never • 1–3: rarely • 4–7: sometimes • 8–10: often • 11: almost always ")

stage_fear = st.selectbox("👉 Do you feel nervous or afraid when speaking on stage? 🎤😰", ["No", "Yes"])
st.caption("Binary → Yes: feels nervous on stage • No: feels confident/neutral")

social_event_attendance = st.number_input("👉 How often do you attend social events? (0–10) 🎉👥", min_value=0.0, max_value=10.0, step=1.0)
st.caption("Scale 0–10 → 0: never • 1–3: rarely • 4–6: sometimes • 7–9: frequently • 10: very often")

going_outside = st.number_input("👉 How often do you go outside for activities or leisure? (0–7) 🌳🚶", min_value=0.0, max_value=7.0, step=1.0)
st.caption("Scale 0–7 → 0: never • 1–2: rarely • 3–4: sometimes • 5–6: often • 7: almost always")

drained_after_socializing = st.selectbox("👉 Do you feel drained after socializing with others? 😓⚡", ["No", "Yes"])
st.caption("Binary → Yes: feels drained after socializing • No: feels energized or neutral")

friends_circle_size = st.number_input("👉 How big is your circle of friends? (0–15) 👥🤝", min_value=0.0, max_value=15.0, step=1.0)
st.caption("Scale 0–15 → 0: no friends • 1–5: small circle • 6–10: medium circle • 11–15: large circle")

post_frequency = st.number_input("👉 How often do you post on social media? (0–10) 📱💬", min_value=0.0, max_value=10.0, step=1.0)
st.caption("Scale 0–10 → 0: never • 1–3: rarely • 4–6: sometimes • 7–9: frequently • 10: very often")

# Label mapping
LABEL_MAP = {0: "Extrovert", 1: "Introvert"}

# ===== Predict button =====
if st.button("Predict"):
    try:
        # Change categorical to binary
        stage_fear_bin = 1 if stage_fear == "Yes" else 0
        drained_bin = 1 if drained_after_socializing == "Yes" else 0

        # Prepare input data
        input_data = np.array([[
            time_spent_alone,
            stage_fear_bin,
            social_event_attendance,
            going_outside,
            drained_bin,
            friends_circle_size,
            post_frequency
        ]])

        # Scale & predict
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        result = LABEL_MAP.get(int(pred), str(pred))

        st.success(f"The prediction suggests that your personality is likely an {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

