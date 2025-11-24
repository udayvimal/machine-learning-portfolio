import streamlit as st
import numpy as np
import pickle
import os

# ------------------ Custom Dark Theme CSS ------------------
st.markdown("""
<style>
/* Dark background gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #eee;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    min-height: 100vh;
}

/* Sidebar dark style */
[data-testid="stSidebar"] {
    background: #121212;
    color: #bbb;
    font-weight: 500;
}

/* Title gradient text */
h1 {
    background: linear-gradient(45deg, #6a11cb, #2575fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Button style */
.stButton > button {
    background: linear-gradient(45deg, #6a11cb, #2575fc);
    color: #fff;
    font-weight: 700;
    border-radius: 15px;
    height: 50px;
    width: 100%;
    font-size: 20px;
    transition: background 0.3s ease;
    box-shadow: 0 0 15px #6a11cbaa;
    margin-top: 15px;
}
.stButton > button:hover {
    background: linear-gradient(45deg, #2575fc, #6a11cb);
    box-shadow: 0 0 25px #2575fccc;
    cursor: pointer;
}

/* Input fields styling */
.stTextInput>div>div>input, .stNumberInput>div>input {
    background-color: #222;
    color: white;
    border: 2px solid #444;
    border-radius: 10px;
    padding: 12px;
    font-size: 18px;
    transition: border-color 0.3s ease;
}
.stTextInput>div>div>input:focus, .stNumberInput>div>input:focus {
    border-color: #2575fc;
    outline: none;
}

/* Prediction output box */
.prediction-box {
    margin-top: 25px;
    padding: 20px;
    border-radius: 15px;
    background: rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 25px rgba(37, 117, 252, 0.5);
    font-size: 22px;
    font-weight: 700;
    text-align: center;
    color: #fff;
}

/* Success style */
.success {
    color: #4CAF50;
    background: rgba(76, 175, 80, 0.15);
}

/* Error style */
.error {
    color: #f44336;
    background: rgba(244, 67, 54, 0.15);
}

/* Sidebar header */
.sidebar .css-1d391kg h2 {
    color: #6a11cb !important;
    font-weight: 800;
}

/* Footer */
footer {
    text-align: center;
    color: #bbb;
    font-size: 14px;
    margin-top: 50px;
}
footer a {
    color: #6a11cb;
    text-decoration: none;
}
footer a:hover {
    text-decoration: underline;
}

/* Responsive tweaks */
@media (max-width: 600px) {
    .stButton > button {
        font-size: 16px;
        height: 45px;
    }
}
</style>
""", unsafe_allow_html=True)

# --------- Page config ---------
st.set_page_config(
    page_title="🏨 Hotel Booking Prediction - Dark Mode",
    page_icon="🏩",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -------- Sidebar ---------
with st.sidebar:
    st.header("🏨 About Hotel Predictor")
    st.write(
        """
        This app predicts whether a hotel booking will be **confirmed** or **canceled** based on key booking details.

        - Powered by a Logistic Regression model  
        - Scaled inputs for accurate prediction  
        - Dark theme for better viewing comfort  
        - Developed by [Uday Vimal](https://github.com/udayvimal)  
        """
    )
    st.markdown("---")
    with st.expander("📁 Show project files"):
        st.write(os.listdir("."))

# --------- Load model & scaler ---------
@st.cache_resource
def load_model_and_scaler():
    with open('logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# --------- Title ---------
st.markdown("<h1>🏩 Hotel Booking Prediction System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#ccc; font-size:18px;'>Fill the booking details below and get instant prediction whether the reservation will be confirmed or canceled.</p>",
    unsafe_allow_html=True,
)

# --------- Input Form ---------
with st.form("prediction_form"):
    lead_time = st.number_input(
        "Lead Time (days before arrival)",
        min_value=0,
        max_value=365,
        value=30,
        help="Number of days between booking and arrival date",
        step=1,
    )
    adults = st.number_input(
        "Number of Adults",
        min_value=1,
        max_value=10,
        value=2,
        help="How many adults are included in the booking",
        step=1,
    )
    previous_cancellations = st.number_input(
        "Previous Cancellations",
        min_value=0,
        max_value=10,
        value=0,
        help="How many bookings were canceled before by this customer",
        step=1,
    )
    special_requests = st.number_input(
        "Total Special Requests",
        min_value=0,
        max_value=10,
        value=1,
        help="Number of special requests made by customer (e.g. late check-in)",
        step=1,
    )
    submitted = st.form_submit_button("Predict Booking Status")

# --------- Prediction Logic ---------
if submitted:
    try:
        features = np.array([[lead_time, adults, previous_cancellations, special_requests]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        # Display prediction with style
        if prediction == 1:
            st.markdown(
                "<div class='prediction-box success'>✅ Booking Confirmed! Your reservation is very likely to succeed.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='prediction-box error'>❌ Booking Likely Canceled. Please reconsider your booking details.</div>",
                unsafe_allow_html=True,
            )

        # Show prediction confidence if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)[0][prediction]
            st.info(f"Prediction Confidence: **{proba * 100:.2f}%**")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# --------- Footer ---------
st.markdown("---")
st.markdown(
    """
    <footer>
    Developed by <a href='https://github.com/udayvimal' target='_blank'>Uday Vimal</a> &bull; Powered by Streamlit
    </footer>
    """,
    unsafe_allow_html=True,
)
