import streamlit as st
import joblib
import re
from PIL import Image

# Load models and vectorizer
nb_model = joblib.load("naive_bayes_model.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Set page config
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# Custom CSS for styling
# Custom CSS for styling (brightened version)
# Bright CSS for title and subtitle
st.markdown(
    """
    <style>
        .main {
            background-color: #f0f2f6;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #1a1a1a;
            font-weight: bold;
        }
        .title {
            color: white !important;
            font-size: 40px;
            font-weight: bold;
        }
        .subtitle {
            color:  white !important;
            font-size: 22px;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            font-size: 18px;
            background-color: #6c63ff;
            color: white;
            border: none;
        }
        .stTextArea>div>textarea {
            border-radius: 10px;
            font-size: 16px;
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #cccccc;
        }
        .not-spam {
            color:  white !important;
            font-weight: bold;
            font-size: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



# Initialize session state for model selection
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Function to set selected model
def set_model(model_name):
    st.session_state.selected_model = model_name

# Header section
st.markdown("<h1 class='title'>📧 Email Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'>Select a model and enter text to detect spam!</h3>", unsafe_allow_html=True)

# Model selection buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🤖 Naïve Bayes"):
        set_model("nb")
with col2:
    if st.button("⚡ Logistic Regression"):
        set_model("lr")

# User input
user_input = st.text_area("✍️ Enter your email text:", height=150)

# Predict function
def predict(text):
    if st.session_state.selected_model is None:
        st.warning("⚠️ Please select a model first.")
        return None

    text_tfidf = vectorizer.transform([text])

    if st.session_state.selected_model == "nb":
        prediction = nb_model.predict(text_tfidf)[0]
    elif st.session_state.selected_model == "lr":
        prediction = lr_model.predict(text_tfidf)[0]

    return "🚀 Spam" if prediction == 1 else "✅ <span class='not-spam'>Not Spam</span>"

# Predict button
if st.button("🔍 Predict"):
    result = predict(user_input)
    if result:
        st.markdown(f"<div class='not-spam'>{result}</div>", unsafe_allow_html=True)