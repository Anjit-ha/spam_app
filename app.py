import streamlit as st
import joblib
import re

# Load models and vectorizer
nb_model = joblib.load("naive_bayes_model.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize session state for model selection
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Function to set selected model
def set_model(model_name):
    st.session_state.selected_model = model_name

# Streamlit UI
st.title("📧 Email Spam Detection")

col1, col2 = st.columns(2)
with col1:
    if st.button("NB (Naïve Bayes)"):
        set_model("nb")
with col2:
    if st.button("LR (Logistic Regression)"):
        set_model("lr")

# Text input
user_input = st.text_input("Enter text:")

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

    return "🚀 Spam" if prediction == 1 else "✅ Not Spam"

# Predict button
if st.button("Predict"):
    result = predict(user_input)
    if result:
        st.success(result)
