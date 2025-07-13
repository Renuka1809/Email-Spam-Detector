import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App UI
st.title("📩 Email Spam Classifier")
st.markdown("Enter a message to check whether it's **Spam** or **Ham**.")

# Input box
msg = st.text_area("Your Message:")

# Predict button
if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        data = vectorizer.transform([msg])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("⚠️ This message is **SPAM**.")
        else:
            st.success("✅ This message is **HAM** (Not Spam).")
