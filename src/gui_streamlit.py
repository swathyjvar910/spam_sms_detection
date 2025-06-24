import streamlit as st
from predict import predict_message

st.set_page_config(page_title="SMS Spam Classifier", layout="centered")

st.title("📩 SMS Spam Classifier")

msg = st.text_area("Enter your SMS message:", height=150)

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = predict_message(msg)
        if prediction == "Spam":
            st.error("🚨 This message is SPAM.")
        else:
            st.success("✅ This message is HAM (not spam).")
