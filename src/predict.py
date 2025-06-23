import joblib
import os
import string
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
def load_model(model_path="models/best_model.pkl", vectorizer_path="models/vectorizer.pkl"):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or Vectorizer files not found. Train the model first.")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
def predict_message(message):
    message_clean = clean_text(message)
    model, vectorizer = load_model()
    message_vec = vectorizer.transform([message_clean])
    prediction = model.predict(message_vec)
    label = "Spam" if prediction[0] == 1 else "Ham"
    return label
if __name__ == "__main__":
    test_message = input("Enter a message: ")
    result = predict_message(test_message)
    print("Prediction: ", result)