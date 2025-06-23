# src/preprocess.py
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path):
    """Load dataset from CSV and rename relevant columns."""
    df = pd.read_csv(path, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Binary labels
    return df

def clean_text(text):
    """Basic text cleaning: lowercasing, remove punctuation, numbers, etc."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
    return text

def preprocess_data(df):
    """Apply text cleaning to the message column."""
    df['message'] = df['message'].apply(clean_text)
    return df

def vectorize_text(train_texts, test_texts):
    """Convert texts to TF-IDF vectors."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

def prepare_data(path, test_size=0.2, random_state=42):
    """Full data prep pipeline."""
    df = load_data(path)
    df = preprocess_data(df)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=test_size, random_state=random_state
    )
    X_train, X_test, vectorizer = vectorize_text(X_train_text, X_test_text)
    return X_train, X_test, y_train, y_test, vectorizer
