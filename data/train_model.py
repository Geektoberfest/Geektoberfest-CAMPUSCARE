"""
Offline training script for CampusCare category classification model.
Trains TF-IDF vectorizer and Logistic Regression classifier and serializes them to joblib.
"""
import os
import re
import unicodedata
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "complaints.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "category_model.joblib")

TOKEN_PATTERN = r"[ऀ-ॿa-zA-Z0-9_]+"

CONTRACTIONS = {
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bdoesn't\b": "does not",
    r"\bdon't\b": "do not",
    r"\bdidn't\b": "did not"
}


def preprocess_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text).strip())
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def train_and_save(data_path=DATA_PATH, model_path=MODEL_PATH):
    print(f"Loading training data from {data_path}...")
    data = pd.read_csv(data_path)

    print(f"Training TF-IDF vectorizer on {len(data)} complaints...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=TOKEN_PATTERN,
        sublinear_tf=True,
        max_df=0.9
    )
    X = vectorizer.fit_transform(data["complaint"].map(preprocess_text))

    print("Fitting Logistic Regression category model...")
    category_model = LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )
    category_model.fit(X, data["category"])

    payload = {
        "vectorizer": vectorizer,
        "category_model": category_model
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(payload, model_path)
    print(f"Model saved successfully to {model_path} ({os.path.getsize(model_path)} bytes)")
    return vectorizer, category_model


if __name__ == "__main__":
    train_and_save()
