import os
import pandas as pd

from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load training dataset
# -----------------------------

DATA_PATH = os.path.join(
    settings.BASE_DIR,
    "data",
    "complaints.csv"
)

data = pd.read_csv(DATA_PATH)


# -----------------------------
# Train category model
# -----------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(data["complaint"])

category_model = LogisticRegression(
    max_iter=1000
)

category_model.fit(
    X,
    data["category"]
)


# -----------------------------
# Predict complaint category
# -----------------------------

def predict_category(text):
    text_vector = vectorizer.transform([text])
    return category_model.predict(text_vector)[0]


# -----------------------------
# Predict priority
# -----------------------------

def predict_priority(text):
    text = text.lower()

    high_words = [
        "emergency",
        "urgent",
        "danger",
        "immediately",
        "accident",
        "fire",
        "medical",
        "injury",
        "severe",
        "critical",
        "life-threatening",
        "electric shock",
        "gas leak",
        "security threat",
        "harassment"
    ]

    medium_words = [
        "broken",
        "not working",
        "problem",
        "issue",
        "failure",
        "damaged",
        "leaking",
        "delay",
        "late",
        "unavailable",
        "unable to access",
        "missing",
        "repair",
        "overcrowded"
    ]

    # Check for high-priority complaints
    for word in high_words:
        if word in text:
            return "High"

    # Check for medium-priority complaints
    for word in medium_words:
        if word in text:
            return "Medium"

    # Default priority
    return "Low"


# -----------------------------
# Detect duplicate complaints
# -----------------------------

def detect_duplicate(text):

    from .models import Complaint

    complaints = list(
        Complaint.objects.values_list(
            "description",
            flat=True
        )
    )

    if not complaints:
        return False

    all_text = complaints + [text]

    duplicate_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    matrix = duplicate_vectorizer.fit_transform(all_text)

    similarities = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    )[0]

    return similarities.max() >= 0.75
