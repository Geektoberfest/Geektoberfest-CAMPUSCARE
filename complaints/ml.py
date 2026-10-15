import os
import re
import unicodedata
import pandas as pd

from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Multilingual configuration & Lexicons
# -----------------------------

TOKEN_PATTERN = r"[ऀ-ॿa-zA-Z0-9_]+"

MARATHI_INDICATORS = {
    "नाही", "आहे", "मध्ये", "येत", "वसतिगृह", "दुरुस्ती", "बाक", "वेळापत्रक",
    "महाविद्यालय", "उशिरा", "चालक", "थांब्या", "मुलांचे", "मुलींचे", "दिवा",
    "नळ", "स्वच्छतागृह", "हजेरी", "गुण", "प्रयोगशाळा", "झाली", "पडले", "चालू",
    "करा", "इमारत", "वर्गातील", "तक्रार", "वाहतूक", "अडचण", "होतात", "केली"
}

HINGLISH_INDICATORS = {
    "mein", "me", "nahi", "nahin", "hai", "hain", "raha", "rahi", "rahe",
    "kaam", "paani", "dikkat", "toota", "tooti", "kharab", "khatra", "jaldi",
    "bohot", "bahut", "aaj", "ka", "ki", "ke", "ko", "mere", "meri", "mera",
    "chahiye", "band", "karo", "chal", "kuch", "kare", "hoga", "hogi", "tha",
    "thi", "the", "bhi", "shuru", "aur", "par", "bhai", "yaar", "ab", "kab"
}

HIGH_PRIORITY_WORDS = [
    # English
    "emergency", "urgent", "danger", "fire", "accident", "unsafe",
    "medical", "security", "immediately", "power outage", "no water",
    "water shortage", "electric shock", "loose wire", "live wire",
    "spark", "sparking", "short circuit", "gas leak", "chemical spill",
    "ceiling plaster fell", "ceiling collapse", "falling debris",
    "vomiting", "food poisoning", "toxic", "fumes",
    "theft", "break in", "unauthorized", "trespass", "recklessly",
    "entire lab offline", "dead", "crashed", "overcrowded",
    "flat tyre", "broke down", "hall ticket", "deadline",
    "sharp nails", "jammed shut", "blackout", "wrong grade",
    "not functioning",
    # Hindi
    "आपातकालीन", "तुरंत", "खतरा", "आग", "दुर्घटना", "असुरक्षित",
    "मेडिकल", "सुरक्षा", "तत्काल", "बिजली गुल", "पानी नहीं", "पानी बंद",
    # Hinglish
    "khatra", "aag", "asurakshit", "turant", "jaldi", "tatkaal",
    "bijli gul", "paani nahi", "power cut", "light nahi",
    # Marathi
    "आणीबाणी", "तातडीने", "धोका", "अपघात", "वैद्यकीय", "लगेच",
    "वीज गेली", "पाणी बंद"
]

MEDIUM_PRIORITY_WORDS = [
    # English
    "not working", "broken", "slow", "problem", "issue", "damaged", "missing",
    "malfunctioning", "fused", "defective", "glitch", "error", "out of order",
    "choked", "clogged", "overflowing", "leaking", "leak", "dripping",
    "spoil", "spoiled", "delayed", "delay", "cancelled", "late", "drain",
    "drainage", "stuck", "dropping connections", "disconnecting", "not cooling",
    "infestation", "discrepancy", "clash", "server error", "failing",
    "cannot connect", "unable to connect", "not heating", "expired", "bad gateway",
    "cracked", "unhygienic", "shortage", "down", "not sending", "not uploaded",
    "skips", "skipping", "less than",
    # Hindi
    "काम नहीं कर रहा", "काम नहीं", "खराब", "टूटा", "टूटी", "धीमा",
    "धीमी", "समस्या", "दिक्कत", "गायब",
    # Hinglish
    "kaam nahi kar raha", "kaam nahi", "work nahi kar raha", "kharab",
    "dikkat", "toota", "tooti", "chal nahi raha", "gayab",
    # Marathi
    "चालत नाही", "काम करत नाही", "नादुरुस्त", "बंद पडले", "तुटलेले",
    "तुटला", "हळू", "अडचण", "दिसत नाही"
]


# -----------------------------
# Preprocessing & Normalization
# -----------------------------

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
    """Normalize unicode, expand contractions, and strip surrounding whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text).strip())
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# -----------------------------
# Language detection
# -----------------------------

def detect_language(text):
    """
    Detects whether the complaint is in English, Hindi, Hinglish, or Marathi.
    - If Devanagari characters are present: distinguishes Marathi vs Hindi based on lexical indicators.
    - If Latin characters: checks for common Hinglish markers; otherwise defaults to English.
    """
    if not text:
        return "English"

    text = preprocess_text(text)

    # Devanagari script detection (ऀ-ॿ)
    if re.search(r"[ऀ-ॿ]", text):
        devanagari_words = set(re.findall(r"[ऀ-ॿ]+", text))
        if devanagari_words & MARATHI_INDICATORS:
            return "Marathi"
        return "Hindi"

    # Latin script: Hinglish vs English detection
    tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if tokens & HINGLISH_INDICATORS:
        return "Hinglish"

    return "English"


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
    ngram_range=(1, 2),
    token_pattern=TOKEN_PATTERN,
    sublinear_tf=True,
    max_df=0.9
)

X = vectorizer.fit_transform(data["complaint"].map(preprocess_text))

category_model = LogisticRegression(
    C=2.0,
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs"
)

category_model.fit(
    X,
    data["category"]
)


# -----------------------------
# Predict complaint category
# -----------------------------

def predict_category(text):
    """Predict category label (backward-compatible signature)."""
    cleaned = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned])
    return category_model.predict(text_vector)[0]


def predict_category_with_confidence(text):
    """Predict category label along with calibrated probability confidence score."""
    cleaned = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned])
    probs = category_model.predict_proba(text_vector)[0]
    best_idx = probs.argmax()
    category = category_model.classes_[best_idx]
    confidence = float(probs[best_idx])
    return category, confidence


# -----------------------------
# Predict priority
# -----------------------------

NEGATION_PATTERNS = [
    r"\bnot\s+an?\s+emergency\b",
    r"\bno\s+emergency\b",
    r"\bnot\s+urgent\b",
    r"\bnot\s+a\s+danger\b",
    r"\bnot\s+dangerous\b",
    r"\bnot\s+critical\b",
]

def predict_priority(text):
    """
    Assign priority using multi-tiered rule engine with negation awareness:
    - High: Life safety hazards, acute outages, security breaches, urgent deadlines.
    - Medium: Operational defects, equipment malfunctions, scheduling discrepancies.
    - Low: General inquiries, administrative feedback, and negated urgency queries.
    """
    text_lower = preprocess_text(text).lower()

    is_negated = any(re.search(pat, text_lower) for pat in NEGATION_PATTERNS)

    if not is_negated and any(word in text_lower for word in HIGH_PRIORITY_WORDS):
        return "High"

    if any(word in text_lower for word in MEDIUM_PRIORITY_WORDS):
        return "Medium"

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

    cleaned_text = preprocess_text(text)
    all_text = [preprocess_text(c) for c in complaints] + [cleaned_text]

    duplicate_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    matrix = duplicate_vectorizer.fit_transform(all_text)

    similarities = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    )[0]

    return similarities.max() >= 0.75
