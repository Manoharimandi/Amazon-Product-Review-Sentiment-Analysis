import streamlit as st
import joblib
import re
import nltk
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# -------------------------------
# Download NLTK Resources
# -------------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Amazon Product Review Sentiment Analysis",
    page_icon="🛒",
    layout="centered"
)

# -------------------------------
# Load Model and Vectorizer
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# -------------------------------
# Initialize NLP
# -------------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# -------------------------------
# Text Preprocessing Function
# -------------------------------
def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    words = word_tokenize(text)

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -------------------------------
# Prediction Function
# -------------------------------
def predict_sentiment(review):

    cleaned_review = preprocess_text(review)

    vector = vectorizer.transform([cleaned_review])

    prediction = model.predict(vector)[0]

    confidence = model.predict_proba(vector).max()

    return prediction, confidence

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🛒 Amazon Product Review Sentiment Analysis")

st.write(
    "Enter an Amazon product review below to predict whether the sentiment is Positive or Negative."
)

review = st.text_area(
    "Enter Review",
    height=200,
    placeholder="Example: This product exceeded my expectations. The quality is amazing!"
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        prediction, confidence = predict_sentiment(review)

        st.subheader("Prediction")

        if prediction == "pos":
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.subheader("Confidence")

        st.progress(float(confidence))

        st.write(f"**{confidence:.2%}**")

st.markdown("---")

st.caption(
    "Model: Logistic Regression | Feature Extraction: TF-IDF | Dataset: Amazon Product Reviews"
)