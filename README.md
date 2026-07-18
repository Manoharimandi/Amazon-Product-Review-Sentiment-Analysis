# Amazon Product Review Sentiment Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://amazon-appuct-review-sentiment-analysis-2zgaqxewytavzodykqgfg9.streamlit.app)

## 🚀 Live Demo

**Try the application here:**  
https://amazon-appuct-review-sentiment-analysis-2zgaqxewytavzodykqgfg9.streamlit.app

## Overview

This project predicts whether an Amazon product review is **Positive** or **Negative** using Natural Language Processing (NLP) and Machine Learning.

The application preprocesses user input, converts text into TF-IDF features, and uses a trained Logistic Regression model to classify the sentiment. A Streamlit web application provides an interactive interface for real-time predictions.

---

## Features

- Amazon review sentiment classification
- Text preprocessing using NLTK
- TF-IDF feature extraction
- Logistic Regression model
- Interactive Streamlit web application
- Real-time prediction with confidence score

---

## Project Structure

```
Amazon-Product-Review-Sentiment-Analysis
│
├── app/
│   └── app.py
├── dataset/
│   └── amazonreviews.tsv
├── model/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebook/
│   └── Amazon_Product_Review_Sentiment_Analysis_Project-2.ipynb
├── images/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit

---

## Model Workflow

1. Load dataset
2. Perform text preprocessing
3. Apply TF-IDF vectorization
4. Train Logistic Regression model
5. Save model using Joblib
6. Predict sentiment using Streamlit

---

## Run the Project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

---

## Example

Input:

```
This product is amazing. I really loved it.
```

Prediction:

```
Positive 😊
```

Input:

```
Very poor quality. Completely disappointed.
```

Prediction:

```
Negative 😞
```

---

## Author

**Manohar Imandi**
