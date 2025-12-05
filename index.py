import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

MODEL_FILE = "spam_nb_model.pkl"
TFIDF_FILE = "spam_tfidf.pkl"
VOCAB_FILE = "spam_vocab.pkl"

df = pd.read_csv("emails.csv")

# X = email text column
X = df["text"]     # <-- change this to the correct column name in your CSV
y = df["Prediction"]

# Load model if exists
if os.path.exists(MODEL_FILE) and os.path.exists(TFIDF_FILE) and os.path.exists(VOCAB_FILE):
    print("Loading saved model and transformers...")
    nb = joblib.load(MODEL_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    vectorizer = joblib.load(VOCAB_FILE)

else:
    print("Training a new model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert text → count vectors
    vectorizer = CountVectorizer(stop_words="english")
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)

    # TF-IDF
    tfidf = TfidfTransformer()
    X_train_tfidf = tfidf.fit_transform(X_train_counts)
    X_test_tfidf = tfidf.transform(X_test_counts)

    # Train NB classifier
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train_tfidf, y_train)

    # Evaluation
    y_pred = nb.predict(X_test_tfidf)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    joblib.dump(nb, MODEL_FILE)
    joblib.dump(tfidf, TFIDF_FILE)
    joblib.dump(vectorizer, VOCAB_FILE)
    print("\nModel and transformers saved.")

# ---------------------------------------
# Prediction Pipeline
# ---------------------------------------

def predict_email(text, vectorizer, tfidf, model):
    x_count = vectorizer.transform([text])
    x_tfidf = tfidf.transform(x_count)
    pred = model.predict(x_tfidf)[0]
    return "Spam" if pred == 1 else "Not Spam"

# Live Prediction Loop
while True:
    email_text = input("\nEnter the email (or 'exit'): ")
    if email_text.lower() in ["exit", "quit"]:
        break
    print("Prediction:", predict_email(email_text, vectorizer, tfidf, nb))
