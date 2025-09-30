import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

MODEL_FILE = "spam_nb_model.pkl"
TFIDF_FILE = "spam_tfidf.pkl"
VOCAB_FILE = "spam_vocab.pkl"


df = pd.read_csv("emails.csv")
X = df.drop(columns=["Email No.", "Prediction"])
y = df["Prediction"]


if os.path.exists(MODEL_FILE) and os.path.exists(TFIDF_FILE) and os.path.exists(VOCAB_FILE):
    print("Loading saved model and TF-IDF...")
    nb = joblib.load(MODEL_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    vocab = joblib.load(VOCAB_FILE)
else:
    print("Training new model...")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF
    tfidf = TfidfTransformer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train NB
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = nb.predict(X_test_tfidf)
    print("Accuracy:\t", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    # Save model, transformer, and vocab
    joblib.dump(nb, MODEL_FILE)
    joblib.dump(tfidf, TFIDF_FILE)
    vocab = [col for col in df.columns if col not in ["Email No.", "Prediction"]]
    joblib.dump(vocab, VOCAB_FILE)
    print("Model, TF-IDF, and vocab saved.")


def preprocess_email(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    return words

def email_to_feature(text, vocab):
    words = preprocess_email(text)
    words_count = {w: words.count(w) for w in set(words)}

    features = np.zeros(len(vocab))
    for i, word in enumerate(vocab):
        features[i] = words_count.get(word, 0)

    return features.reshape(1, -1)

def predict_email(text, vocab, tfidf, model):
    X_new = email_to_feature(text, vocab)
    X_new_tfidf = tfidf.transform(X_new)
    pred = model.predict(X_new_tfidf)[0]
    return "Spam" if pred == 1 else "Not Spam"


while True:
    email_text = input("Enter the email (or type 'exit' to quit):\t")
    if email_text.lower() in ["exit", "quit"]:
        break
    print("Prediction:", predict_email(email_text, vocab, tfidf, nb))
