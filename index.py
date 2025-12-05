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

# Load dataset
df = pd.read_csv("emails.csv")

# X = all word columns, y = label
X = df.drop(columns=["Email No.", "Prediction"])
y = df["Prediction"]

# Load previous model if available
if os.path.exists(MODEL_FILE) and os.path.exists(TFIDF_FILE) and os.path.exists(VOCAB_FILE):
    print("Loading saved model and TF-IDF...")
    nb = joblib.load(MODEL_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    vocab = joblib.load(VOCAB_FILE)

else:
    print("Training new model...")

    # Split into train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF transformer (we already have numeric bag-of-words)
    tfidf = TfidfTransformer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train Naive Bayes model
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = nb.predict(X_test_tfidf)
    print("Accuracy:\t", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    # Save model, TF-IDF, and vocab list
    joblib.dump(nb, MODEL_FILE)
    joblib.dump(tfidf, TFIDF_FILE)
    vocab = list(X.columns)   # WORD column names
    joblib.dump(vocab, VOCAB_FILE)

    print("Model, TF-IDF, and vocab saved.")


# ----------------------------- #
#     TEXT → FEATURES HELPER    #
# ----------------------------- #

def preprocess_email(text):
    """Clean email text → return word list."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    return words


def email_to_feature(text, vocab):
    """
    Convert input email text into a bag-of-words count vector.
    Length = number of vocabulary words.
    """
    words = preprocess_email(text)
    words_count = {w: words.count(w) for w in set(words)}

    features = np.zeros(len(vocab))
    for i, word in enumerate(vocab):
        features[i] = words_count.get(word, 0)

    return features.reshape(1, -1)


def predict_email(text, vocab, tfidf, model):
    words_vector = email_to_feature(text, vocab)

    # Convert to DataFrame to match training format
    X_new_df = pd.DataFrame(words_vector, columns=vocab)

    X_new_tfidf = tfidf.transform(X_new_df)
    pred = model.predict(X_new_tfidf)[0]

    return "Spam" if pred == 1 else "Not Spam"



# ----------------------------- #
#      USER INPUT LOOP          #
# ----------------------------- #

while True:
    email_text = input("Enter the email (or type 'exit' to quit):\t")
    if email_text.lower() in ["exit", "quit"]:
        break

    print("Prediction:", predict_email(email_text, vocab, tfidf, nb))
