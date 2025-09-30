# 📧 Spam Email Detector (Naive Bayes + TF-IDF)

This project is a **machine learning-based Spam Detector** built using **Naive Bayes** and **TF-IDF vectorization** with `scikit-learn`.  
It classifies emails as **Spam** or **Not Spam (Ham)** based on the [Email Spam Classification Dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv).  

---

## 🚀 Features
- Preprocessing of email text (cleaning & tokenization).
- TF-IDF vectorization for feature engineering.
- Multinomial Naive Bayes classifier for training & prediction.
- Model persistence with **Joblib** (saves trained model, vocab, and TF-IDF).
- Interactive CLI: Enter an email and get instant classification.
- High accuracy on test data.

---

## 🛠️ Tech Stack
- Python 3.x
- Pandas, NumPy
- scikit-learn
- Joblib
- Regex (for preprocessing)

---

## 📂 Project Structure
Spam_detector/
│── emails.csv # Dataset
│── index.py # Main script
│── spam_nb_model.pkl # Saved model (after training)
│── spam_tfidf.pkl # Saved TF-IDF transformer
│── spam_vocab.pkl # Saved vocabulary
│── README.md # Project documentation


---

## ⚙️ Installation
1. Clone the repository:
    ```bash
        git clone https://github.com/AdityaDabgotra/spam-detector.git
        cd spam-detector
    ```
2. Install dependencies:
    ```bash
        pip install -r requirements.txt
    ```
3. Download dataset from Kaggle
    Place emails.csv in the project root.

## 📊 Model Performance

    On test data:

    High accuracy (typically >95% with proper preprocessing).

    Detailed metrics (precision, recall, F1-score) are shown during training.

## 🔮 Future Improvements

    Add a web API using Flask/FastAPI for real-time spam detection.

    Deploy as a simple web app with a UI.

        Try advanced models (Logistic Regression, Random Forest, Deep Learning).

## 👨‍🏫 Learnings

    How TF-IDF improves feature representation.

    Why Naive Bayes is effective for text classification.

    Importance of saving/loading trained models for reusability.

## 📜 License
    This project is licensed under the MIT License.
