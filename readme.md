📧 Spam Email Detector (Naive Bayes + TF-IDF)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=pythonhttps://img.shields.io/badge/ML-scikit--learn-orange?

![License](https://img.shields.io/badge

![Status](https://img.shields.io/badge

A machine learning-based Spam Detector built on Naive Bayes and TF-IDF vectorization with scikit-learn.
The model classifies emails as Spam or Ham (Not Spam) using the Email Spam Classification Dataset.

🚀 Features
Text preprocessing (cleaning, regex, tokenization).

TF-IDF vectorization for feature engineering.

Multinomial Naive Bayes classifier for training & prediction.

Model persistence with Joblib (saves trained model, TF-IDF transformer, and vocabulary).

Interactive CLI: Enter an email → instantly get classified as Spam or Ham.

Achieves 95%+ accuracy on test data.

🛠️ Tech Stack
Language: Python 3.x

Libraries: Pandas, NumPy, scikit-learn, Joblib, Regex

📂 Project Structure
text
Spam_detector/
│── emails.csv              # Dataset
│── index.py                # Main script
│── spam_nb_model.pkl       # Saved Naive Bayes model
│── spam_tfidf.pkl          # Saved TF-IDF transformer
│── spam_vocab.pkl          # Saved vocabulary
│── requirements.txt        # Dependencies
│── README.md               # Project documentation
⚙️ Installation & Setup
Clone the repo:

bash
git clone https://github.com/AdityaDabgotra/spam-detector.git
cd spam-detector
Install dependencies:

bash
pip install -r requirements.txt
Download dataset from Kaggle and place it as emails.csv in the root directory.

▶️ Usage
Train the Model
bash
python index.py --train
Run Interactive CLI for Predictions
bash
python index.py --predict
You can then input example emails such as:

text
Input: "Congratulations! You won a free lottery ticket, click here to claim"
Output: SPAM 🚨

Input: "Hi Team, please find attached the project report for review."
Output: HAM ✅
📊 Model Performance
Metric	Score
Accuracy	95%+
Precision	High
Recall	High
F1-Score	High
🔮 Future Improvements
Deploy as a REST API (Flask/FastAPI) for real-time spam detection.

Add a web-based UI for ease of use.

Experiment with more models:

Logistic Regression

Random Forest

Deep Neural Networks

👨‍🏫 Learnings
TF-IDF significantly improves text feature representation.

Naive Bayes is powerful for text data due to word independence assumptions.

Saving/loading models ensures reproducibility and deployment readiness.

📜 License
This project is licensed under the MIT License.

