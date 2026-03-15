📧 SmartMail AI – Intelligent Email Spam & Phishing Detection System

SmartMail AI is an advanced machine learning–based email security system designed to automatically detect spam, phishing, and malicious emails in real time. It combines natural language processing (NLP), ensemble learning, and rule-based threat analysis to provide accurate classification, explainability, and risk assessment for incoming emails.

The system supports live email input, highlights suspicious keywords, detects phishing links, assigns threat severity levels, and continuously learns from logged data.

🚀 Features

✅ Real-time email classification (Spam / Ham)
✅ Ensemble ML model (Naive Bayes + Logistic Regression)
✅ Advanced NLP preprocessing

Text cleaning

Stopword removal

Stemming

TF-IDF word and character n-grams

✅ Phishing link detection
✅ Threat scoring and severity classification
✅ Explainable AI – highlights spam-indicating keywords
✅ Confusion matrix and model evaluation
✅ Automatic logging of scanned emails
✅ Supports retraining using collected email logs

🧠 Technologies Used

Programming Language: Python

Libraries & Tools:

pandas, NumPy

scikit-learn

NLTK

matplotlib, seaborn

SciPy

joblib

🤖 Machine Learning Models

Multinomial Naive Bayes

Logistic Regression

Soft Voting Ensemble Classifier

Feature Engineering:

TF-IDF Word N-Grams

TF-IDF Character N-Grams

📂 Dataset

Initially trained using the UCI SMS Spam Collection Dataset.

Supports continuous learning using locally generated logs stored in:

dataset/email_logs.csv


Each prediction is logged with:

Email content

Classification result

Confidence score

Threat severity

Risk indicators

▶️ How to Run
1️⃣ Install dependencies
pip install pandas numpy nltk scikit-learn matplotlib seaborn scipy joblib

2️⃣ Download NLTK resources (first run only)
import nltk
nltk.download('stopwords')
nltk.download('punkt')

3️⃣ Run the application
python smart_mail.py

💡 Usage

Paste the full email content into the terminal.

Type END on a new line to analyze the email.

Type EXIT to quit the program.

The system displays:

Classification result

Category (Spam, Phishing, Scam, Safe)

Confidence level

Threat score

Highlighted suspicious words

Detected links

📊 Sample Output
RESULT      : SPAM 🚫
CATEGORY    : 🎣 PHISHING
CONFIDENCE  : 94.6%
SEVERITY    : 🔴 CRITICAL
THREAT SCORE: 9/10

🔮 Future Enhancements

Add Support Vector Machine (SVM) model

Web-based dashboard interface

Automatic periodic retraining

Email API integration

Deep learning support

Cloud deployment

🧑‍💻 Author

SUSMITHA P
