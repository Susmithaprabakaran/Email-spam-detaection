import pandas as pd
import numpy as np
import re, os, nltk
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------- Color Engine ----------------
class C:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    ORANGE = "\033[38;5;208m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

# ---------------- UI Engine ----------------
def ui_line(char="─", n=78): return char * n

def ui_box(title):
    print(f"\n{C.BLUE}{ui_line('═')}")
    print(f"│ {title.center(74)} │")
    print(f"{ui_line('═')}{C.RESET}")

def ui_section(title):
    print(f"\n{C.PURPLE}{ui_line('─')}")
    print(f"▶ {title}")
    print(f"{ui_line('─')}{C.RESET}")

# ---------------- NLTK ----------------
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('punkt')

# =====================================================
# ✅ LOAD DATASET FROM email_logs.csv
# =====================================================
DATA_FILE = "email_logs.csv"

if not os.path.exists(DATA_FILE):
    print(f"{C.RED}❌ Dataset file not found: {DATA_FILE}{C.RESET}")
    exit()

df = pd.read_csv(DATA_FILE)

# Keep only required columns
df = df[['email', 'result']]

# Rename
df.columns = ['text', 'label']

# Convert text to string to ensure .str accessor works
df['text'] = df['text'].astype(str)

# Normalize labels → spam / ham
df['label'] = df['label'].astype(str).str.upper()
df['label'] = df['label'].apply(lambda x: 'spam' if 'SPAM' in x else 'ham')

# Remove empty rows
df.dropna(subset=['text'], inplace=True)
df = df[df['text'].str.strip() != ""]

print(f"\n{C.BLUE}{C.BOLD}Dataset loaded successfully{C.RESET}")
print(df['label'].value_counts())

# Safety check
if len(df) < 30:
    print(f"{C.YELLOW}⚠ Not enough samples for training yet. Collect more emails.{C.RESET}")
    exit()

# ---------------- Text Cleaning ----------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " link ", text)
    text = re.sub('[^a-z0-9 ]', ' ', text)  # Keep alphanumeric and spaces
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)
# Remove rows with empty cleaned text
df = df[df['clean_text'].str.strip() != ""]

# ---------------- Feature Extraction ----------------
word_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=1)
char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,4), min_df=1)

X = hstack([
    word_vectorizer.fit_transform(df['clean_text']),
    char_vectorizer.fit_transform(df['clean_text'])
])

y = df['label'].map({'ham':0, 'spam':1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Models ----------------
nb = MultinomialNB()
lr = LogisticRegression(max_iter=2000, class_weight='balanced')

ensemble = VotingClassifier(
    estimators=[('nb', nb), ('lr', lr)],
    voting='soft',
    weights=[1,2]
)

# ---------------- Cross Validation ----------------
print("\nRunning cross-validation...")
scores = cross_val_score(ensemble, X, y, cv=5, scoring='f1')
print("Cross-validated F1 Score:", round(scores.mean(),4))

# ---------------- Train ----------------
ensemble.fit(X_train, y_train)

# ---------------- Evaluation ----------------
y_pred = ensemble.predict(X_test)
print("\n===== FINAL MODEL RESULTS =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred),4))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - SmartMail AI")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------- Save Models ----------------
joblib.dump(ensemble, "smartmail_model.pkl")
joblib.dump(word_vectorizer, "word_vectorizer.pkl")
joblib.dump(char_vectorizer, "char_vectorizer.pkl")

# =====================================================
# 🔥 REST OF YOUR PIPELINE CONTINUES UNCHANGED
# (Threat scoring, phishing detection, UI, logging, etc.)
# =====================================================

print(f"\n{C.GREEN}✅ Model trained successfully using email_logs.csv{C.RESET}")
print(f"{C.CYAN}🚀 SmartMail AI is ready for live detection.{C.RESET}")
