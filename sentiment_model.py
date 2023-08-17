# sentiment_model.py
import string
import sqlite3
import aiosqlite
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError


nltk.download('punkt')
nltk.download('stopwords')

conn = sqlite3.connect('sentiment_analysis2.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS sentiment_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                proba REAL NOT NULL
             )''')


conn.commit()
conn.close()

async def save_sentiment_to_database(text, prediction, proba):
    async with aiosqlite.connect('sentiment_analysis2.db') as conn:
        c = await conn.cursor()
        await c.execute('INSERT INTO sentiment_analysis (text, prediction, proba) VALUES (?, ?, ?)', (text, prediction, proba))
        await conn.commit()

def get_sentiment_data():
    conn = sqlite3.connect('sentiment_analysis2.db')
    c = conn.cursor()

    data = c.execute('SELECT * FROM sentiment_analysis').fetchall()

    conn.close()

    return data


def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    words = word_tokenize(text)                                       #çeviri tablosu maketrans ile noktalama işaretlerini kaldırdıtan sonra tokenize yaptık
    stop_words = set(stopwords.words('english'))                      #çünkü tokenize noktalama işaretlerini de bir karakter olarak ele alır.
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if len(word) > 1]
    return " ".join(words)

def load_lr_model():
    return joblib.load('lr_model.pkl')

def load_vectorizer():
    return joblib.load('vectorizer.pkl')

def predict_sentiment(text, new_text_vector=None):
    if new_text_vector is None:
        vectorizer = load_vectorizer()
        new_text_vector = vectorizer.transform([text])

    lr_model = load_lr_model()
    lr_model_proba = lr_model.predict_proba(new_text_vector)[0].tolist()

    class_labels = ["negative", "neutral", "positive"]
    max_prob_index = lr_model_proba.index(max(lr_model_proba))
    sentiment_prediction = f"{class_labels[max_prob_index]} %{max(lr_model_proba) * 100:.2f}"

    return sentiment_prediction, lr_model_proba



