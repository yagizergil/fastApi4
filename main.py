# main.py
import sqlite3
from urllib.request import Request
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentiment_model import predict_sentiment
from fastapi.middleware.cors import CORSMiddleware
from sentiment_model import save_sentiment_to_database
from fastapi.responses import JSONResponse
from sentiment_model import predict_sentiment, load_vectorizer, save_sentiment_to_database, get_sentiment_data


app = FastAPI()

origins = [
    "http://localhost:3000",  # Bu URL'i frontend'inizle değiştirin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE","OPTIONS"],  # Tüm HTTP metodlarını ekledik
    allow_headers=["*"],  # Tüm başlıklara izin veriyoruz, daha güvenli bir ayar yapabilirsiniz
)




vectorizer = joblib.load('vectorizer.pkl')
class SentimentInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    prediction: str
    lr_model_proba: list

@app.post("/predict_sentiment")

async def predict_sentiment_endpoint(input_data: SentimentInput):
    text = input_data.text
    prediction, lr_model_proba = predict_sentiment(text)
    return SentimentOutput(prediction=prediction, lr_model_proba=lr_model_proba)


@app.get("/get_sentiment_data")
async def get_sentiment_data_endpoint():
    data = get_sentiment_data()
    return data



@app.post("/save_sentiment")
async def save_sentiment_endpoint(input_data: SentimentInput):
    new_text_vector = vectorizer.transform([input_data.text])
    sentiment_prediction, _ = predict_sentiment(input_data.text, new_text_vector)
    label = sentiment_prediction.split()[0]
    await save_sentiment_to_database(input_data.text, label, sentiment_prediction)
    return {"mesaj": "Veriler başarıyla veritabanına kaydedildi."}

if __name__ == "__main__":
    app.run(port=8000)