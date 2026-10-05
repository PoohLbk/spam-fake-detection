import os
import re
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Spam & Fake News Detector API",
    description="API for classifying text into Spam/Ham and Fake/Real News"
)

# ----------------------------------------------------
# 1. CORS Middleware (เปิดให้ Streamlit เรียกใช้งานได้)
# ----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# 2. โหลดโมเดลอย่างปลอดภัยด้วย Dynamic Path
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPAM_MODEL_PATH = os.path.join(BASE_DIR, 'spam_model.joblib')
FAKE_MODEL_PATH = os.path.join(BASE_DIR, 'fake_model.joblib')

spam_model = joblib.load(SPAM_MODEL_PATH)
fake_model = joblib.load(FAKE_MODEL_PATH)

# ----------------------------------------------------
# 3. กำหนด Schema ของข้อมูลที่รับเข้ามา
# ----------------------------------------------------
class TextRequest(BaseModel):
    text: str

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# ----------------------------------------------------
# 4. Endpoints
# ----------------------------------------------------
# Root Endpoint สำหรับเช็กว่า Server ออนไลน์ไหม (GET Method)
@app.get("/")
def read_root():
    return {"status": "online", "message": "FastAPI Spam & Fake News API is running!"}

# Predict Endpoint สำหรับรับข้อความมาทำ Prediction (POST Method)
@app.post("/predict")
def predict_text(request: TextRequest):
    raw_text = request.text
    cleaned = clean_text(raw_text)
    
    if not cleaned:
        return {"error": "Text cannot be empty."}

    # พยากรณ์ Spam
    spam_pred = spam_model.predict([cleaned])[0]
    spam_proba = spam_model.predict_proba([cleaned])[0][1]

    # พยากรณ์ Fake News
    fake_pred = fake_model.predict([cleaned])[0]
    fake_proba = fake_model.predict_proba([cleaned])[0][1]

    return {
        "input_text": raw_text,
        "spam_analysis": {
            "is_spam": bool(spam_pred == 1),
            "confidence": float(spam_proba if spam_pred == 1 else 1 - spam_proba)
        },
        "fake_news_analysis": {
            "is_fake": bool(fake_pred == 1),
            "confidence": float(fake_proba if fake_pred == 1 else 1 - fake_proba)
        }
    }
