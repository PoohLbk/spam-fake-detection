import joblib
import re
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Text Classification API",
    description="API for detecting Spam emails and Fake News text."
)

# โหลดโมเดล
spam_model = joblib.load('spam_model.joblib')
fake_model = joblib.load('fake_model.joblib')

class TextRequest(BaseModel):
    text: str

def clean_text(text: str) -> str:
    """ฟังก์ชันทำความสะอาดข้อความเบื้องต้น"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # ลบ URL
    text = re.sub(r'[^\w\s]', '', text) # ลบเครื่องหมายวรรคตอน
    return text.strip()

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
