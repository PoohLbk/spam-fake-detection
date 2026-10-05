import os
import re
import joblib
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPAM_MODEL_PATH = os.path.join(BASE_DIR, 'spam_model.joblib')
FAKE_MODEL_PATH = os.path.join(BASE_DIR, 'fake_model.joblib')

# สั่งรัน train_models.py หากยังไม่มีไฟล์โมเดล
if not os.path.exists(SPAM_MODEL_PATH) or not os.path.exists(FAKE_MODEL_PATH):
    print("⚠️ ไม่พบไฟล์โมเดล .joblib ในโฟลเดอร์ backend... กำลังสั่งรัน train_models.py อัตโนมัติ")
    train_script_path = os.path.join(BASE_DIR, 'train_models.py')
    if os.path.exists(train_script_path):
        subprocess.run(["python", train_script_path], check=True)
        print("✅ สร้างไฟล์โมเดลเรียบร้อยแล้ว!")

# โหลดโมเดลเข้าสู่ระบบ
spam_model = joblib.load(SPAM_MODEL_PATH)
fake_model = joblib.load(FAKE_MODEL_PATH)
print("🎉 โหลดโมเดล Spam และ Fake News เข้าสู่ระบบสำเร็จ!")

class TextRequest(BaseModel):
    text: str

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

@app.get("/")
def read_root():
    return {"status": "online", "message": "FastAPI Spam & Fake News API is running successfully!"}

@app.post("/predict")
def predict_text(request: TextRequest):
    raw_text = request.text
    cleaned = clean_text(raw_text)
    if not cleaned:
        cleaned = raw_text

    try:
        spam_pred = spam_model.predict([cleaned])[0]
        spam_proba = spam_model.predict_proba([cleaned])[0][1]

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")
