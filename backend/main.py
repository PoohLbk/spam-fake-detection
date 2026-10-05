import os
import re
import joblib
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------------------------------------
# 1. เริ่มต้นสร้าง FastAPI App
# ----------------------------------------------------
app = FastAPI(
    title="Spam & Fake News Detector API",
    description="API สำหรับตรวจจับข้อความสแปม (Spam) และข่าวปลอม (Fake News) ด้วย Machine Learning"
)

# ----------------------------------------------------
# 2. ตั้งค่า CORS Middleware 
# (จำเป็นมากสำหรับการ Deploy บน Render เพื่อให้ Streamlit เรียกใช้ได้)
# ----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตให้ทุก Domain เรียกใช้งาน API นี้ได้
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# 3. จัดการตำแหน่งไฟล์โมเดล และสั่ง Train อัตโนมัติหากหาไฟล์ไม่พบ
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPAM_MODEL_PATH = os.path.join(BASE_DIR, 'spam_model.joblib')
FAKE_MODEL_PATH = os.path.join(BASE_DIR, 'fake_model.joblib')

# ตรวจสอบว่ามีไฟล์โมเดลทั้งสองในโฟลเดอร์ backend หรือไม่
if not os.path.exists(SPAM_MODEL_PATH) or not os.path.exists(FAKE_MODEL_PATH):
    print("⚠️ ไม่พบไฟล์โมเดล .joblib ในโฟลเดอร์ backend... กำลังสั่งรัน train_models.py อัตโนมัติ")
    train_script_path = os.path.join(BASE_DIR, 'train_models.py')
    
    if os.path.exists(train_script_path):
        subprocess.run(["python", train_script_path], check=True)
        print("✅ สร้างไฟล์โมเดลเรียบร้อยแล้ว!")
    else:
        print("❌ ไม่พบไฟล์ train_models.py สำหรับสร้างโมเดล")

# โหลดโมเดลเตรียมพร้อมใช้งาน
try:
    spam_model = joblib.load(SPAM_MODEL_PATH)
    fake_model = joblib.load(FAKE_MODEL_PATH)
    print("🎉 โหลดโมเดล Spam และ Fake News เข้าสู่ระบบสำเร็จ!")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")

# ----------------------------------------------------
# 4. โครงสร้างข้อมูลที่รับจาก Client (Schema) & Preprocessing
# ----------------------------------------------------
class TextRequest(BaseModel):
    text: str

def clean_text(text: str) -> str:
    """ฟังก์ชันทำความสะอาดข้อความเบื้องต้น"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # ลบ URL
    text = re.sub(r'[^\w\s]', '', text)  # ลบเครื่องหมายวรรคตอน
    return text.strip()

# ----------------------------------------------------
# 5. API Endpoints
# ----------------------------------------------------

# Root Endpoint สำหรับตรวจเช็กสถานะเซิร์ฟเวอร์ (GET Request)
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "FastAPI Spam & Fake News API is running successfully!"
    }

# Predict Endpoint สำหรับรับข้อความมาพยากรณ์ผล (POST Request)
@app.post("/predict")
def predict_text(request: TextRequest):
    raw_text = request.text
    cleaned = clean_text(raw_text)
    
    if not cleaned:
        raise HTTPException(status_code=400, detail="ข้อความที่ส่งมาต้องไม่เป็นค่าว่าง")

    try:
        # 1. พยากรณ์ Spam (1 = Spam, 0 = Ham)
        spam_pred = spam_model.predict([cleaned])[0]
        spam_proba = spam_model.predict_proba([cleaned])[0][1]

        # 2. พยากรณ์ Fake News (1 = Fake, 0 = Real)
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
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการประมวลผลโมเดล: {str(e)}")
