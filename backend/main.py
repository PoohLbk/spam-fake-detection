from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os

app = FastAPI()

# ----------------------------------------------------
# เพิ่ม CORS Middleware (จำเป็นมากสำหรับการ Deploy)
# ----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตให้ทุก Domain เรียกใช้งานได้
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root Endpoint สำหรับเช็กว่า Server ยังทำงานอยู่ไหม
@app.get("/")
def read_root():
    return {"status": "online", "message": "FastAPI Spam & Fake News API is running!"}

# ... โค้ดส่วนอื่นๆ คงเดิม ...
