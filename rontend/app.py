import streamlit as st
import requests

# URL ของ Render Backend (ต้องลงท้ายด้วย /predict)
BACKEND_URL = "https://spam-fake-detection.onrender.com/predict"

st.title("🛡️ Spam Email & Fake News Detector")
user_input = st.text_area("กรอกข้อความที่ต้องการตรวจสอบ:")

if st.button("วิเคราะห์ข้อความ"):
    if user_input.strip():
        try:
            # ส่งคำขอ POST ไปยัง Render Backend
            response = requests.post(BACKEND_URL, json={"text": user_input})
            
            if response.status_code == 200:
                result = response.json()
                st.write("ผลการวิเคราะห์:", result)
            else:
                st.error(f"เกิดข้อผิดพลาดจาก Server: {response.status_code}")
        except Exception as e:
            st.error(f"ไม่สามารถเชื่อมต่อ Backend ได้: {e}")
