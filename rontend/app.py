import streamlit as st
import requests

# URL ของ Render Backend (ต้องระบุ /predict ต่อท้าย)
BACKEND_URL = "https://spam-fake-detection.onrender.com/predict"

st.title("Spam Email & Fake News Detector")
user_input = st.text_area("กรอกข้อความที่ต้องการตรวจสอบ:")

if st.button("วิเคราะห์ข้อความ"):
    if user_input.strip():
        with st.spinner("กำลังเชื่อมต่อ Server (กรณี Server เพิ่งตื่นอาจใช้เวลา 30-60 วินาที)..."):
            try:
                # ตั้ง timeout ไว้ 60 วินาทีเผื่อ Server พึ่งปลุกตัวเอง (Cold Start)
                response = requests.post(BACKEND_URL, json={"text": user_input}, timeout=60)
                
                if response.status_code == 200:
                    st.success("ประมวลผลสำเร็จ!")
                    st.json(response.json())
                else:
                    st.error(f"Server ตอบกลับด้วย Status Code: {response.status_code}")
            except requests.exceptions.Timeout:
                st.error("หมดเวลาเชื่อมต่อ (Timeout) กรุณากดลองใหม่อีกครั้ง")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
