import streamlit as st
import requests

BACKEND_URL = "https://spam-fake-detection.onrender.com/predict"

st.title("🛡️ Spam Email & Fake News Detector")
user_input = st.text_area("กรอกข้อความที่ต้องการตรวจสอบ:")

if st.button("วิเคราะห์ข้อความ", type="primary"):
    if user_input.strip():
        with st.spinner("กำลังเชื่อมต่อ Server (หากเป็นการใช้งานครั้งแรก Server ฟรีอาจใช้เวลาปลุกตัวเองประมาณ 1 นาที)..."):
            try:
                # เพิ่ม timeout เป็น 120 วินาทีเพื่อรองรับ Cold Start บน Render
                response = requests.post(
                    BACKEND_URL, 
                    json={"text": user_input}, 
                    timeout=120
                )
                
                if response.status_code == 200:
                    st.success("ประมวลผลสำเร็จ!")
                    st.json(response.json())
                else:
                    st.error(f"Server ตอบกลับด้วย Status Code: {response.status_code}")
            except requests.exceptions.Timeout:
                st.error("⏳ หมดเวลาเชื่อมต่อ (Timeout): Server บน Render อยู่ในโหมดหลับ กำลังปลุกตัวเอง... กรุณากดปุ่มวิเคราะห์ใหม่อีกครั้งใน 10-20 วินาทีครับ")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}")
