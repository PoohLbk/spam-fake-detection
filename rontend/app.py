import streamlit as st
import requests

# การตั้งค่าหน้า Streamlit
st.set_page_config(page_title="AI Spam & Fake News Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ Spam Email & Fake News Detector")
st.write("ป้อนข้อความหรือเนื้อหาอีเมลลงในช่องด้านล่างเพื่อตรวจสอบสแปมและข่าวปลอม")

# ช่องรับข้อความจากผู้ใช้
user_input = st.text_area("กรอกข้อความที่ต้องการตรวจสอบ:", height=150, placeholder="Type or paste text here...")

# ปุ่มกดวิเคราะห์
if st.button("ตรวจจับข้อความ (Analyze)", type="primary"):
    if not user_input.strip():
        st.warning("กรุณากรอกข้อความก่อนกดตรวจจับ")
    else:
        with st.spinner("กำลังประมวลผลด้วย AI..."):
            try:
                # เรียกไปยัง API ของ FastAPI
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"text": user_input}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.divider()
                    st.subheader("📊 ผลการวิเคราะห์")

                    col1, col2 = st.columns(2)

                    # แสดงผล Spam
                    spam_info = result["spam_analysis"]
                    with col1:
                        st.markdown("### 📧 Spam Email")
                        if spam_info["is_spam"]:
                            st.error("⚠️ ตรวจพบข้อความ SPAM")
                        else:
                            st.success("✅ ข้อความปกติตัวจริง (HAM)")
                        st.caption(f"ความมั่นใจ: {spam_info['confidence']*100:.2f}%")

                    # แสดงผล Fake News
                    fake_info = result["fake_news_analysis"]
                    with col2:
                        st.markdown("### 📰 Fake News")
                        if fake_info["is_fake"]:
                            st.error("⚠️ มีแนวโน้มเป็น ข่าวปลอม")
                        else:
                            st.success("✅ มีแนวโน้มเป็น ข่าวจริง")
                        st.caption(f"ความมั่นใจ: {fake_info['confidence']*100:.2f}%")

                else:
                    st.error("เกิดข้อผิดพลาดในการเชื่อมต่อกับ Server")

            except Exception as e:
                st.error(f"ไม่สามารถเชื่อมต่อกับ FastAPI Backend ได้: {e}")
