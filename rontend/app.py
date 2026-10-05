import streamlit as st
import requests

# ----------------------------------------------------
# 1. ตั้งค่าหน้าตาเบื้องต้น (Page Config)
# ----------------------------------------------------
st.set_page_config(
    page_title="AI Text Classifier - Spam & Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render Backend URL
BACKEND_URL = "https://spam-fake-detection.onrender.com/predict"

# ----------------------------------------------------
# 2. ตกแต่งด้วย Custom CSS (Modern & Clean Design)
# ----------------------------------------------------
st.markdown("""
<style>
    /* ปรับแต่งส่วนหัวของหน้า */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* ปรับแต่งการ์ดแสดงผล */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    .card-spam { background-color: #FEF2F2; border-left: 6px solid #EF4444; }
    .card-ham { background-color: #F0FDF4; border-left: 6px solid #22C55E; }
    .card-fake { background-color: #FFFBEB; border-left: 6px solid #F59E0B; }
    .card-real { background-color: #EFF6FF; border-left: 6px solid #3B82F6; }
</style>
""", unsafe_unsafe_html=True)

# ----------------------------------------------------
# 3. Sidebar (ข้อมูลระบบ & แนะนำการใช้งาน)
# ----------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/shield.png", width=80)
    st.title("เกี่ยวกับระบบ")
    st.info(
        "**AI Spam & Fake News Detector**\n\n"
        "ระบบตรวจจับสแปมและข่าวปลอมด้วยเทคโนโลยี Machine Learning (TF-IDF & Logistic Regression) "
        "ประมวลผลผ่าน FastAPI High-Performance Backend"
    )
    st.divider()
    st.caption("🚀 Version 1.0.0 | Hosted on Render & Streamlit Cloud")

# ----------------------------------------------------
# 4. Main UI Content
# ----------------------------------------------------
st.markdown('<div class="main-header">🛡️ AI Content Authenticity Verifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ระบบวิเคราะห์และตรวจสอบข้อความสแปม (Spam) และข่าวปลอม (Fake News) ด้วยปัญญาประดิษฐ์</div>', unsafe_allow_html=True)

# ตัวอย่างข้อความทดสอบแบบรวดเร็ว (Quick Test Buttons)
st.markdown("##### 💡 เลือกข้อความตัวอย่างเพื่อทดสอบอย่างรวดเร็ว:")
col_sample1, col_sample2, col_sample3 = st.columns(3)

sample_text = ""
if col_sample1.button("📧 ตัวอย่าง Spam Email"):
    sample_text = "URGENT: You have been selected to win a $1,000 Amazon Gift Card! Click here to claim your reward before it expires in 24 hours: http://bit.ly/claim-prize-now"

if col_sample2.button("📰 ตัวอย่าง Fake News"):
    sample_text = "BREAKING: Scientists confirm that drinking warm lemon water cures all virus infections instantly."

if col_sample3.button("✉️️ ตัวอย่าง ข้อความปกติ"):
    sample_text = "Hi team, please review the attached meeting notes for tomorrow's project discussion at 10 AM."

# ช่องกรอกข้อความ
user_input = st.text_area(
    "กรอกข้อความหรือเนื้อหาข่าวที่ต้องการตรวจสอบ:",
    value=sample_text,
    height=160,
    placeholder="พิมพ์หรือคัดลอกข้อความภาษาอังกฤษ/ไทย มาวางที่นี่..."
)

col_btn, _ = st.columns([1, 4])
with col_btn:
    submit_btn = st.button("🔍 เริ่มวิเคราะห์ข้อความ", type="primary", use_container_width=True)

# ----------------------------------------------------
# 5. การประมวลผลและการแสดงผลลัพธ์ (Result Dashboard)
# ----------------------------------------------------
if submit_btn:
    if not user_input.strip():
        st.warning("⚠️ กรุณากรอกข้อความก่อนกดเริ่มวิเคราะห์")
    else:
        with st.spinner("⏳ กำลังเชื่อมต่อ AI Engine บน Render Cloud เพื่อวิเคราะห์ข้อมูล..."):
            try:
                response = requests.post(BACKEND_URL, json={"text": user_input}, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    st.divider()
                    st.subheader("📊 ผลการวิเคราะห์ (Analysis Results)")

                    spam_info = result.get("spam_analysis", {})
                    fake_info = result.get("fake_news_analysis", {})

                    col_res1, col_res2 = st.columns(2)

                    # --- แสดงผล Spam Analysis ---
                    with col_res1:
                        is_spam = spam_info.get("is_spam", False)
                        spam_conf = spam_info.get("confidence", 0.0)
                        
                        if is_spam:
                            st.markdown(
                                f"""
                                <div class="result-card card-spam">
                                    <h3 style="color:#DC2626; margin:0;">🚨 ตรวจพบ SPAM</h3>
                                    <p style="color:#7F1D1D; margin-top:5px;">ข้อความนี้มีแนวโน้มเป็นสแปม หรือการหลอกลวง</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="result-card card-ham">
                                    <h3 style="color:#16A34A; margin:0;">✅ ข้อความทั่วไป (HAM)</h3>
                                    <p style="color:#14532D; margin-top:5px;">ไม่พบพฤติกรรมหรือคำสุ่มเสี่ยงสแปม</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        st.write(f"**ความมั่นใจของ AI (Confidence):** {spam_conf * 100:.2f}%")
                        st.progress(spam_conf)

                    # --- แสดงผล Fake News Analysis ---
                    with col_res2:
                        is_fake = fake_info.get("is_fake", False)
                        fake_conf = fake_info.get("confidence", 0.0)

                        if is_fake:
                            st.markdown(
                                f"""
                                <div class="result-card card-fake">
                                    <h3 style="color:#D97706; margin:0;">⚠️ มีแนวโน้มเป็น ข่าวปลอม</h3>
                                    <p style="color:#78350F; margin-top:5px;">เนื้อหามีสัญลักษณ์ข่าวโฆษณาชวนเชื่อ หรือข่าวบิดเบือน</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="result-card card-real">
                                    <h3 style="color:#2563EB; margin:0;">📰 มีแนวโน้มเป็น ข่าวจริง</h3>
                                    <p style="color:#1E3A8A; margin-top:5px;">เนื้อหามีลักษณะใกล้เคียงกับข่าวสารทั่วไป</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        st.write(f"**ความมั่นใจของ AI (Confidence):** {fake_conf * 100:.2f}%")
                        st.progress(fake_conf)

                else:
                    st.error(f"❌ เกิดข้อผิดพลาดจากเซิร์ฟเวอร์ (Status Code: {response.status_code})")
            
            except requests.exceptions.Timeout:
                st.error("⏳ **Timeout:** เซิร์ฟเวอร์ Render อยู่ในโหมด Cold Start กรุณากดปุ่มวิเคราะห์ใหม่อีกครั้งใน 10-20 วินาที")
            except Exception as e:
                st.error(f"❌ ไม่สามารถเชื่อมต่อกับ Backend ได้: {e}")
