import streamlit as st
import requests

# ----------------------------------------------------
# 1. Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Text Classification System - Spam & Fake News",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render Backend URL
BACKEND_URL = "https://spam-fake-detection.onrender.com/predict"

# ----------------------------------------------------
# 2. Custom CSS for Clean & Professional UI
# ----------------------------------------------------
st.markdown("""
<style>
    /* Clean Header Typography - ปรับฟอนต์หัวข้อเป็นสีขาว */
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #FFFFFF !important; /* บังคับเปลี่ยนสีฟอนต์เป็นสีขาว */
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 12px 20px;
        background: linear-gradient(135deg, #1E293B, #0F172A); /* ใส่พื้นหลังเข้มเพื่อให้ข้อความสีขาวเด่นชัด */
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .sub-header {
        font-size: 0.95rem;
        color: #94A3B8; /* ปรับสีคำอธิบายเป็นสีเทาสว่างเพื่อรองรับข้อความสีขาว */
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 1.8rem;
    }
    /* Result Cards Design */
    .result-card {
        padding: 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
    }
    .card-spam { background-color: #FEF2F2; border-left: 5px solid #DC2626; }
    .card-ham { background-color: #F0FDF4; border-left: 5px solid #16A34A; }
    .card-fake { background-color: #FFFBEB; border-left: 5px solid #D97706; }
    .card-real { background-color: #EFF6FF; border-left: 5px solid #2563EB; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 3. Sidebar (Technical & System Information)
# ----------------------------------------------------
with st.sidebar:
    st.title("ระบบวิเคราะห์ข้อความ")
    st.markdown(
        "**AI Content Classifier**\n\n"
        "ระบบประมวลผลภาษาธรรมชาติเพื่อจำแนกข้อความสแปม (Spam) "
        "และตรวจสอบแนวโน้มข่าวปลอม (Fake News) พัฒนาด้วย Scikit-learn และ FastAPI"
    )
    st.divider()
    st.caption("ระบบเชื่อมต่อ API บน Render Cloud")

# ----------------------------------------------------
# 4. Main UI Layout
# ----------------------------------------------------
st.markdown('<div class="main-header">🛡️ ระบบตรวจสอบข้อความสแปมและข่าวปลอม</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Text Authenticity and Intent Classification Engine</div>', unsafe_allow_html=True)

# Quick Test Buttons
st.markdown("##### ข้อความตัวอย่างสำหรับทดสอบ:")
col_sample1, col_sample2, col_sample3 = st.columns(3)

sample_text = ""
if col_sample1.button("ตัวอย่าง Spam Email", use_container_width=True):
    sample_text = "URGENT: You have been selected to win a $1,000 Amazon Gift Card! Click here to claim your reward before it expires in 24 hours: http://bit.ly/claim-prize-now"

if col_sample2.button("ตัวอย่าง Fake News", use_container_width=True):
    sample_text = "BREAKING: Scientists confirm that drinking warm lemon water cures all virus infections instantly."

if col_sample3.button("ตัวอย่าง ข้อความทั่วไป", use_container_width=True):
    sample_text = "Hi team, please review the attached meeting notes for tomorrow's project discussion at 10 AM."

# Text Area Input
user_input = st.text_area(
    "กรอกข้อความที่ต้องการวิเคราะห์:",
    value=sample_text,
    height=150,
    placeholder="ระบุข้อความภาษาอังกฤษหรือภาษาไทยเพื่อนำเข้าประมวลผล..."
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    submit_btn = st.button("ประมวลผลข้อความ", type="primary", use_container_width=True)

# ----------------------------------------------------
# 5. Results & Analysis Output
# ----------------------------------------------------
if submit_btn:
    if not user_input.strip():
        st.warning("กรุณาระบุข้อความก่อนเริ่มการวิเคราะห์")
    else:
        with st.spinner("กำลังเชื่อมต่อเซิร์ฟเวอร์เพื่อประมวลผลข้อมูล..."):
            try:
                response = requests.post(BACKEND_URL, json={"text": user_input}, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    st.divider()
                    st.subheader("ผลการวิเคราะห์")

                    spam_info = result.get("spam_analysis", {})
                    fake_info = result.get("fake_news_analysis", {})

                    col_res1, col_res2 = st.columns(2)

                    # --- 1. Spam Analysis Card ---
                    with col_res1:
                        st.markdown("##### การตรวจสอบข้อความสแปม (Spam)")
                        is_spam = spam_info.get("is_spam", False)
                        spam_conf = spam_info.get("confidence", 0.0)
                        
                        if is_spam:
                            st.markdown(
                                f"""
                                <div class="result-card card-spam">
                                    <h4 style="color:#DC2626; margin:0;">⚠️ ตรวจพบ SPAM</h4>
                                    <p style="color:#7F1D1D; margin-top:5px; margin-bottom:0; font-size:0.9rem;">
                                        ข้อความนี้มีลักษณะเข้าข่ายการสแปม หรือการหลอกลวง
                                    </p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="result-card card-ham">
                                    <h4 style="color:#16A34A; margin:0;">✅ ข้อความทั่วไป (HAM)</h4>
                                    <p style="color:#14532D; margin-top:5px; margin-bottom:0; font-size:0.9rem;">
                                        ไม่พบสัญลักษณ์หรือพฤติกรรมเข้าข่ายข้อความสแปม
                                    </p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        st.write(f"ค่าความน่าจะเป็น (Confidence): **{spam_conf * 100:.2f}%**")
                        st.progress(spam_conf)

                    # --- 2. Fake News Analysis Card ---
                    with col_res2:
                        st.markdown("##### การตรวจสอบข่าวปลอม (Fake News)")
                        is_fake = fake_info.get("is_fake", False)
                        fake_conf = fake_info.get("confidence", 0.0)

                        if is_fake:
                            st.markdown(
                                f"""
                                <div class="result-card card-fake">
                                    <h4 style="color:#D97706; margin:0;">⚠️ มีแนวโน้มเป็น ข่าวปลอม</h4>
                                    <p style="color:#78350F; margin-top:5px; margin-bottom:0; font-size:0.9rem;">
                                        เนื้อหามีสัญลักษณ์ของข่าวชวนเชื่อ หรือข้อมูลบิดเบือน
                                    </p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="result-card card-real">
                                    <h4 style="color:#2563EB; margin:0;">✅ มีแนวโน้มเป็น ข่าวจริง</h4>
                                    <p style="color:#1E3A8A; margin-top:5px; margin-bottom:0; font-size:0.9rem;">
                                        เนื้อหามีลักษณะโครงสร้างภาษาใกล้เคียงข่าวสารทั่วไป
                                    </p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        st.write(f"ค่าความน่าจะเป็น (Confidence): **{fake_conf * 100:.2f}%**")
                        st.progress(fake_conf)

                else:
                    st.error(f"เกิดข้อผิดพลาดจากเซิร์ฟเวอร์ (HTTP {response.status_code})")
            
            except requests.exceptions.Timeout:
                st.error("การเชื่อมต่อหมดเวลา (Timeout) เนื่องจากเซิร์ฟเวอร์อยู่ในช่วง Cold Start กรุณากดประมวลผลใหม่อีกครั้งใน 10-20 วินาที")
            except Exception as e:
                st.error(f"ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์ประมวลผลได้: {e}")
