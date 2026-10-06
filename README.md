# 🛡️ Thai Spam & Fake News Detection System

ระบบตรวจจับสแปมและข่าวปลอมภาษาไทย พัฒนาด้วย **Python, Scikit-learn, PyThaiNLP** และให้บริการผ่าน **FastAPI (Backend)** พร้อมหน้าจอผู้ใช้ **Streamlit (Frontend)**
https://spam-fake-detection-ngqssq394bmgdhgqckbaut.streamlit.app/
---

## 🚀 Features (คุณสมบัติเด่น)
- **Fake News Detection**: จำแนกข่าวจริงและข่าวปลอมภาษาไทยด้วยโมเดล Machine Learning
- **Spam Detection**: ตรวจจับข้อความสแปม หลอกลวง ลิงก์ดูดเงิน ทั้งภาษาไทยและภาษาอังกฤษ
- **Thai NLP Support**: ใช้ `PyThaiNLP (newmm)` ในการตัดคำภาษาไทยอย่างแม่นยำ
- **API & UI**: ให้บริการผ่าน REST API ด้วย FastAPI และมี Web Interface สำหรับทดสอบผ่าน Streamlit

---

## 📊 Model Performance & Evaluation (ประสิทธิภาพโมเดล)

โมเดล Fake News Detection ได้รับการเทรนและประเมินผลด้วยชุดข้อมูล **`EXt1/Thai-True-Fake-News`** จาก Hugging Face ผ่านกระบวนการ 5-Fold Cross-Validation

### 📈 Learning Curve (Accuracy vs Training Size)

**สรุปผลการประเมิน:**
- **Training Accuracy**: ~93.8%
- **Validation Accuracy**: ~89.0%
- โมเดลเรียนรู้ได้ดี เส้น Validation Accuracy เพิ่มขึ้นอย่างต่อเนื่องตามปริมาณข้อมูล และมีส่วนต่าง (Gap) กับ Training Accuracy เพียง ~5% แสดงว่าโมเดลไม่มีปัญหา Overfitting และมีความเสถียรสูงเมื่อนำไปใช้งานกับข้อมูลใหม่

---

## 🛠️ Tech Stack (เทคโนโลยีที่ใช้)
- **Language**: Python 3.10+
- **NLP & ML**: Scikit-learn, PyThaiNLP, Pandas, Joblib
- **Dataset**: Hugging Face (`EXt1/Thai-True-Fake-News`)
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Deployment**: Render / Streamlit Cloud

---

<img width="989" height="590" alt="accuracy_learning_curve_xy" src="https://github.com/user-attachments/assets/dc46b0d1-6d46-47a2-ac6d-06d89530546e" />
