import os
import joblib
import pandas as pd
from datasets import load_dataset
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPAM_MODEL_PATH = os.path.join(BASE_DIR, 'spam_model.joblib')
FAKE_MODEL_PATH = os.path.join(BASE_DIR, 'fake_model.joblib')

def thai_tokenizer(text):
    return word_tokenize(str(text), engine='newmm')

# ----------------------------------------------------
# 1. Train & Evaluate Fake News Model (Hugging Face)
# ----------------------------------------------------
print("⏳ กำลังโหลด Dataset ข่าวจริง/ข่าวปลอมภาษาไทยจาก Hugging Face...")
dataset = load_dataset("EXt1/Thai-True-Fake-News")
df_fake = pd.DataFrame(dataset['train'])

# พิมพ์ชื่อคอลัมน์ทั้งหมดออกมาดูเพื่อความชัวร์
print(f"📋 ชื่อคอลัมน์ใน Dataset: {df_fake.columns.tolist()}")

# เช็คชื่อคอลัมน์ข้อความอัตโนมัติ
text_col = None
for col in ['text', 'content', 'news', 'statement', 'text_th']:
    if col in df_fake.columns:
        text_col = col
        break

# เช็คชื่อคอลัมน์ Label อัตโนมัติ
label_col = None
for col in ['label', 'target', 'class', 'is_fake']:
    if col in df_fake.columns:
        label_col = col
        break

if not text_col or not label_col:
    # ถ้าหาไม่เจอ ให้ใช้คอลัมน์แรกเป็นข้อความ คอลัมน์ที่สองเป็น label
    text_col = df_fake.columns[0]
    label_col = df_fake.columns[1]

print(f"🎯 ใช้คอลัมน์ข้อความ: '{text_col}' และ คอลัมน์ Label: '{label_col}'")

# กำหนด Features (X) และ Label (y)
X_fake = df_fake[text_col]
y_fake = df_fake[label_col]
# ----------------------------------------------------
# 2. Train & Evaluate Spam Model
# ----------------------------------------------------
print("⏳ กำลังเตรียมข้อมูลและเทรน Spam Model...")
spam_data = [
    # SPAM (1)
    ("Win a $1000 Walmart gift card now! Click here", 1),
    ("Congratulations! You have been selected for a free prize", 1),
    ("ยินดีด้วย! คุณได้รับอนุมัติวงเงินกู้ฉุกเฉิน 50,000 บาท ดอกเบี้ย 0% ไม่ต้องค้ำประกัน แอดไลน์เลย", 1),
    ("ด่วน! สิทธิ์กู้ยืมเงินด่วนอนุมัติไวใน 5 นาที ถอนเงินได้ทันที คลิก http://bit.ly/claim", 1),
    ("เว็บตรงไม่ผ่านเอเย่นต์ สมัครวันนี้รับโบนัสฟรี 300% ฝากถอนออโต้", 1),
    ("URGENT: You have been selected to win a $1,000 Amazon Gift Card!", 1),
    
    # HAM (0)
    ("Please review the meeting minutes from yesterday", 0),
    ("Are we still meeting for lunch today at 12?", 0),
    ("ขอส่งสรุปผลการประชุมประจำสัปดาห์ และกำหนดการส่งมอบงานครับ", 0),
    ("เรียนทุกท่าน รบกวนตรวจสอบเอกสารแนบสำหรับวาระการประชุมวันพรุ่งนี้ครับ", 0),
    ("แจ้งเตือนยอดชำระค่าบริการอินเทอร์เน็ตประจำเดือน สามารถชำระผ่านแอปพลิเคชันธนาคารได้", 0),
    ("Hi team, please review the attached slide deck for tomorrow's project meeting", 0)
]

X_spam, y_spam = zip(*spam_data)

# แบ่งข้อมูลทำ Evaluation (หากมีข้อมูลปริมาณมาก)
X_train_spam, X_test_spam, y_train_spam, y_test_spam = train_test_split(
    X_spam, y_spam, test_size=0.2, random_state=42
)

spam_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=thai_tokenizer, token_pattern=None)),
    ('clf', LogisticRegression())
])

spam_pipeline.fit(X_train_spam, y_train_spam)

# ประเมินค่า Accuracy
y_pred_spam = spam_pipeline.predict(X_test_spam)
spam_acc = accuracy_score(y_test_spam, y_pred_spam)

print("==========================================")
print(f"🎯 Spam Model Accuracy: {spam_acc * 100:.2f}%")
print("==========================================")

# เซฟโมเดลโดยใช้ข้อมูลทั้งหมด
spam_pipeline.fit(X_spam, y_spam)
joblib.dump(spam_pipeline, SPAM_MODEL_PATH)
print("✅ บันทึก spam_model.joblib เรียบร้อยแล้ว!")
