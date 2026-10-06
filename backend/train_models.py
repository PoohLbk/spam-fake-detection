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

# กำหนด Features (X) และ Label (y)
X_fake = df_fake['text']
y_fake = df_fake['label']

# แบ่งข้อมูลเป็น Train Set (80%) และ Test Set (20%) สำหรับวัด Accuracy
X_train_fake, X_test_fake, y_train_fake, y_test_fake = train_test_split(
    X_fake, y_fake, test_size=0.2, random_state=42, stratify=y_fake
)

print(f"📊 ขนาดข้อมูล Fake News ทั้งหมด: {len(df_fake)} (Train: {len(X_train_fake)}, Test: {len(X_test_fake)})")

fake_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=thai_tokenizer, token_pattern=None, max_features=10000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# เทรนโมเดลด้วย Train Set
fake_pipeline.fit(X_train_fake, y_train_fake)

# ประเมินประสิทธิภาพบน Test Set (การหาค่า Accuracy)
y_pred_fake = fake_pipeline.predict(X_test_fake)
fake_acc = accuracy_score(y_test_fake, y_pred_fake)

print("\n==========================================")
print(f"🎯 Fake News Model Accuracy: {fake_acc * 100:.2f}%")
print("==========================================")
print(classification_report(y_test_fake, y_pred_fake, target_names=['Real News', 'Fake News']))

# บันทึกโมเดล
joblib.dump(fake_pipeline, FAKE_MODEL_PATH)
print("✅ บันทึก fake_model.joblib เรียบร้อยแล้ว!\n")


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
