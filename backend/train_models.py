import os
import joblib
import pandas as pd
from datasets import load_dataset
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPAM_MODEL_PATH = os.path.join(BASE_DIR, 'spam_model.joblib')
FAKE_MODEL_PATH = os.path.join(BASE_DIR, 'fake_model.joblib')

def thai_tokenizer(text):
    return word_tokenize(str(text), engine='newmm')

# ----------------------------------------------------
# 1. Train Fake News Model ด้วย Dataset จาก Hugging Face
# ----------------------------------------------------
print("⏳ กำลังโหลด Dataset ข่าวจริง/ข่าวปลอมภาษาไทยจาก Hugging Face...")
# โหลดข้อมูล EXt1/Thai-True-Fake-News
dataset = load_dataset("EXt1/Thai-True-Fake-News")
df_fake = pd.DataFrame(dataset['train'])

# ตรวจสอบชื่อคอลัมน์ (ส่วนใหญ่จะเป็น text/content และ label/target)
# สมมติคอลัมน์ข้อความคือ 'text' และเลเบลคือ 'label' (1=Fake, 0=True)
X_fake = df_fake['text']
y_fake = df_fake['label']

print(f"📊 โหลดข้อมูลข่าวทั้งหมด {len(df_fake)} รายการ สั่งเริ่มเทรนโมเดล...")

fake_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=thai_tokenizer, token_pattern=None, max_features=10000)),
    ('clf', LogisticRegression(max_iter=1000))
])

fake_pipeline.fit(X_fake, y_fake)
joblib.dump(fake_pipeline, FAKE_MODEL_PATH)
print("✅ เทรนและบันทึก fake_model.joblib จาก Hugging Face สำเร็จ!")


# ----------------------------------------------------
# 2. Train Spam Model (ตัวอย่างข้อมูลผสม ไทย + อังกฤษ)
# ----------------------------------------------------
spam_data = [
    ("Win a $1000 Walmart gift card now! Click here", 1),
    ("Congratulations! You have been selected for a free prize", 1),
    ("ยินดีด้วย! คุณได้รับอนุมัติวงเงินกู้ฉุกเฉิน 50,000 บาท ดอกเบี้ย 0% แอดไลน์เลย", 1),
    ("ด่วน! สิทธิ์กู้ยืมเงินด่วนอนุมัติไวใน 5 นาที ถอนเงินได้ทันที คลิก http://bit.ly/claim", 1),
    ("เว็บตรงไม่ผ่านเอเย่นต์ สมัครวันนี้รับโบนัสฟรี 300% ฝากถอนออโต้", 1),
    ("Please review the meeting minutes from yesterday", 0),
    ("Are we still meeting for lunch today at 12?", 0),
    ("ขอส่งสรุปผลการประชุมประจำสัปดาห์ และกำหนดการส่งมอบงานครับ", 0),
    ("เรียนทุกท่าน รบกวนตรวจสอบเอกสารแนบสำหรับวาระการประชุมวันพรุ่งนี้ครับ", 0),
    ("แจ้งเตือนยอดชำระค่าบริการอินเทอร์เน็ตประจำเดือน สามารถชำระผ่านแอปพลิเคชันธนาคารได้", 0)
]
X_spam, y_spam = zip(*spam_data)

spam_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=thai_tokenizer, token_pattern=None)),
    ('clf', LogisticRegression())
])
spam_pipeline.fit(X_spam, y_spam)
joblib.dump(spam_pipeline, SPAM_MODEL_PATH)
print("✅ บันทึก spam_model.joblib สำเร็จ!")
