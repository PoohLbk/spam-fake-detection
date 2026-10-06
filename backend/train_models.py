import os
import joblib
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPAM_MODEL_PATH = os.path.join(BASE_DIR, 'spam_model.joblib')
FAKE_MODEL_PATH = os.path.join(BASE_DIR, 'fake_model.joblib')

# ฟังก์ชันตัดคำ
def thai_tokenizer(text):
    return word_tokenize(text, engine='newmm')

# --- 1. Spam Dataset ---
spam_data = [
    ("Win a $1000 Walmart gift card now! Click here", 1),
    ("Congratulations! You have been selected for a free prize", 1),
    ("ยินดีด้วย! คุณได้รับอนุมัติวงเงินกู้ฉุกเฉิน 50,000 บาท ดอกเบี้ย 0% แอดไลน์เลย", 1),
    ("Please review the meeting minutes from yesterday", 0),
    ("Are we still meeting for lunch today at 12?", 0),
    ("ขอส่งสรุปผลการประชุมประจำสัปดาห์ และกำหนดการส่งมอบงานครับ", 0)
]
X_spam, y_spam = zip(*spam_data)

spam_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=thai_tokenizer, token_pattern=None)),
    ('clf', LogisticRegression())
])
spam_pipeline.fit(X_spam, y_spam)
joblib.dump(spam_pipeline, SPAM_MODEL_PATH)

# --- 2. Fake News Dataset ---
fake_data = [
    ("Scientists discover drinking green tea turns humans into reptiles", 1),
    ("ด่วน! ดื่มน้ำมะนาวผสมโซดาตอนเช้าช่วยรักษาโรคมะเร็งหายขาดได้ใน 7 วัน", 1),
    ("The central bank announced a new policy regarding interest rates today", 0),
    ("กรมอุตุนิยมวิทยาออกประกาศเตือนฝนตกหนักถึงหนักมากบริเวณประเทศไทยตอนบน", 0)
]
X_fake, y_fake = zip(*fake_data)

fake_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=thai_tokenizer, token_pattern=None)),
    ('clf', LogisticRegression())
])
fake_pipeline.fit(X_fake, y_fake)
joblib.dump(fake_pipeline, FAKE_MODEL_PATH)

print("Saved updated Thai-supported models successfully!")
