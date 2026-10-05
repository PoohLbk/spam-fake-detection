import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAKE_MODEL_PATH = os.path.join(BASE_DIR, 'fake_model.joblib')

def train_fake_news_model():
    print("⏳ กำลังโหลดและเทรน Fake News Dataset...")
    
    # สามารถใส่ URL หรือ Path ของไฟล์ CSV จาก Kaggle/GitHub ได้
    # หรือถ้ามีไฟล์ News.csv อยู่ในโฟลเดอร์ data/
    data_path = os.path.join(BASE_DIR, '..', 'data', 'News.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # กรณีไม่มีไฟล์โลคัล สามารถดึงจาก Public URL ได้
        url = "https://raw.githubusercontent.com/project-datasets/fake-news/main/News.csv"
        df = pd.read_csv(url)

    # ทำความสะอาดข้อมูลเบื้องต้น
    df = df.dropna(subset=['text', 'class'])
    
    X = df['text']
    y = df['class']

    # แบ่ง Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # สร้างและเทรน Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # บันทึกโมเดล
    joblib.dump(pipeline, FAKE_MODEL_PATH)
    print(f"✅ เทรน Fake News Model สำเร็จ! เซฟไว้ที่: {FAKE_MODEL_PATH}")

if __name__ == "__main__":
    train_fake_news_model()
