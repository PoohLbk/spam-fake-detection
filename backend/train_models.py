import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# หา Path ของโฟลเดอร์ backend โดยตรง
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPAM_MODEL_PATH = os.path.join(BASE_DIR, 'spam_model.joblib')
FAKE_MODEL_PATH = os.path.join(BASE_DIR, 'fake_model.joblib')

# --- [1. Train Spam Model] ---
spam_data = [
    ("Win a $1000 Walmart gift card now! Click here", 1),
    ("Congratulations! You have been selected for a free prize", 1),
    ("Please review the meeting minutes from yesterday", 0),
    ("Are we still meeting for lunch today at 12?", 0)
]
X_spam, y_spam = zip(*spam_data)

spam_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
spam_pipeline.fit(X_spam, y_spam)

# บันทึกไฟล์ลงโฟลเดอร์ backend
joblib.dump(spam_pipeline, SPAM_MODEL_PATH)
print(f"Saved spam_model.joblib to {SPAM_MODEL_PATH} successfully!")


# --- [2. Train Fake News Model] ---
fake_data = [
    ("Scientists discover drinking green tea turns humans into reptiles", 1),
    ("Shocking leak reveals secret moon base built by ancient aliens", 1),
    ("The central bank announced a new policy regarding interest rates today", 0),
    ("Local weather forecast predicts mild rain over the weekend", 0)
]
X_fake, y_fake = zip(*fake_data)

fake_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
fake_pipeline.fit(X_fake, y_fake)

# บันทึกไฟล์ลงโฟลเดอร์ backend
joblib.dump(fake_pipeline, FAKE_MODEL_PATH)
print(f"Saved fake_model.joblib to {FAKE_MODEL_PATH} successfully!")
