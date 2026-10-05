import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. ข้อมูลตัวอย่างสำหรับ Spam Detection
spam_data = [
    ("Win a $1000 Walmart gift card now! Click here", 1),
    ("Congratulations! You have been selected for a free prize", 1),
    ("Please review the meeting minutes from yesterday", 0),
    ("Are we still meeting for lunch today at 12?", 0)
]

X_spam, y_spam = zip(*spam_data)

# สร้าง Pipeline สำหรับ Spam
spam_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
spam_pipeline.fit(X_spam, y_spam)
joblib.dump(spam_pipeline, 'spam_model.joblib')
print("Saved spam_model.joblib successfully!")

# 2. ข้อมูลตัวอย่างสำหรับ Fake News Detection
fake_data = [
    ("Scientists discover drinking green tea turns humans into reptiles", 1),
    ("Shocking leak reveals secret moon base built by ancient aliens", 1),
    ("The central bank announced a new policy regarding interest rates today", 0),
    ("Local weather forecast predicts mild rain over the weekend", 0)
]

X_fake, y_fake = zip(*fake_data)

# สร้าง Pipeline สำหรับ Fake News
fake_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
fake_pipeline.fit(X_fake, y_fake)
joblib.dump(fake_pipeline, 'fake_model.joblib')
print("Saved fake_model.joblib successfully!")
