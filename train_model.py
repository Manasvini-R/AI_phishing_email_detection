import pandas as pd
import numpy as np
import joblib
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  # Fastest model
from sklearn.linear_model import LogisticRegression  # More accurate
from sklearn.metrics import accuracy_score

# ⏳ Start timer to measure training time
start_time = time.time()

# 📂 Load dataset
file_path = "emails.csv"
df = pd.read_csv(file_path)

# 🔍 Check for missing values
df.dropna(inplace=True)

# 🏷 Define features and labels
X = df["Email Content"]  # Email text
y = df["Label"]  # Phishing (1) or Legitimate (0)

# 🏎 Optimize training by using only 50,000 samples if dataset is too large
if len(df) > 50000:
    df = df.sample(n=50000, random_state=42)

# ✨ Convert text data to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# 🎯 Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 🚀 Train Model (Naïve Bayes for speed, Logistic Regression for accuracy)
model = MultinomialNB()  # Change to LogisticRegression() for better accuracy
model.fit(X_train, y_train)

# 🎯 Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Training Completed! Accuracy: {accuracy:.2%}")

# 💾 Save Model and Vectorizer
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# ⏳ Display total training time
end_time = time.time()
print(f"⏳ Training Time: {end_time - start_time:.2f} seconds")
