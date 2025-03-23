import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

# Load AI Model & Vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

nltk.download("stopwords")

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = " ".join(word for word in text.split() if word not in stopwords.words('english'))  # Remove stopwords
    return text

def is_phishing_email(email_text):
    """Predicts if an email is phishing using the AI model."""
    processed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(vectorized_text)
    return bool(prediction[0])  # Returns True if phishing, False otherwise
