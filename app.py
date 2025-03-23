import re
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained phishing detection model
try:
    model = joblib.load("phishing_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure 'phishing_model.pkl' and 'vectorizer.pkl' exist.")
    model, vectorizer = None, None  # Prevent crashes

# Define phishing red flags
RED_FLAGS = {
    "urgency": r"(urgent|immediately|action required|account will be closed|verify now|limited time offer)",
    "suspicious_links": r"https?://[^\s]+",  # Detects all links
    "spelling_errors": r"\b(pleese|clik|recieve|updat|verif|confrim)\b",
    "sender_mismatch": r"(@gmail\.com|@yahoo\.com|@hotmail\.com)",
    "fake_login_attempt": r"(unusual activity|suspicious login attempt|your account has been compromised)",
    "request_for_personal_info": r"(confirm your password|enter your credit card details|provide your SSN)",
    "financial_fraud": r"(you have won|congratulations|lottery|free gift|claim your prize|money transfer request)",
    "threatening_language": r"(legal action will be taken|lawsuit|arrest warrant|blacklisted|FBI warning|IRS audit)",
    "unusual_attachment": r"(\.exe|\.zip|\.rar|\.scr|\.js|\.bat|\.vbs|\.ps1)",
    "spoofed_sender": r"(support@|helpdesk@|billing@|security@).*(gmail|yahoo|outlook)\.com",
    "unknown_sender": r"(dear customer|dear user|valued client)",
}

def detect_red_flags(email_text):
    """Identify phishing red flags in the email text."""
    red_flag_list = []
    for flag, pattern in RED_FLAGS.items():
        if re.search(pattern, email_text, re.IGNORECASE):
            red_flag_list.append(flag)
    return red_flag_list

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    red_flags = []
    email_text = ""

    if request.method == "POST":
        email_text = request.form["email_text"]

        if model is None or vectorizer is None:
            result = "Error: Model is not loaded properly."
        else:
            try:
                # Convert text into features
                email_features = vectorizer.transform([email_text])
                prediction = model.predict(email_features)[0]

                # Identify red flags
                red_flags = detect_red_flags(email_text)

                result = "🚨 Phishing Email Detected!" if prediction == 1 else "✅ Legitimate Email"
            except Exception as e:
                result = f"Error: {str(e)}"

    return render_template("index.html", result=result, red_flags=red_flags, email_text=email_text)

if __name__ == "__main__":
    app.run(debug=True)
