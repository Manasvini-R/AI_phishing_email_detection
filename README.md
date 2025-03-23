# AI_phishing_email_detection
Detects phishing email using ML

This project is an AI-powered phishing email detection system that classifies emails as phishing or legitimate using a trained machine learning model. It includes a Flask web app for real-time email analysis and a command-line utility for quick phishing checks.

🚀 Features  
🛡 AI-Powered Detection – Uses a trained Naïve Bayes model with TF-IDF vectorization.  
🔍 Red Flag Analysis – Detects suspicious patterns like urgent requests, fake login alerts, and financial fraud attempts.  
🌐 Flask Web Interface – Provides a user-friendly interface for email verification.  
⚡ Fast Predictions – Processes and classifies emails in real time.  
📁 Trained on Real Data – Uses a labeled dataset of phishing and legitimate emails.  

🔬 How It Works  
Extracts features from email text using TF-IDF vectorization.  
Detects phishing indicators like urgency, fake links, and financial fraud.  
Predicts whether the email is phishing using a trained Naïve Bayes model.  

🛡 Security Considerations  
Prevents false positives by analyzing multiple factors.  
Uses preprocessing techniques to filter unnecessary text.  
Model can be retrained with updated datasets.  

