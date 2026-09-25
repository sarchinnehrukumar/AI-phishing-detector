# 🛡️ AI Phishing & Suspicious Email Detector

A machine-learning web application that analyzes email text and classifies it as **legitimate or suspicious**.

## 🌐 Live Demo

https://sarchin-ai-phishing-detector.streamlit.app

## 📌 Project Overview

This project uses Natural Language Processing and machine learning to identify suspicious, spam, and phishing-style email content.

The application:

- Accepts email text from the user
- Converts the text into TF-IDF features
- Uses a Logistic Regression classifier
- Generates a suspicious-email score
- Applies a security-focused classification threshold
- Shows important features that contributed to the suspicious prediction

## 🧠 Machine Learning Pipeline

Email Text  
↓  
TF-IDF Vectorization  
↓  
Unigrams + Bigrams  
↓  
Logistic Regression  
↓  
Suspicious-Class Score  
↓  
Optimized Threshold  
↓  
Legitimate / Suspicious Classification

## 📊 Dataset

The project uses the Kaggle Phishing Email Dataset.

After duplicate removal, the dataset contained approximately:

**82,078 emails**

Classes:

- `0` = Legitimate
- `1` = Suspicious / Spam / Phishing-type email

## 🤖 Model

The final model uses:

- Word-level TF-IDF
- Unigrams and bigrams
- Logistic Regression
- Precision-recall based threshold optimization

### TF-IDF configuration

```python
TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=50000,
    min_df=2,
    stop_words="english",
    sublinear_tf=True
)
📈 Model Performance

At the default threshold of 0.50:

False Positives: 93
False Negatives: 84

A security-focused threshold of approximately:

0.3775

was selected to reduce missed suspicious emails.

At this threshold:

Precision: ~97.62%
Recall: ~99.52%
False Positives: 206
False Negatives: 41

The lower threshold prioritizes detecting suspicious emails while accepting additional false alarms.

🔍 Model Explainability

The application also examines the TF-IDF features and Logistic Regression coefficients contributing to each suspicious prediction.

Example influential features may include:

account
click
verify
password
account verification

This helps explain why the model classified an email as suspicious.

🛠️ Technologies Used
Python
Pandas
Scikit-learn
TF-IDF
Logistic Regression
Joblib
Streamlit
Git
GitHub
Streamlit Community Cloud
💻 Run Locally

Clone the repository and enter the project folder.

Create a virtual environment:

python3 -m venv .venv

Activate it on macOS:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run the application:

python -m streamlit run app.py

Then open:

http://localhost:8501

⚠️ Disclaimer

The suspicious-email score is a machine-learning model output and should not be interpreted as a guaranteed real-world probability that an email is phishing.

The training dataset contains both spam and phishing-style content, so this project is best described as a suspicious-email detection system rather than a pure phishing detector.