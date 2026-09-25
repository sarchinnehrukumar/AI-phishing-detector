import streamlit as st
import joblib

# Load trained artifacts
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
threshold = joblib.load("threshold.pkl")

# Page configuration
st.set_page_config(
    page_title="AI Phishing Email Detector",
    page_icon="🛡️"
)

st.title("🛡️ AI Phishing Email Detector")

st.write(
    "Paste an email below to analyze whether it appears "
    "legitimate or suspicious."
)

email_text = st.text_area(
    "Email content",
    height=250,
    placeholder="Paste the email here..."
)

if st.button("Analyze Email"):

    if email_text.strip() == "":
        st.warning("Please enter an email first.")

    else:

        # Convert email into TF-IDF features
        email_vector = vectorizer.transform([email_text])

        # Get suspicious-class probability
        probability = model.predict_proba(email_vector)[0][1]

        risk_percentage = probability * 100

        # Classification
        if probability >= threshold:
            st.error("🚨 SUSPICIOUS EMAIL")
        else:
            st.success("✅ LEGITIMATE EMAIL")

        # Display score
        st.metric(
            "Suspicious Email Score",
            f"{risk_percentage:.2f}%"
        )

        # Visual progress bar
        st.progress(float(probability))

        # Simple score level
        if probability < threshold:
            risk_level = "Low"
        elif probability < 0.75:
            risk_level = "Elevated"
        else:
            risk_level = "High"

        st.write("Risk Level:", risk_level)

        st.caption(
            "The score represents the model's prediction, "
            "not a guaranteed real-world probability of phishing."
        )