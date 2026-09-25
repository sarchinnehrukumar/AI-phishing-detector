import streamlit as st
import joblib

# -----------------------------
# Load saved ML components
# -----------------------------

model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
threshold = joblib.load("threshold.pkl")


# -----------------------------
# Explainability function
# -----------------------------

def get_top_suspicious_features(
    email_vector,
    vectorizer,
    model,
    top_n=5
):

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    email_vector = email_vector.tocoo()

    contributions = []

    for feature_index, tfidf_value in zip(
        email_vector.col,
        email_vector.data
    ):

        contribution = (
            tfidf_value * coefficients[feature_index]
        )

        # Positive contribution pushes toward class 1
        if contribution > 0:

            contributions.append(
                (
                    feature_names[feature_index],
                    contribution
                )
            )

    # Highest contribution first
    contributions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return contributions[:top_n]


# -----------------------------
# Streamlit page
# -----------------------------

st.set_page_config(
    page_title="AI Phishing Email Detector",
    page_icon="🛡️"
)

st.title("🛡️ AI Phishing Email Detector")

st.write(
    "Paste an email below to analyze whether it appears "
    "legitimate or suspicious."
)


# -----------------------------
# User input
# -----------------------------

email_text = st.text_area(
    "Email content",
    height=250,
    placeholder="Paste the email here..."
)
top_features = []

# -----------------------------
# Analyze button
# -----------------------------

if st.button("Analyze Email"):

    if email_text.strip() == "":

        st.warning("Please enter an email first.")

    else:

        # Convert email into TF-IDF features
        email_vector = vectorizer.transform(
            [email_text]
        )

        # Get probability of suspicious class
        probability = model.predict_proba(
            email_vector
        )[0][1]

        risk_percentage = probability * 100


        # -----------------------------
        # Classification
        # -----------------------------

        if probability >= threshold:

            st.error("🚨 SUSPICIOUS EMAIL")

        else:

            st.success("✅ LEGITIMATE EMAIL")


        # -----------------------------
        # Risk score
        # -----------------------------

        st.metric(
            "Suspicious Email Score",
            f"{risk_percentage:.2f}%"
        )

        st.progress(
            float(probability)
        )


        # -----------------------------
        # Risk level
        # -----------------------------

        if probability < threshold:

            risk_level = "Low"

        elif probability < 0.75:

            risk_level = "Elevated"

        else:

            risk_level = "High"

        st.write(
            "Risk Level:",
            risk_level
        )


        # -----------------------------
        # Explainability
        # -----------------------------

        top_features = get_top_suspicious_features(
            email_vector,
            vectorizer,
            model
        )

        if top_features:

            st.subheader(
                "Features contributing to the suspicious score"
            )

            for feature, contribution in top_features:

                st.write(
                    f"• {feature}"
                )


        # -----------------------------
        # Disclaimer
        # -----------------------------

        st.caption(
            "The score represents the model's prediction, "
            "not a guaranteed real-world probability of phishing."
        )