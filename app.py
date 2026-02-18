import os
import numpy as np
import joblib
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="TrustPilot Review Analyzer",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
MODELS_DIR = "models"
CLASS_TO_SENT = {0: -1, 1: 0, 2: 1}
SENTIMENT_LABELS = {
    -1: ("Negative", "❌", "red"),
    0: ("Neutral", "⚪", "grey"),
    1: ("Positive", "✅", "green")
}

# --- Load Models (Cached) ---
@st.cache_resource
def load_models():
    try:
        themes_clf = joblib.load(os.path.join(MODELS_DIR, "themes_clf.joblib"))
        thresholds = np.load(os.path.join(MODELS_DIR, "themes_thresholds.npy"))
        sent_liv = joblib.load(os.path.join(MODELS_DIR, "sent_livraison.joblib"))
        sent_sav = joblib.load(os.path.join(MODELS_DIR, "sent_sav.joblib"))
        sent_prod = joblib.load(os.path.join(MODELS_DIR, "sent_produit.joblib"))
        return themes_clf, thresholds, sent_liv, sent_sav, sent_prod
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the 'models' directory exists.")
        return None, None, None, None, None

themes_clf, thresholds, sent_liv, sent_sav, sent_prod = load_models()

# --- Helper Functions ---
def predict_sentiment(model, text):
    return CLASS_TO_SENT[int(model.predict([text])[0])]

def analyze_review(text):
    if not themes_clf:
        return None

    proba = themes_clf.predict_proba([text])[0]
    pred = (proba >= thresholds).astype(int)
    
    results = {
        "delivery": {
            "detected": bool(pred[0]),
            "probability": float(proba[0]),
            "sentiment": None
        },
        "customer_service": {
            "detected": bool(pred[1]),
            "probability": float(proba[1]),
            "sentiment": None
        },
        "product": {
            "detected": bool(pred[2]),
            "probability": float(proba[2]),
            "sentiment": None
        }
    }

    if results["delivery"]["detected"]:
        results["delivery"]["sentiment"] = predict_sentiment(sent_liv, text)
    
    if results["customer_service"]["detected"]:
        results["customer_service"]["sentiment"] = predict_sentiment(sent_sav, text)
        
    if results["product"]["detected"]:
        results["product"]["sentiment"] = predict_sentiment(sent_prod, text)
        
    return results

def display_sentiment(sentiment_val):
    if sentiment_val is None:
        return "—"
    label, icon, color = SENTIMENT_LABELS.get(sentiment_val, ("Unknown", "❓", "black"))
    return f":{color}[{icon} {label}]"

# --- UI Layout ---

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This POC analyzes TrustPilot reviews to identify themes and sentiment.
    
    **Themes Detected:**
    - 🚚 Delivery
    - ☎️ Customer Service
    - 📦 Product
    
    **Models:**
    - TF-IDF + Logistic Regression
    """)
    
    st.markdown("---")
    st.subheader("Example Reviews")
    example_reviews = [
        "Delivery was super fast but the product broke after one day.",
        "Customer service was rude and unhelpful. Never buying again.",
        "Excellent quality! I love it. Shipping took a while though.",
        "Just okay. Nothing special."
    ]
    
    selected_example = st.selectbox("Try an example:", ["Select..."] + example_reviews)

# Main Content
st.title("⭐ TrustPilot Review Analyzer")
st.markdown("Analyze customer feedback for themes and sentiment automatically.")

# Input Area
input_text = st.text_area(
    "Paste a customer review:",
    value=selected_example if selected_example != "Select..." else "",
    height=150,
    placeholder="e.g., The delivery was late, but the support team helped me refund it quickly."
)

if st.button("Analyze Review", type="primary"):
    if not input_text.strip():
        st.warning("⚠️ Please enter a review text to analyze.")
    elif themes_clf is None:
        st.error("Models are not loaded correctly.")
    else:
        with st.spinner("Analyzing..."):
            results = analyze_review(input_text)
            
        st.markdown("### Analysis Results")
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🚚 Delivery")
            data = results["delivery"]
            if data["detected"]:
                st.success(f"**Detected** ({data['probability']:.1%})")
                st.markdown(f"**Sentiment:** {display_sentiment(data['sentiment'])}")
            else:
                st.markdown(f"Not Detected ({data['probability']:.1%})")
                st.progress(data['probability'])

        with col2:
            st.markdown("#### ☎️ Customer Service")
            data = results["customer_service"]
            if data["detected"]:
                st.success(f"**Detected** ({data['probability']:.1%})")
                st.markdown(f"**Sentiment:** {display_sentiment(data['sentiment'])}")
            else:
                st.markdown(f"Not Detected ({data['probability']:.1%})")
                st.progress(data['probability'])

        with col3:
            st.markdown("#### 📦 Product")
            data = results["product"]
            if data["detected"]:
                st.success(f"**Detected** ({data['probability']:.1%})")
                st.markdown(f"**Sentiment:** {display_sentiment(data['sentiment'])}")
            else:
                st.markdown(f"Not Detected ({data['probability']:.1%})")
                st.progress(data['probability'])

        # Raw Output Expander
        with st.expander("View Raw JSON Output"):
            st.json(results)
