import os
import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="TrustPilot POC", layout="centered")

st.title("TrustPilot Review Analyzer (POC)")
st.caption("Themes + Sentiment per Theme | TF-IDF + Logistic")

MODELS_DIR = "models"

themes_clf = joblib.load(os.path.join(MODELS_DIR, "themes_clf.joblib"))
thresholds = np.load(os.path.join(MODELS_DIR, "themes_thresholds.npy"))

sent_liv = joblib.load(os.path.join(MODELS_DIR, "sent_livraison.joblib"))
sent_sav = joblib.load(os.path.join(MODELS_DIR, "sent_sav.joblib"))
sent_prod = joblib.load(os.path.join(MODELS_DIR, "sent_produit.joblib"))

CLASS_TO_SENT = {0: -1, 1: 0, 2: 1}

def pretty_sent(s):
    if s is None:
        return "—"
    if s == 1:
        return "✅ Positive"
    if s == 0:
        return "⚪ Neutral"
    return "❌ Negative"

def predict(text: str):
    proba = themes_clf.predict_proba([text])[0]
    pred = (proba >= thresholds).astype(int)

    out = {
        "delivery_theme": int(pred[0]),
        "customer_service_theme": int(pred[1]),
        "product_theme": int(pred[2]),
        "p_delivery": float(proba[0]),
        "p_customer_service": float(proba[1]),
        "p_product": float(proba[2]),
        "delivery_sentiment": None,
        "customer_service_sentiment": None,
        "product_sentiment": None,
    }

    if out["delivery_theme"] == 1:
        out["delivery_sentiment"] = CLASS_TO_SENT[int(sent_liv.predict([text])[0])]
    if out["customer_service_theme"] == 1:
        out["customer_service_sentiment"] = CLASS_TO_SENT[int(sent_sav.predict([text])[0])]
    if out["product_theme"] == 1:
        out["product_sentiment"] = CLASS_TO_SENT[int(sent_prod.predict([text])[0])]

    return out

text = st.text_area(
    "Paste a review:",
    height=160,
    placeholder="Delivery was late, customer support was rude, but the product works great."
)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter a review.")
    else:
        res = predict(text)

        st.subheader("Themes")
        st.write(f"🚚 Delivery: **{res['delivery_theme']}** (p={res['p_delivery']:.2f})")
        st.write(f"☎️ Customer Service: **{res['customer_service_theme']}** (p={res['p_customer_service']:.2f})")
        st.write(f"📦 Product: **{res['product_theme']}** (p={res['p_product']:.2f})")

        st.subheader("Sentiment per Theme")
        st.write(f"🚚 Delivery sentiment: **{pretty_sent(res['delivery_sentiment'])}**")
        st.write(f"☎️ Customer Service sentiment: **{pretty_sent(res['customer_service_sentiment'])}**")
        st.write(f"📦 Product sentiment: **{pretty_sent(res['product_sentiment'])}**")

        st.subheader("Raw Output")
        st.json(res)
