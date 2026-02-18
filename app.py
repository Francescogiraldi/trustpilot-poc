import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="TrustPilot AI Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; font-weight: 700; }
    .sub-header { font-size: 1.5rem; color: #4a4a4a; margin-top: 20px; }
    .stProgress > div > div > div > div { background-color: #1f77b4; }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .sentiment-positive { color: #28a745; font-weight: bold; }
    .sentiment-neutral { color: #6c757d; font-weight: bold; }
    .sentiment-negative { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Multilingual Support ---
LANGUAGES = {
    "English": {
        "title": "TrustPilot Review AI Analyzer",
        "subtitle": "Advanced Theme & Sentiment Analysis using Transformers",
        "input_label": "Paste a customer review:",
        "analyze_btn": "Analyze Review",
        "themes": ["Delivery", "Customer Service", "Product"],
        "sentiment_labels": {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"},
        "loading": "Loading AI Models... (This may take a minute on first run)",
        "analyzing": "Analyzing...",
        "results_title": "Analysis Results",
        "confidence": "Confidence",
        "sentiment": "Sentiment",
        "raw_json": "View Raw JSON Output",
        "about_title": "About",
        "about_text": """
        This application uses state-of-the-art **Transformer models** (Zero-Shot Classification & Sentiment Analysis) 
        to analyze customer feedback.
        
        **Models Used:**
        - **Themes:** `facebook/bart-large-mnli` (Zero-Shot)
        - **Sentiment:** `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
        """,
        "examples": [
            "The delivery was incredibly fast, but the product stopped working after two days.",
            "Customer service was rude and unhelpful. I will never buy from here again.",
            "Amazing quality! The item is perfect. Shipping took a bit long though.",
            "It's okay, nothing special."
        ]
    },
    "Français": {
        "title": "Analyseur IA TrustPilot",
        "subtitle": "Analyse avancée des thèmes et sentiments avec Transformers",
        "input_label": "Collez un avis client :",
        "analyze_btn": "Analyser l'avis",
        "themes": ["Livraison", "Service Client", "Produit"],
        "sentiment_labels": {"positive": "Positif", "neutral": "Neutre", "negative": "Négatif"},
        "loading": "Chargement des modèles IA... (Cela peut prendre une minute au premier lancement)",
        "analyzing": "Analyse en cours...",
        "results_title": "Résultats de l'analyse",
        "confidence": "Confiance",
        "sentiment": "Sentiment",
        "raw_json": "Voir la sortie JSON brute",
        "about_title": "À propos",
        "about_text": """
        Cette application utilise des modèles **Transformers de pointe** (Zero-Shot Classification & Sentiment Analysis) 
        pour analyser les retours clients.
        
        **Modèles utilisés :**
        - **Thèmes :** `facebook/bart-large-mnli` (Zero-Shot)
        - **Sentiment :** `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
        """,
        "examples": [
            "La livraison a été incroyablement rapide, mais le produit a cessé de fonctionner après deux jours.",
            "Le service client a été impoli et inutile. Je n'achèterai plus jamais ici.",
            "Qualité incroyable ! L'article est parfait. L'expédition a pris un peu de temps cependant.",
            "C'est correct, rien de spécial."
        ]
    }
}

# --- Sidebar Language Selector ---
st.sidebar.title("Settings / Paramètres")
selected_lang_code = st.sidebar.radio("Language / Langue", ["English", "Français"])
lang = LANGUAGES[selected_lang_code]

# --- Load Models (Cached) ---
@st.cache_resource
def load_models():
    # Zero-Shot Classification for Themes
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    # Sentiment Analysis (Multilingual)
    sentiment_analyzer = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    return classifier, sentiment_analyzer

with st.spinner(lang["loading"]):
    try:
        classifier, sentiment_analyzer = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# --- Helper Functions ---
def analyze_text(text, themes):
    # 1. Theme Detection (Zero-Shot)
    # We use multi_label=True because a review can be about multiple things (e.g., Delivery AND Product)
    theme_results = classifier(text, candidate_labels=themes, multi_label=True)
    
    # 2. Sentiment Analysis
    # We analyze the sentiment of the whole text for now. 
    # Ideally, aspect-based sentiment analysis (ABSA) would be better, but standard pipelines do whole-text.
    # To improve, we could split sentences or use a specific ABSA model, but let's stick to global sentiment per theme logic for simplicity in this POC.
    sent_result = sentiment_analyzer(text)[0]
    
    # Map sentiment label to our format
    label_map = {
        "positive": "positive",
        "neutral": "neutral",
        "negative": "negative"
    }
    # Some models return "5 stars", "1 star", etc.
    # The lxyuan model returns "positive", "negative", "neutral" usually.
    # Let's handle generic outputs.
    sentiment_label = sent_result['label'].lower()
    if "star" in sentiment_label:
        # Handle star-based models if we switch later
        pass 
        
    return {
        "themes": {
            label: score for label, score in zip(theme_results['labels'], theme_results['scores'])
        },
        "sentiment": {
            "label": sentiment_label,
            "score": sent_result['score']
        }
    }

def get_sentiment_color(label):
    if "pos" in label: return "green"
    if "neg" in label: return "red"
    return "grey"

def get_sentiment_icon(label):
    if "pos" in label: return "✅"
    if "neg" in label: return "❌"
    return "⚪"

# --- Main UI ---
st.title(lang["title"])
st.markdown(f"*{lang['subtitle']}*")

# Input Section
col1, col2 = st.columns([2, 1])

with col1:
    selected_example = st.selectbox("Try an example / Essayer un exemple:", ["Select..."] + lang["examples"])
    input_text = st.text_area(
        lang["input_label"],
        value=selected_example if selected_example != "Select..." else "",
        height=150
    )

with col2:
    st.markdown("### " + lang["about_title"])
    st.markdown(lang["about_text"])

if st.button(lang["analyze_btn"], type="primary"):
    if not input_text.strip():
        st.warning("⚠️ Please enter text.")
    else:
        with st.spinner(lang["analyzing"]):
            start_time = time.time()
            results = analyze_text(input_text, lang["themes"])
            end_time = time.time()
            
        st.markdown(f"### {lang['results_title']} <small>(Time: {end_time - start_time:.2f}s)</small>", unsafe_allow_html=True)
        
        # Display Sentiment (Global)
        sent_label = results["sentiment"]["label"]
        sent_score = results["sentiment"]["score"]
        sent_color = get_sentiment_color(sent_label)
        sent_icon = get_sentiment_icon(sent_label)
        
        st.markdown(f"""
        <div class="card">
            <h4>Overall Sentiment / Sentiment Global</h4>
            <h2 style="color: {sent_color};">{sent_icon} {sent_label.title()} <small>({sent_score:.1%})</small></h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Themes
        st.markdown("#### Detected Themes / Thèmes Détectés")
        
        # Create columns for themes
        cols = st.columns(len(lang["themes"]))
        
        # Sort themes by score to show highest first? Or keep fixed order?
        # Let's keep fixed order for consistency with UI columns.
        
        for idx, theme in enumerate(lang["themes"]):
            score = results["themes"].get(theme, 0.0)
            is_detected = score > 0.5  # Threshold
            
            with cols[idx]:
                st.markdown(f"**{theme}**")
                st.progress(score)
                if is_detected:
                    st.success(f"Detected ({score:.1%})")
                else:
                    st.caption(f"Not detected ({score:.1%})")

        # Raw JSON
        with st.expander(lang["raw_json"]):
            st.json(results)

# Footer
st.markdown("---")
st.markdown("Powered by **Hugging Face Transformers** & **Streamlit**")
