import streamlit as st
from transformers import pipeline
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
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .sentiment-icon { font-size: 1.5rem; margin-right: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Multilingual Configuration ---
CONFIG = {
    "English": {
        "title": "TrustPilot Aspect-Based Sentiment Analysis",
        "subtitle": "Detects sentiment (Positive/Negative/Neutral) SPECIFICALLY for each theme.",
        "input_label": "Paste a customer review:",
        "analyze_btn": "Analyze Review",
        "loading": "Loading Multilingual AI Model (XLM-Roberta)...",
        "analyzing": "Analyzing aspects...",
        "aspects": {
            "Delivery": {
                "labels": ["satisfied with delivery", "dissatisfied with delivery", "neutral about delivery", "delivery not mentioned"],
                "mapping": {
                    "satisfied with delivery": ("Positive", "✅", "green"),
                    "dissatisfied with delivery": ("Negative", "❌", "red"),
                    "neutral about delivery": ("Neutral", "⚪", "grey"),
                    "delivery not mentioned": ("Not Detected", "➖", "lightgrey")
                }
            },
            "Customer Service": {
                "labels": ["satisfied with customer service", "dissatisfied with customer service", "neutral about customer service", "customer service not mentioned"],
                "mapping": {
                    "satisfied with customer service": ("Positive", "✅", "green"),
                    "dissatisfied with customer service": ("Negative", "❌", "red"),
                    "neutral about customer service": ("Neutral", "⚪", "grey"),
                    "customer service not mentioned": ("Not Detected", "➖", "lightgrey")
                }
            },
            "Product": {
                "labels": ["satisfied with product", "dissatisfied with product", "neutral about product", "product not mentioned"],
                "mapping": {
                    "satisfied with product": ("Positive", "✅", "green"),
                    "dissatisfied with product": ("Negative", "❌", "red"),
                    "neutral about product": ("Neutral", "⚪", "grey"),
                    "product not mentioned": ("Not Detected", "➖", "lightgrey")
                }
            }
        },
        "examples": [
            "The delivery was incredibly fast, but the product stopped working after two days.",
            "Customer service was rude, but the item itself is great quality.",
            "I haven't received my package yet, and support isn't replying.",
            "It's okay, nothing special."
        ]
    },
    "Français": {
        "title": "Analyse de Sentiment par Aspect (ABSA)",
        "subtitle": "Détecte le sentiment (Positif/Négatif/Neutre) SPÉCIFIQUEMENT pour chaque thème.",
        "input_label": "Collez un avis client :",
        "analyze_btn": "Analyser l'avis",
        "loading": "Chargement du modèle multilingue IA (XLM-Roberta)...",
        "analyzing": "Analyse des aspects...",
        "aspects": {
            "Livraison": {
                "labels": ["satisfait de la livraison", "insatisfait de la livraison", "neutre sur la livraison", "livraison non mentionnée"],
                "mapping": {
                    "satisfait de la livraison": ("Positif", "✅", "green"),
                    "insatisfait de la livraison": ("Négatif", "❌", "red"),
                    "neutre sur la livraison": ("Neutre", "⚪", "grey"),
                    "livraison non mentionnée": ("Non Détecté", "➖", "lightgrey")
                }
            },
            "Service Client": {
                "labels": ["satisfait du service client", "insatisfait du service client", "neutre sur le service client", "service client non mentionné"],
                "mapping": {
                    "satisfait du service client": ("Positif", "✅", "green"),
                    "insatisfait du service client": ("Négatif", "❌", "red"),
                    "neutre sur le service client": ("Neutre", "⚪", "grey"),
                    "service client non mentionné": ("Non Détecté", "➖", "lightgrey")
                }
            },
            "Produit": {
                "labels": ["satisfait du produit", "insatisfait du produit", "neutre sur le produit", "produit non mentionné"],
                "mapping": {
                    "satisfait du produit": ("Positif", "✅", "green"),
                    "insatisfait du produit": ("Négatif", "❌", "red"),
                    "neutre sur le produit": ("Neutre", "⚪", "grey"),
                    "produit non mentionné": ("Non Détecté", "➖", "lightgrey")
                }
            }
        },
        "examples": [
            "La livraison a été super rapide, mais le produit est tombé en panne après deux jours.",
            "Le service client était désagréable, mais l'article est de très bonne qualité.",
            "Je n'ai pas encore reçu mon colis et le support ne répond pas.",
            "C'est correct, sans plus."
        ]
    }
}

# --- Sidebar ---
st.sidebar.title("Configuration")
selected_lang_code = st.sidebar.radio("Language / Langue", ["English", "Français"])
lang_config = CONFIG[selected_lang_code]

# --- Load Model (Cached) ---
@st.cache_resource
def load_model():
    # using XLM-Roberta-Large-XNLI for robust multilingual zero-shot classification
    return pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

with st.spinner(lang_config["loading"]):
    try:
        classifier = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- Analysis Function ---
def analyze_aspects(text, config):
    results = {}
    
    # We iterate over each aspect (Delivery, SAV, Product)
    # And run a specific zero-shot classification for that aspect alone
    for aspect_name, aspect_data in config["aspects"].items():
        candidate_labels = aspect_data["labels"]
        
        # Run classification
        # Hypothesis template is not strictly needed for XLM-R XNLI but can help. 
        # However, the pipeline handles it. We pass the raw labels which are descriptive sentences.
        output = classifier(text, candidate_labels=candidate_labels, multi_label=False)
        
        # Get top prediction
        top_label = output["labels"][0]
        top_score = output["scores"][0]
        
        # Map to UI elements
        sentiment_text, icon, color = aspect_data["mapping"][top_label]
        
        results[aspect_name] = {
            "label": top_label,
            "score": top_score,
            "sentiment": sentiment_text,
            "icon": icon,
            "color": color,
            "is_detected": "not mentioned" not in top_label and "non mentionnée" not in top_label and "non mentionné" not in top_label
        }
        
    return results

# --- Main UI ---
st.title(lang_config["title"])
st.markdown(f"*{lang_config['subtitle']}*")

# Input
col1, col2 = st.columns([2, 1])
with col1:
    selected_example = st.selectbox("Example / Exemple:", ["Select..."] + lang_config["examples"])
    input_text = st.text_area(
        lang_config["input_label"],
        value=selected_example if selected_example != "Select..." else "",
        height=150
    )

with col2:
    st.info("""
    **Logic Explained:**
    Instead of checking global sentiment, we ask the AI 3 specific questions:
    1. How does the user feel about **Delivery**?
    2. How does the user feel about **Service**?
    3. How does the user feel about the **Product**?
    """)

if st.button(lang_config["analyze_btn"], type="primary"):
    if not input_text.strip():
        st.warning("⚠️ Please enter text.")
    else:
        with st.spinner(lang_config["analyzing"]):
            start_time = time.time()
            results = analyze_aspects(input_text, lang_config)
            end_time = time.time()
            
        st.markdown(f"### Results <small>({end_time - start_time:.2f}s)</small>", unsafe_allow_html=True)
        
        # Display Cards
        cols = st.columns(3)
        
        for idx, (aspect_name, data) in enumerate(results.items()):
            with cols[idx]:
                # Card Styling
                opacity = "1.0" if data["is_detected"] else "0.5"
                border = f"2px solid {data['color']}" if data["is_detected"] else "1px dashed grey"
                
                st.markdown(f"""
                <div style="
                    background-color: #ffffff;
                    padding: 20px;
                    border-radius: 10px;
                    border: {border};
                    opacity: {opacity};
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <h3 style="margin:0; color: #333;">{aspect_name}</h3>
                    <hr style="margin: 10px 0;">
                    <div style="font-size: 2rem; margin: 10px 0;">{data['icon']}</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: {data['color']}">{data['sentiment']}</div>
                    <div style="font-size: 0.8rem; color: #888; margin-top: 5px;">Confidence: {data['score']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

        # Raw Data
        with st.expander("Debug / JSON"):
            st.json(results)
