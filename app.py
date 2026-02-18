import streamlit as st
import time
from transformers import pipeline

# --- Configuration ---
st.set_page_config(
    page_title="TrustPilot Review Analyzer",
    page_icon="⭐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        text-align: center;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Result Cards */
    .result-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        flex-wrap: wrap;
        margin-top: 2rem;
    }
    
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        width: 100%;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #eee;
        transition: transform 0.2s;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 15px;
        border-bottom: 1px solid #f0f0f0;
        padding-bottom: 10px;
    }
    
    .card-icon {
        font-size: 1.5rem;
    }
    
    .card-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: #333;
        margin: 0;
    }
    
    .sentiment-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        text-align: center;
        width: 100%;
    }
    
    .badge-positive { background-color: #e6f4ea; color: #1e7e34; }
    .badge-negative { background-color: #ffebee; color: #c62828; }
    .badge-neutral { background-color: #f5f5f5; color: #616161; }
    .badge-none { background-color: #ffffff; color: #ccc; border: 1px dashed #ddd; }
    
    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        background-color: #f0f0f0;
        margin-top: 10px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background-color: #333;
        border-radius: 3px;
    }

    /* Input Area */
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 15px;
        font-size: 1rem;
    }
    
    .stButton button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: 600;
        background-color: #00b67a; /* TrustPilot Green */
        color: white;
        border: none;
    }
    
    .stButton button:hover {
        background-color: #00a06b;
        color: white;
    }
    
</style>
""", unsafe_allow_html=True)

# --- Load Optimized Model ---
# Switching to a more specialized model for review sentiment if possible,
# but XLM-Roberta is generally best for Zero-Shot.
# Let's try refining the prompt/labels to be more explicit.
@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Using BART-Large-MNLI which is often sharper for English/General logic than XLM-R (though XLM-R is better for multilingual).
# If the user is testing in English primarily, BART might give better results.
# Let's stick to BART for better English performance, or keep XLM-R if multilingual is key.
# User said "bad predictions", let's try a different hypothesis template.

classifier = load_model()

# --- Analysis Logic ---
def analyze(text):
    themes = {
        "Delivery": {
            "icon": "🚚",
            "labels": ["fast delivery", "slow delivery", "delivery issue", "no delivery mention"],
            "map": {
                "fast delivery": ("Positive", "badge-positive"),
                "slow delivery": ("Negative", "badge-negative"),
                "delivery issue": ("Negative", "badge-negative"),
                "no delivery mention": ("Not Mentioned", "badge-none")
            }
        },
        "Customer Service": {
            "icon": "☎️",
            "labels": ["helpful support", "rude support", "no support mention"],
            "map": {
                "helpful support": ("Positive", "badge-positive"),
                "rude support": ("Negative", "badge-negative"),
                "no support mention": ("Not Mentioned", "badge-none")
            }
        },
        "Product": {
            "icon": "📦",
            "labels": ["great product", "bad product", "average product", "no product mention"],
            "map": {
                "great product": ("Positive", "badge-positive"),
                "bad product": ("Negative", "badge-negative"),
                "average product": ("Neutral", "badge-neutral"),
                "no product mention": ("Not Mentioned", "badge-none")
            }
        }
    }
    
    results = {}
    
    for theme, config in themes.items():
        # Hypothesis Template is crucial for Zero-Shot accuracy
        out = classifier(
            text, 
            config["labels"], 
            hypothesis_template="This review is about {}.", 
            multi_label=False
        )
        
        top_label = out["labels"][0]
        score = out["scores"][0]
        
        # Threshold check: if even the top label is low confidence, assume not mentioned
        # But "no mention" label usually handles this better.
        
        sent_label, badge_class = config["map"].get(top_label, ("Neutral", "badge-neutral"))
        
        results[theme] = {
            "icon": config["icon"],
            "sentiment": sent_label,
            "badge": badge_class,
            "score": score,
            "raw_label": top_label
        }
        
    return results

# --- UI Layout ---
st.markdown("<h1 class='main-title'>TrustPilot Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Instant feedback analysis powered by AI</p>", unsafe_allow_html=True)

# Input
review_text = st.text_area(
    "Review Text",
    placeholder="e.g. The shipping was super fast but the product quality is terrible.",
    height=120,
    label_visibility="collapsed"
)

# Analyze Button
if st.button("Analyze Review"):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Processing..."):
            res = analyze(review_text)
            
        # Display Results Grid
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        
        # Columns for cards
        c1, c2, c3 = st.columns(3)
        
        cols = [c1, c2, c3]
        keys = list(res.keys())
        
        for i, col in enumerate(cols):
            theme = keys[i]
            data = res[theme]
            
            with col:
                st.markdown(f"""
                <div class="result-card">
                    <div class="card-header">
                        <span class="card-icon">{data['icon']}</span>
                        <h3 class="card-title">{theme}</h3>
                    </div>
                    <div class="sentiment-badge {data['badge']}">
                        {data['sentiment']}
                    </div>
                    <div style="margin-top: 10px; font-size: 0.8rem; color: #888; text-align: right;">
                        Confidence: {data['score']:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
with st.expander("ℹ️ How it works"):
    st.write("This tool uses a BART-Large-MNLI Zero-Shot classifier to detect specific sentiments for Delivery, Support, and Product independently.")
