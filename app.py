"""
🏥 Insurance Cost Prediction System
Advanced AI-Powered Health Insurance Cost Predictor
Made by: Ujan Pradhan & Rishav Prakash
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import joblib
import google.generativeai as genai
import warnings
from datetime import datetime
import time
import os
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Insurance Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in;
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .main-header p {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 50px;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .ai-insights {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
        margin: 1.5rem 0;
    }
    
    .ai-insights h3 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.3rem;
    }
    
    .risk-high { background: #ff6b6b; color: white; }
    .risk-medium { background: #ffd93d; color: #333; }
    .risk-low { background: #6bcf7f; color: white; }
    
    .profile-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .profile-item {
        display: flex;
        justify-content: space-between;
        padding: 0.8rem 0;
        border-bottom: 1px solid #eee;
    }
    
    .profile-label { font-weight: 600; color: #667eea; }
    .profile-value { font-weight: 500; color: #333; }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        margin-top: 3rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .footer h4 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS - FIXED WITH JOBLIB
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all models from the models/ directory"""
    
    try:
        # Check models directory
        if not os.path.exists('models'):
            st.error("❌ models/ directory not found")
            st.info(f"Current directory: {os.getcwd()}")
            return None, None, None, None, None, None, False
        
        # Load preprocessing scaler
        scaler = joblib.load("models/insurance_scaler.pkl")
        
        # Load Deep Learning model
        dl_model = tf.keras.models.load_model("models/insurance_dl_model.h5", compile=False)
        dl_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Load ML models
        rf_model = joblib.load("models/insurance_rf_model.pkl")
        xgb_model = joblib.load("models/insurance_xgb_model.pkl")
        lgb_model = joblib.load("models/insurance_lgb_model.pkl")
        
        # Load feature columns
        with open("models/feature_columns.txt", "r") as f:
            feature_cols = [line.strip() for line in f.readlines()]
        
        return scaler, dl_model, rf_model, xgb_model, lgb_model, feature_cols, True
        
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("💡 Ensure all model files are in the 'models/' directory")
        return None, None, None, None, None, None, False
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        import traceback
        with st.expander("🔍 Show detailed error"):
            st.code(traceback.format_exc())
        return None, None, None, None, None, None, False

# Load models
with st.spinner("🔄 Loading AI models..."):
    scaler, dl_model, rf_model, xgb_model, lgb_model, feature_cols, models_loaded = load_models()

# Stop if models didn't load
if not models_loaded:
    st.error("⚠️ App cannot start without models. Please check error messages above.")
    st.stop()

# ============================================================================
# CONFIGURE GEMINI AI - gemini-2.0-flash-lite
# ============================================================================

try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Use Gemini 2.0 Flash Lite (faster, cheaper, excellent quality)
        model_gemini = genai.GenerativeModel(
            'gemini-2.0-flash-lite',
            generation_config={
                'temperature': 0.7,
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
        )
        
        AI_ENABLED = True
    else:
        AI_ENABLED = False
        
except Exception as e:
    AI_ENABLED = False
    st.sidebar.warning(f"⚠️ Gemini AI unavailable: {str(e)}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(df):
    """Create features for prediction"""
    df_new = df.copy()
