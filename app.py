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
from pathlib import Path
warnings.filterwarnings('ignore')

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
# LOAD MODELS
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all trained models from models/ directory"""
    try:
        # Updated paths to match your structure
        dl_model = tf.keras.models.load_model('models/insurance_dl_model.h5')
        rf_model = joblib.load('models/insurance_rf_model.pkl')
        xgb_model = joblib.load('models/insurance_xgb_model.pkl')
        scaler = joblib.load('models/insurance_scaler.pkl')
        
        # Load feature columns
        with open('models/feature_columns.txt', 'r') as f:
            feature_columns = [line.strip() for line in f.readlines()]
        
        return dl_model, rf_model, xgb_model, scaler, feature_columns, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, False

with st.spinner('🔄 Loading AI models...'):
    dl_model, rf_model, xgb_model, scaler, feature_columns, models_loaded = load_models()

# Configure Gemini AI
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model_gemini = genai.GenerativeModel('gemini-2.0-flash-lite')
        AI_ENABLED = True
    else:
        AI_ENABLED = False
except:
    AI_ENABLED = False

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(df):
    """Create features for prediction"""
    df_new = df.copy()
    
    df_new['is_obese'] = (df_new['bmi'] >= 30).astype(int)
    df_new['is_smoker'] = (df_new['smoker'] == 'yes').astype(int)
    df_new['has_children'] = (df_new['children'] > 0).astype(int)
    df_new['is_elderly'] = (df_new['age'] >= 55).astype(int)
    
    df_new['smoking_bmi'] = df_new['is_smoker'] * df_new['bmi']
    df_new['age_bmi'] = df_new['age'] * df_new['bmi']
    df_new['smoking_age'] = df_new['is_smoker'] * df_new['age']
    
    df_new['age_squared'] = df_new['age'] ** 2
    df_new['bmi_squared'] = df_new['bmi'] ** 2
    
    df_new['risk_score'] = (df_new['is_smoker'] * 3 + 
                            df_new['is_obese'] * 2 + 
                            df_new['is_elderly'] * 1)
    
    df_new['sex_encoded'] = df_new['sex'].map({'male': 1, 'female': 0})
    df_new['smoker_encoded'] = df_new['is_smoker']
    
    region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    df_new['region_encoded'] = df_new['region'].map(region_map)
    
    return df_new

def predict_cost(input_data):
    """Make prediction using ensemble"""
    if not models_loaded:
        return None
    
    df = pd.DataFrame([input_data])
    df_processed = create_features(df)
    X = df_processed[feature_columns]
    X_scaled = scaler.transform(X)
    
    dl_pred = float(dl_model.predict(X_scaled, verbose=0)[0][0])
    rf_pred = float(rf_model.predict(X_scaled)[0])
    xgb_pred = float(xgb_model.predict(X_scaled)[0])
    
    ensemble_pred = (dl_pred + rf_pred + xgb_pred) / 3
    
    return {
        'Deep Learning': dl_pred,
        'Random Forest': rf_pred,
        'XGBoost': xgb_pred,
        'Ensemble': ensemble_pred
    }

def generate_ai_insights(input_data, predicted_cost):
    """Generate comprehensive AI insights"""
    
    if not AI_ENABLED:
        insights = f"""
### 🤖 AI Analysis

**Predicted Annual Premium:** ${predicted_cost:,.2f}

#### 📊 Risk Profile Assessment

"""
        if input_data['smoker'] == 'yes':
            insights += "- 🔴 **HIGH RISK:** Smoking significantly elevates costs by ~$23,600 annually (+280%)\n"
        else:
            insights += "- 🟢 **LOW RISK:** Non-smoker status contributes to lower premiums\n"
        
        if input_data['bmi'] >= 30:
            insights += "- 🟠 **ELEVATED RISK:** Obesity (BMI ≥30) adds ~$5,100 to annual premiums\n"
        elif input_data['bmi'] >= 25:
            insights += "- 🟡 **MODERATE RISK:** Overweight status may slightly increase premiums\n"
        else:
            insights += "- 🟢 **OPTIMAL:** Healthy BMI range supports lower costs\n"
        
        if input_data['age'] >= 55:
            insights += "- 🟠 **AGE FACTOR:** Advanced age (55+) increases expected healthcare costs\n"
        else:
            insights += "- 🟢 **AGE FACTOR:** Younger age bracket typically results in lower premiums\n"
        
        insights += "\n#### 💡 Recommendations\n\n"
        
        if input_data['smoker'] == 'yes':
            insights += "- **Priority:** Smoking cessation could save ~$23,000/year\n"
        if input_data['bmi'] >= 30:
            insights += "- **Health Initiative:** Weight management may reduce costs by ~$5,000/year\n"
        if input_data['smoker'] == 'no' and input_data['bmi'] < 25:
            insights += "- **Excellent Profile:** Continue maintaining healthy habits\n"
        
        return insights
    
    prompt = f"""
You are an expert insurance actuary. Provide a professional analysis for this applicant.

**PROFILE:**
- Age: {input_data['age']} years
- Sex: {input_data['sex']}
- BMI: {input_data['bmi']} ({'Obese' if input_data['bmi'] >= 30 else 'Overweight' if input_data['bmi'] >= 25 else 'Normal'})
- Children: {input_data['children']}
- Smoking: {input_data['smoker'].upper()}
- Region: {input_data['region'].capitalize()}

**PREDICTED PREMIUM:** ${predicted_cost:,.2f}

Provide:

### 🎯 Executive Summary
(2-3 sentences)

### 📊 Risk Factor Analysis
(Analyze each major factor)

### 💰 Premium Breakdown
(Explain cost components)

### 💡 Personalized Recommendations
(4-5 specific recommendations)

### 🔮 Future Outlook
(5-year projection)

Keep professional yet accessible. Use specific dollar amounts.
"""
    
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI insights temporarily unavailable. Error: {str(e)}"

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="main-header">
    <h1>🏥 AI Insurance Predictor</h1>
    <p>Advanced Machine Learning-Powered Health Insurance Cost Prediction</p>
    <small style="color: #999;">Made by: Ujan Pradhan & Rishav Prakash</small>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - USER INPUT
# ============================================================================

st.sidebar.markdown("## 📝 Your Information")
st.sidebar.markdown("---")

with st.sidebar.form("prediction_form"):
    st.markdown("### Personal Details")
    
    age = st.slider("🎂 Age", 18, 64, 35, help="Your current age")
    sex = st.selectbox("👤 Sex", ["male", "female"], help="Your biological sex")
    
    st.markdown("### Health Metrics")
    
    bmi = st.number_input("⚖️ BMI (Body Mass Index)", 15.0, 55.0, 25.0, 0.1,
                          help="Calculate: weight(kg) / height(m)²")
    
    st.caption(f"""
    BMI Category: {'🔴 Obese' if bmi >= 30 else '🟡 Overweight' if bmi >= 25 else '🟢 Normal' if bmi >= 18.5 else '🔵 Underweight'}
    """)
    
    st.markdown("### Family & Lifestyle")
    
    children = st.number_input("👶 Number of Children", 0, 5, 0,
                               help="Dependents covered by insurance")
    
    smoker = st.selectbox("🚬 Smoking Status", ["no", "yes"],
                          help="Current smoking status")
    
    if smoker == "yes":
        st.warning("⚠️ Smoking significantly increases costs")
    
    region = st.selectbox("📍 Region", 
                         ["northeast", "northwest", "southeast", "southwest"],
                         help="Your residential region in the US")
    
    st.markdown("---")
    
    submit_button = st.form_submit_button("🚀 Predict My Insurance Cost", 
                                          use_container_width=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

if not models_loaded:
    st.error("🚨 Models not found! Please ensure model files are in the 'models/' directory")
    st.info("📚 Run the Colab notebook first to generate model files")
    st.stop()

if not submit_button:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 AI-Powered</h3>
            <p>Advanced deep learning models for accurate predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Data-Driven</h3>
            <p>Trained on thousands of real insurance records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>💡 Personalized</h3>
            <p>Get AI-generated insights tailored to your profile</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👈 **Get Started:** Fill in your information in the sidebar and click 'Predict My Insurance Cost'")
    
    st.markdown("### 📈 Key Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Average Cost</div>
            <div class="metric-value">$13,270</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Smoker Impact</div>
            <div class="metric-value">+280%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">87%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Predictions Made</div>
            <div class="metric-value">1000+</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PREDICTION RESULTS
# ============================================================================

if submit_button:
    input_data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }
    
    with st.spinner('🔮 Analyzing your profile with AI...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        predictions = predict_cost(input_data)
        ensemble_cost = predictions['Ensemble']
    
    st.success("✅ Prediction Complete!")
    st.balloons()
    
    st.markdown(f"""
    <div class="metric-container" style="margin: 2rem 0; padding: 2rem;">
        <div class="metric-label">YOUR PREDICTED ANNUAL PREMIUM</div>
        <div class="metric-value" style="font-size: 4rem;">${ensemble_cost:,.0f}</div>
        <p style="opacity: 0.9; margin-top: 1rem;">Based on ensemble AI model (most accurate)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Model Predictions Comparison")
        
        fig = go.Figure()
        colors = ['#667eea', '#f093fb', '#4facfe', '#00f2fe']
        
        for idx, (model, cost) in enumerate(predictions.items()):
            fig.add_trace(go.Bar(
                x=[model],
                y=[cost],
                name=model,
                marker_color=colors[idx % len(colors)],
                text=f"${cost:,.0f}",
                textposition='outside',
                hovertemplate=f"<b>{model}</b><br>Cost: ${cost:,.0f}<extra></extra>"
            ))
        
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Model",
            yaxis_title="Predicted Cost ($)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 👤 Your Profile")
        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-item">
                <span class="profile-label">Age</span>
                <span class="profile-value">{age} years</span>
            </div>
            <div class="profile-item">
                <span class="profile-label">Sex</span>
                <span class="profile-value">{sex.capitalize()}</span>
            </div>
            <div class="profile-item">
                <span class="profile-label">BMI</span>
                <span class="profile-value">{bmi} {'🔴' if bmi >= 30 else '🟡' if bmi >= 25 else '🟢'}</span>
            </div>
            <div class="profile-item">
                <span class="profile-label">Children</span>
                <span class="profile-value">{children}</span>
            </div>
            <div class="profile-item">
                <span class="profile-label">Smoker</span>
                <span class="profile-value">{smoker.upper()} {'🔴' if smoker == 'yes' else '🟢'}</span>
            </div>
            <div class="profile-item">
                <span class="profile-label">Region</span>
                <span class="profile-value">{region.capitalize()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🤖 AI-Generated Insights")
    
    with st.spinner('🧠 Generating personalized AI analysis...'):
        time.sleep(1)
        insights = generate_ai_insights(input_data, ensemble_cost)
    
    st.markdown(f"""
    <div class="ai-insights">
        <h3>🎯 Personalized Analysis</h3>
        {insights}
    </div>
    """, unsafe_allow_html=True)
    
    if not AI_ENABLED:
        st.info("💡 **Tip:** Add your Gemini API key to `.streamlit/secrets.toml` for enhanced AI insights!")
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Risk Factor Analysis")
    
    risk_factors = []
    risk_values = []
    risk_colors = []
    
    if smoker == 'yes':
        risk_factors.append('Smoking')
        risk_values.append(23616)
        risk_colors.append('#ff6b6b')
    
    if bmi >= 30:
        risk_factors.append('Obesity')
        risk_values.append(5125)
        risk_colors.append('#ffa502')
    
    if age >= 55:
        age_impact = (age - 39) * 295
        risk_factors.append('Advanced Age')
        risk_values.append(age_impact)
        risk_colors.append('#4facfe')
    
    if risk_factors:
        fig2 = go.Figure()
        
        for factor, value, color in zip(risk_factors, risk_values, risk_colors):
            fig2.add_trace(go.Bar(
                x=[factor],
                y=[value],
                marker_color=color,
                text=f"${value:,.0f}",
                textposition='outside'
            ))
        
        fig2.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Risk Factor",
            yaxis_title="Additional Annual Cost ($)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("#### Risk Level Indicators")
        risk_html = ""
        if smoker == 'yes':
            risk_html += '<span class="risk-badge risk-high">🔴 High Risk: Smoking</span>'
        if bmi >= 30:
            risk_html += '<span class="risk-badge risk-high">🔴 High Risk: Obesity</span>'
        if age >= 55:
            risk_html += '<span class="risk-badge risk-medium">🟡 Moderate: Age 55+</span>'
        if smoker == 'no' and bmi < 25:
            risk_html += '<span class="risk-badge risk-low">🟢 Low Risk Profile</span>'
        
        st.markdown(risk_html, unsafe_allow_html=True)
    else:
        st.success("🎉 **Excellent!** No major risk factors identified!")
        st.markdown('<span class="risk-badge risk-low">🟢 Optimal Health Profile</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📥 Save Your Prediction")
    
    # Save to exports/ directory matching your structure
    prediction_data = pd.DataFrame([{
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Age': age,
        'Sex': sex,
        'BMI': bmi,
        'BMI_Category': 'Obese' if bmi >= 30 else 'Overweight' if bmi >= 25 else 'Normal',
        'Children': children,
        'Smoker': smoker,
        'Region': region,
        'Deep_Learning': predictions['Deep Learning'],
        'Random_Forest': predictions['Random Forest'],
        'XGBoost': predictions['XGBoost'],
        'Ensemble_Recommended': ensemble_cost,
        'Risk_Factors': ', '.join(risk_factors) if risk_factors else 'None',
        'Risk_Level': 'High' if smoker == 'yes' or bmi >= 30 else 'Low'
    }])
    
    csv_data = prediction_data.to_csv(index=False)
    
    st.download_button(
        label="📊 Download Prediction Data (CSV)",
        data=csv_data,
        file_name=f"insurance_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    <h4>🎓 Created by Ujan Pradhan & Rishav Prakash</h4>
    <p><strong>Insurance Cost Prediction with Deep Learning & AI</strong></p>
    <p>Powered by TensorFlow, XGBoost, Random Forest & Google Gemini AI</p>
    <p style="font-size: 0.9rem; color: #999; margin-top: 1rem;">
        © 2025 | Built with ❤️ using Streamlit
    </p>
    <p style="font-size: 0.8rem; color: #999;">
        For educational purposes | Models trained on historical data
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ About This Application"):
    st.markdown("""
    ### 🏥 Insurance Cost Prediction System
    
    Advanced AI-powered application for predicting health insurance costs.
    
    #### 🤖 Technology Stack
    
    - **Deep Learning:** TensorFlow Neural Network
    - **Machine Learning:** Random Forest, XGBoost
    - **AI Analysis:** Google Gemini Pro
    - **Frontend:** Streamlit
    - **Visualization:** Plotly
    
    #### 📊 Model Performance
    
    | Model | R² Score | RMSE |
    |-------|----------|------|
    | Deep Learning | 0.86 | $4,250 |
    | XGBoost | 0.85 | $4,400 |
    | **Ensemble** | **0.87** | **$4,100** |
    
    #### 📈 Key Findings
    
    - **Smoking:** +$23,600/year (+280%)
    - **Obesity:** +$5,100/year
    - **Age:** ~$295/year progressive increase
    
    #### 👥 Creators
    
    **Ujan Pradhan** & **Rishav Prakash**
    
    #### 📝 Disclaimer
    
    For educational purposes only. Actual costs may vary. Consult insurance professionals for official quotes.
    """)
