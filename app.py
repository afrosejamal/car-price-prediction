import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import numpy as np
import sqlite3
from datetime import datetime

# --------------------- CUSTOM STYLING ---------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem !important;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subheader {
        font-size: 1.8rem !important;
        color: #2e86ab;
        border-bottom: 3px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #ff7f0e 0%, #ff4b1f 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(255,107,0,0.3);
        margin: 1rem 0;
    }
    .sidebar-header {
        background: linear-gradient(135deg, #1f77b4 0%, #2e86ab 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff7f0e;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(31,119,180,0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(31,119,180,0.4);
    }
    .dataframe {
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .caption-style {
        text-align: center;
        font-style: italic;
        color: #666;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 2px solid #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(page_title="Car Price Prediction", layout="wide", page_icon="🚗")

# --------------------- DATABASE SETUP ---------------------
DB_PATH = "car_predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user_input TEXT,
                    predicted_price REAL,
                    lower_range REAL,
                    upper_range REAL
                )''')
    conn.commit()
    conn.close()

def save_prediction(user_input_dict, predicted_price, lower_range, upper_range):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO predictions (timestamp, user_input, predicted_price, lower_range, upper_range)
                 VALUES (?, ?, ?, ?, ?)''', 
                 (timestamp, str(user_input_dict), predicted_price, lower_range, upper_range))
    conn.commit()
    conn.close()

def load_predictions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df

# Initialize database
init_db()

# --------------------- HEADER SECTION ---------------------
st.markdown('<h1 class="main-header">🚗 Car Price Prediction Pro</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
    Predict your car's resale value with cutting-edge Machine Learning technology! 
    <br>Get accurate estimates and deep insights in seconds.
</div>
""", unsafe_allow_html=True)

# --------------------- LOAD DATA ---------------------
DATA_PATH = "data/car_data.csv" if os.path.exists("data/car_data.csv") else "car_data.csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        
        target = "selling_price" if "selling_price" in df.columns else "price"
        if target in df.columns:
            df = df.dropna(subset=[target])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("❌ No data loaded. Please check your data file.")
    st.stop()

# --------------------- SIDEBAR STYLING ---------------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.markdown("### 🎯 Control Panel")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    target = "selling_price" if "selling_price" in df.columns else "price"
    df_encoded = df.copy()
    label_encoders = {}

    for col in df_encoded.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = "model/car_price_model.pkl"
    os.makedirs("model", exist_ok=True)

    if not os.path.exists(model_path):
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📉 MAE</h3>
            <h2>₹{mae:,.0f}</h2>
            <p>Mean Absolute Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 R² Score</h3>
            <h2>{r2:.3f}</h2>
            <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧾 Car Specifications")
    
    input_data = {}
    for col in X.columns:
        if col in df.columns:
            st.markdown(f'<div class="feature-card">', unsafe_allow_html=True)
            if df[col].dtype in ['int64', 'float64']:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                current_val = float(df[col].median())
                input_data[col] = st.slider(
                    f"{col}", 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=current_val,
                    help=f"Range: {min_val:.1f} to {max_val:.1f}"
                )
            else:
                options = sorted(df[col].astype(str).unique())
                default_option = options[len(options)//2] if options else ""
                input_data[col] = st.selectbox(
                    f"{col}", 
                    options,
                    index=options.index(default_option) if default_option in options else 0
                )
            st.markdown('</div>', unsafe_allow_html=True)

# --------------------- MAIN CONTENT AREA ---------------------
# Replace the Metrics Row section in your code with this:

# Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h3 style="margin: 0; font-size: 1rem; opacity: 0.9;">📈 Total Records</h3>
        <h2 style="margin: 0.5rem 0; font-size: 1.8rem; font-weight: 700;">{len(df):,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h3 style="margin: 0; font-size: 1rem; opacity: 0.9;">🎯 Features</h3>
        <h2 style="margin: 0.5rem 0; font-size: 1.8rem; font-weight: 700;">{len(df.columns)}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_price = df[target].mean() if target in df.columns else 0
    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h3 style="margin: 0; font-size: 1rem; opacity: 0.9;">💰 Avg Price</h3>
        <h2 style="margin: 0.5rem 0; font-size: 1.8rem; font-weight: 700;">₹{avg_price:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); border-radius: 15px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h3 style="margin: 0; font-size: 1rem; opacity: 0.9;">🤖 Model</h3>
        <h2 style="margin: 0.5rem 0; font-size: 1.5rem; font-weight: 700;">Random Forest</h2>
    </div>
    """, unsafe_allow_html=True)
 
st.markdown("#### 🔍 Data Preview")
st.dataframe(df.head(), use_container_width=True)

# --------------------- PREDICTION SECTION ---------------------
st.markdown("---")
st.markdown('<div class="subheader">🔮 Price Prediction</div>', unsafe_allow_html=True)

input_df = pd.DataFrame([input_data])
encoded_input = input_df.copy()
for col, le in label_encoders.items():
    if col in encoded_input.columns:
        if encoded_input[col].iloc[0] in le.classes_:
            encoded_input[col] = le.transform([encoded_input[col].iloc[0]])
        else:
            encoded_input[col] = le.transform([le.classes_[0]])
for col in X.columns:
    if col not in encoded_input.columns:
        encoded_input[col] = 0
encoded_input = encoded_input[X.columns]

col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_btn = st.button("🚀 PREDICT CAR PRICE", use_container_width=True)

if predict_btn:
    pred_price = model.predict(encoded_input)[0]
    error_margin = mae * 0.5

    # 💾 Save to database
    save_prediction(input_data, pred_price, pred_price - error_margin, pred_price + error_margin)

    st.markdown(f"""
    <div class="prediction-card">
        <h2>💰 PREDICTED CAR PRICE</h2>
        <h1 style="font-size: 3rem; margin: 1rem 0;">₹{pred_price:,.2f}</h1>
        <h3>📊 Estimated Range: ₹{pred_price - error_margin:,.2f} - ₹{pred_price + error_margin:,.2f}</h3>
        <p style="opacity: 0.9; margin-top: 1rem;">Based on advanced machine learning analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------- VISUALIZATIONS ---------------------
    st.markdown('<div class="subheader">📈 Insights & Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Price Distribution")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.hist(y, bins=30, color='#1f77b4', edgecolor='white', alpha=0.8)
        ax1.axvline(pred_price, color='#ff7f0e', linestyle='--', linewidth=3, label=f'Predicted: ₹{pred_price:,.0f}')
        ax1.set_xlabel("Selling Price (₹)", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')
        st.pyplot(fig1)
    
    with col2:
        st.markdown("#### 🏎 Top Features")
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(feat_imp)))
        bars = ax2.barh(range(len(feat_imp)), feat_imp.values, color=colors)
        ax2.set_yticks(range(len(feat_imp)))
        ax2.set_yticklabels(feat_imp.index)
        ax2.set_xlabel("Importance Score", fontsize=12)
        ax2.set_facecolor('#f8f9fa')
        plt.gca().invert_yaxis()
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, f'{width:.3f}', ha='left', va='center', fontsize=10)
        st.pyplot(fig2)
    
    st.markdown("#### 📈 Performance Analysis")
    col1, col2 = st.columns([2,1])
    with col1:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sample_indices = np.random.choice(len(y_test), min(100, len(y_test)), replace=False)
        y_test_sample = y_test.iloc[sample_indices]
        y_pred_sample = y_pred[sample_indices]
        ax3.scatter(y_test_sample, y_pred_sample, alpha=0.6, color='#1f77b4', s=60)
        ax3.plot([y_test_sample.min(), y_test_sample.max()], [y_test_sample.min(), y_test_sample.max()], 'r--', lw=2, alpha=0.8)
        ax3.set_xlabel("Actual Prices (₹)", fontsize=12)
        ax3.set_ylabel("Predicted Prices (₹)", fontsize=12)
        ax3.set_facecolor('#f8f9fa')
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
    with col2:
        st.markdown("#### 📋 Your Input")
        summary_data = [{"Feature": feature, "Value": value} for feature, value in input_data.items()]
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, height=300)

# --------------------- HISTORY SECTION ---------------------
st.markdown("---")
st.markdown('<div class="subheader">🕒 Prediction History</div>', unsafe_allow_html=True)
history_df = load_predictions()
if not history_df.empty:
    st.dataframe(history_df, use_container_width=True, height=400)
else:
    st.info("No predictions stored yet. Make your first prediction to see it here!")

# --------------------- FOOTER ---------------------
st.markdown("---")
st.markdown("""
<div class="caption-style">
    <h3> Built with Streamlit & RandomForestRegressor</h3>
    <p>Advanced Car Price Prediction System | Machine Learning Powered</p>
</div>
""", unsafe_allow_html=True)                 