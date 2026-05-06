import streamlit as st
import pandas as pd
import os
import pickle
from datetime import datetime

st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="📸",
    layout="wide"
)

st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        font-size: 18px;
        padding: 12px;
        border-radius: 10px;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
        transition: all 0.3s ease;
    }
    h1 {
        color: #2c3e50;
        border-bottom: 3px solid #4CAF50;
        padding-bottom: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        margin-top: 5px;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("pages", exist_ok=True)

st.title("📸 Smart Attendance System")
st.markdown("### Face Recognition Based Attendance Management")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📝 Register New User", use_container_width=True):
        st.switch_page("pages/1_Register.py")

with col2:
    if st.button("✅ Mark Attendance", use_container_width=True):
        st.switch_page("pages/2_Attendance.py")

with col3:
    if st.button("📊 View Records", use_container_width=True):
        st.switch_page("pages/3_Records.py")

# System Status
st.markdown("---")
st.markdown("### 📊 System Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    if os.path.exists("data/face_db.pkl"):
        try:
            with open("data/face_db.pkl", "rb") as f:
                data = pickle.load(f)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(data.get('names', []))}</div>
                    <div class="metric-label">Registered Users</div>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">0</div>
                <div class="metric-label">Registered Users</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0</div>
            <div class="metric-label">Registered Users</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    if os.path.exists("data/attendance.csv"):
        df = pd.read_csv("data/attendance.csv")
        today_count = len(df[df['Date'] == datetime.now().strftime("%Y-%m-%d")]) if len(df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{today_count}</div>
            <div class="metric-label">Today's Present</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0</div>
            <div class="metric-label">Today's Present</div>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">🟢</div>
        <div class="metric-label">System Active</div>
    </div>
    """, unsafe_allow_html=True)