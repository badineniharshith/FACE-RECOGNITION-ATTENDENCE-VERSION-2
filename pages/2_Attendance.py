import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os
import pandas as pd
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Mark Attendance",
    page_icon="✅",
    layout="wide"
)

# Detect if running in cloud
IS_CLOUD = os.environ.get('STREAMLIT_SHARING', False) or os.environ.get('STREAMLIT_CLOUD', False)

# Premium CSS
st.markdown("""
<style>
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero-section h1 {
        font-size: 2.5rem;
        margin: 0;
        color: white;
    }
    .camera-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 14px;
        margin: 5px;
    }
    .status-active {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    .success-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .error-card {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .info-card {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 5px solid #17a2b8;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stat-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    }
    .stat-number {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

from utils.optimized_face import OptimizedFaceDetector, OptimizedFaceRecognizer
from utils.database import AttendanceDB

# Initialize system
@st.cache_resource
def init_system():
    detector = OptimizedFaceDetector()
    recognizer = OptimizedFaceRecognizer()
    db = AttendanceDB()
    return detector, recognizer, db

detector, recognizer, attendance_db = init_system()

# Simple liveness detection
def is_real_face(face_img):
    if face_img is None or face_img.size == 0:
        return False, 0, "No face detected"
    
    h, w = face_img.shape[:2]
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var > 400:
        return False, 0.2, "Too sharp - possible photo"
    elif laplacian_var < 30:
        return False, 0.3, "Too blurry"
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    
    if edge_density > 0.35:
        return False, 0.3, "Too many sharp edges"
    elif edge_density < 0.03:
        return False, 0.3, "Too few details"
    
    texture_variance = np.var(gray)
    if texture_variance < 20:
        return False, 0.2, "Uniform texture - possible print"
    
    score = (min(1.0, laplacian_var / 200) * 0.4 + 
             (1.0 - abs(edge_density - 0.12) / 0.12) * 0.3 + 
             min(1.0, texture_variance / 150) * 0.3)
    score = max(0, min(1.0, score))
    
    is_real = score > 0.45
    
    if is_real:
        return True, score, f"Real face (Score: {score:.2%})"
    else:
        return False, score, f"Spoof detected (Score: {score:.2%})"

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1>✅ Smart Attendance System</h1>
    <p>Advanced Face Recognition + Anti-Spoofing</p>
</div>
""", unsafe_allow_html=True)

# Check registered users
if len(recognizer.names) == 0:
    st.warning("⚠️ No registered users found! Please go to Registration page first.")
    if st.button("Go to Registration", use_container_width=True):
        st.switch_page("pages/1_Register.py")
    st.stop()

# Stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(recognizer.names)}</div>
        <div class="stat-label">Registered</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    df = attendance_db.get_attendance_records()
    today_count = len(df[df['Date'] == datetime.now().strftime("%Y-%m-%d")]) if len(df) > 0 else 0
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{today_count}</div>
        <div class="stat-label">Present Today</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">🛡️</div>
        <div class="stat-label">Anti-Spoofing</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">🎯</div>
        <div class="stat-label">Live Only</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Section - Photo Upload Only (for cloud compatibility)
st.markdown('<div class="camera-card">', unsafe_allow_html=True)

st.info("📸 **Upload a photo** - The system will verify if it's a real face or a photo")
st.markdown("""
<div class="info-card">
    💡 Tips for best results:
    <ul>
        <li>Use good lighting (not too dark or bright)</li>
        <li>Look straight at the camera</li>
        <li>Remove glasses if possible</li>
        <li>Ensure your face is clearly visible</li>
    </ul>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    with st.spinner("Analyzing image..."):
        image = Image.open(uploaded_file)
        frame = np.array(image)
        
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if frame[0,0,0] > frame[0,0,2]:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        faces = detector.detect_faces_optimized(frame, fast_mode=False)
        
        if len(faces) == 0:
            st.error("❌ No face detected! Please ensure your face is clearly visible.")
        else:
            st.success(f"✅ {len(faces)} face(s) detected")
            
            for face_data in faces:
                face_img = face_data['face_img']
                x1, y1, x2, y2 = face_data['bbox']
                
                # Check if it's a real face
                is_real, liveness_score, liveness_reason = is_real_face(face_img)
                
                if not is_real or liveness_score < 0.45:
                    st.markdown(f'<div class="error-card">❌ SPOOF DETECTED: {liveness_reason}<br>Please upload a REAL face photo, not a screenshot or printed photo.</div>', unsafe_allow_html=True)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "SPOOF DETECTED", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    st.markdown(f'<div class="success-card">✅ REAL FACE VERIFIED! (Liveness: {liveness_score:.2%})</div>', unsafe_allow_html=True)
                    
                    # Recognize the face
                    name, confidence = recognizer.recognize_face(face_img, threshold=0.45)
                    
                    if name and confidence > 0.45:
                        details = recognizer.student_details.get(name, {})
                        success = attendance_db.mark_attendance(name, details)
                        
                        if success:
                            st.markdown(f'<div class="success-card">✅ Attendance marked for {name}! (Confidence: {confidence:.2%})</div>', unsafe_allow_html=True)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(frame, f"{name} ✓", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            st.info(f"📌 {name} already marked attendance today!")
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{name} (Already)", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        st.warning(f"⚠️ Real face detected but NOT REGISTERED in system (Confidence: {confidence:.2%})")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, "Not Registered", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Display processed image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Verification Result", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Today's Attendance
st.markdown("---")
st.markdown("### 📊 Today's Attendance")

df = attendance_db.get_attendance_records()
today = datetime.now().strftime("%Y-%m-%d")
today_df = df[df['Date'] == today] if len(df) > 0 else pd.DataFrame()

if len(today_df) > 0:
    st.dataframe(today_df[['Name', 'Student_ID', 'Department', 'Time']].sort_values('Time'), 
                use_container_width=True, hide_index=True)
else:
    st.info("ℹ️ No attendance marked yet today")

if st.button("← Back to Home", use_container_width=True):
    st.switch_page("app.py")
