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

# Premium CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
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
    
    .status-spoof {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    .success-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe69e 100%);
        border-left: 5px solid #ffc107;
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
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
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

# Advanced Liveness Detection Functions
def detect_liveness_real_time(face_img, frame_history=None):
    """
    Advanced liveness detection to prevent photo spoofing
    Returns: (is_live, confidence, reason)
    """
    if face_img is None or face_img.size == 0:
        return False, 0, "No face detected"
    
    h, w = face_img.shape[:2]
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    scores = []
    reasons = []
    
    # 1. Texture Analysis (Real faces have natural skin texture)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture_score = min(1.0, laplacian_var / 200)
    scores.append(texture_score)
    if texture_score < 0.3:
        reasons.append("Smooth texture (possible photo)")
    else:
        reasons.append("Natural texture detected")
    
    # 2. Frequency Domain Analysis (Photos have different frequency patterns)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    # Calculate high frequency ratio
    center_y, center_x = h//2, w//2
    radius = min(h, w)//4
    low_freq_region = magnitude[center_y-radius:center_y+radius, center_x-radius:center_x+radius]
    high_freq_region = np.copy(magnitude)
    high_freq_region[center_y-radius:center_y+radius, center_x-radius:center_x+radius] = 0
    
    low_freq_sum = np.sum(low_freq_region)
    high_freq_sum = np.sum(high_freq_region)
    
    if low_freq_sum > 0:
        freq_ratio = high_freq_sum / low_freq_sum
        freq_score = min(1.0, freq_ratio / 0.5)
        scores.append(freq_score)
        if freq_score < 0.3:
            reasons.append("Unnatural frequency pattern")
        else:
            reasons.append("Natural frequency pattern")
    
    # 3. Edge Detection (Real faces have natural edge transitions)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    edge_score = 1.0 - abs(edge_density - 0.15) / 0.15
    edge_score = max(0, min(1.0, edge_score))
    scores.append(edge_score)
    if edge_score < 0.3:
        reasons.append("Unnatural edge pattern")
    else:
        reasons.append("Natural edge pattern")
    
    # 4. Brightness Analysis (Screen reflections cause unnatural brightness)
    brightness = np.mean(gray)
    brightness_score = 1.0 - abs(brightness - 127) / 127
    brightness_score = max(0, min(1.0, brightness_score))
    scores.append(brightness_score)
    if brightness > 200:
        reasons.append("Too bright (possible screen reflection)")
    elif brightness < 50:
        reasons.append("Too dark")
    else:
        reasons.append("Good lighting")
    
    # 5. Color Distribution (Real faces have natural color variation)
    if len(face_img.shape) == 3:
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:,:,1])
        saturation_score = min(1.0, saturation / 100)
        scores.append(saturation_score)
        if saturation_score < 0.3:
            reasons.append("Low color saturation (possible photo)")
        else:
            reasons.append("Natural color variation")
    
    # 6. Local Binary Patterns (Texture analysis)
    def get_lbp(image):
        lbp = np.zeros_like(image)
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] > center) << 7
                code |= (image[i-1, j] > center) << 6
                code |= (image[i-1, j+1] > center) << 5
                code |= (image[i, j+1] > center) << 4
                code |= (image[i+1, j+1] > center) << 3
                code |= (image[i+1, j] > center) << 2
                code |= (image[i+1, j-1] > center) << 1
                code |= (image[i, j-1] > center) << 0
                lbp[i, j] = code
        return lbp
    
    try:
        lbp = get_lbp(gray)
        lbp_hist = np.histogram(lbp.ravel(), bins=256, range=(0, 256))[0]
        lbp_entropy = -np.sum((lbp_hist / np.sum(lbp_hist)) * np.log2(lbp_hist / np.sum(lbp_hist) + 1e-7))
        entropy_score = min(1.0, lbp_entropy / 8)
        scores.append(entropy_score)
        if entropy_score < 0.3:
            reasons.append("Uniform texture (possible printout)")
        else:
            reasons.append("Natural texture variation")
    except:
        pass
    
    # Calculate final liveness score
    final_score = np.mean(scores)
    is_live = final_score > 0.55  # Threshold for liveness
    
    # Combine reasons
    reason = " | ".join(reasons[:3])  # Show top 3 reasons
    
    return is_live, final_score, reason

def detect_motion_liveness(frame_history):
    """Detect natural head movement (real people move, photos don't)"""
    if len(frame_history) < 5:
        return True, 0.5, "Analyzing movement..."
    
    # Calculate optical flow between frames
    movements = []
    for i in range(1, len(frame_history)):
        prev_gray = cv2.cvtColor(frame_history[i-1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame_history[i], cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2))
        movements.append(magnitude)
    
    avg_movement = np.mean(movements)
    movement_score = min(1.0, avg_movement / 2.0)
    
    is_moving = movement_score > 0.1
    return is_moving, movement_score, f"Movement: {movement_score:.2f}"

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1>✅ Smart Attendance System</h1>
    <p>Advanced Anti-Spoofing Technology - Only Real Faces Accepted</p>
</div>
""", unsafe_allow_html=True)

# Check registered users
if len(recognizer.names) == 0:
    st.markdown("""
    <div class="warning-card">
        ⚠️ No registered users found! Please go to Registration page first.
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Registration", use_container_width=True):
        st.switch_page("pages/1_Register.py")
    st.stop()

# Stats Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(recognizer.names)}</div>
        <div class="stat-label">Registered Users</div>
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
        <div class="stat-label">Anti-Spoofing ON</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">🎯</div>
        <div class="stat-label">Live Detection</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Camera Section
st.markdown('<div class="camera-card">', unsafe_allow_html=True)

# Mode selection
mode = st.radio("", ["🎥 Live Mode (Recommended)", "📸 Photo Mode (Fallback)"], horizontal=True, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

if mode == "🎥 Live Mode (Recommended)":
    st.markdown('<div class="status-badge status-active">🛡️ LIVE MODE + ANTI-SPOOFING ACTIVE</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card">💡 Looking for REAL faces only. Photos/screens will be rejected.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    liveness_placeholder = st.empty()
    log_container = st.container()
    
    if 'auto_scan_active' not in st.session_state:
        st.session_state.auto_scan_active = True
    if 'marked_names' not in st.session_state:
        st.session_state.marked_names = set()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⏹️ Stop Scanning", use_container_width=True):
            st.session_state.auto_scan_active = False
            st.rerun()
    
    if st.session_state.auto_scan_active:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access camera!")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            status_placeholder.markdown('<div class="success-card">🟢 Camera active - Scanning for LIVE faces...</div>', unsafe_allow_html=True)
            
            frame_count = 0
            frame_history = []
            spoof_attempts = 0
            
            while st.session_state.auto_scan_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Store frame history for motion detection
                frame_history.append(frame.copy())
                if len(frame_history) > 10:
                    frame_history.pop(0)
                
                # Process every 3rd frame
                if frame_count % 3 == 0:
                    faces = detector.detect_faces_optimized(frame, fast_mode=True)
                    
                    if len(faces) > 0:
                        for face_data in faces:
                            face_img = face_data['face_img']
                            x1, y1, x2, y2 = face_data['bbox']
                            
                            if (x2 - x1) > 60 and (y2 - y1) > 60:
                                # Check liveness
                                is_live, liveness_score, liveness_reason = detect_liveness_real_time(face_img, frame_history)
                                
                                if is_live and liveness_score > 0.55:
                                    # REAL face detected
                                    liveness_placeholder.markdown(f'<div class="success-card">✅ REAL face verified! (Score: {liveness_score:.2%}) - {liveness_reason}</div>', unsafe_allow_html=True)
                                    
                                    # Recognize face
                                    name, confidence = recognizer.recognize_face(face_img, threshold=0.45)
                                    
                                    if name and confidence > 0.45:
                                        if name not in st.session_state.marked_names:
                                            details = recognizer.student_details.get(name, {})
                                            success = attendance_db.mark_attendance(name, details)
                                            
                                            if success:
                                                st.session_state.marked_names.add(name)
                                                with log_container:
                                                    st.markdown(f'<div class="success-card">✅ REAL: {name} - Attendance Marked! (Confidence: {confidence:.2%}, Liveness: {liveness_score:.2%})</div>', unsafe_allow_html=True)
                                                
                                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                                label = f"LIVE: {name} ✓"
                                                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                            else:
                                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                                label = f"{name}"
                                                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        else:
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            label = f"{name} (Done)"
                                            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    else:
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                        label = "LIVE - Not Registered"
                                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                else:
                                    # SPOOF detected - Photo or screen
                                    spoof_attempts += 1
                                    liveness_placeholder.markdown(f'<div class="error-card">⚠️ SPOOF DETECTED! Photos/Screens not allowed. (Score: {liveness_score:.2%}) - {liveness_reason}</div>', unsafe_allow_html=True)
                                    
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    cv2.putText(frame, "SPOOF DETECTED!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            else:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                                cv2.putText(frame, "Too Close", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    if len(faces) == 0:
                        status_placeholder.markdown('<div class="info-card">🔍 Scanning for faces...</div>', unsafe_allow_html=True)
                    else:
                        status_placeholder.markdown(f'<div class="success-card">🎯 Found {len(faces)} face(s) | {len(st.session_state.marked_names)} marked | Spoof attempts: {spoof_attempts}</div>', unsafe_allow_html=True)
                
                # Overlay text
                cv2.putText(frame, f"Anti-Spoofing ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Marked: {len(st.session_state.marked_names)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "Photos/Screens Rejected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                time.sleep(0.03)
            
            cap.release()
            cv2.destroyAllWindows()

else:  # Photo Mode with liveness check
    st.markdown('<div class="status-badge status-active">📸 PHOTO MODE + ANTI-SPOOFING</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-card">⚠️ Photos of screens/prints will be rejected. Please take a live photo.</div>', unsafe_allow_html=True)
    
    picture = st.camera_input("Take a live photo", key="manual_capture")
    
    if picture:
        with st.spinner("Analyzing for liveness..."):
            image = Image.open(picture)
            frame = np.array(image)
            
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if frame[0,0,0] > frame[0,0,2]:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            faces = detector.detect_faces_optimized(frame, fast_mode=False)
            
            if len(faces) == 0:
                st.markdown('<div class="warning-card">❌ No face detected!</div>', unsafe_allow_html=True)
            else:
                for face_data in faces:
                    face_img = face_data['face_img']
                    x1, y1, x2, y2 = face_data['bbox']
                    
                    # Check liveness FIRST
                    is_live, liveness_score, liveness_reason = detect_liveness_real_time(face_img)
                    
                    if not is_live or liveness_score < 0.55:
                        st.markdown(f'<div class="error-card">❌ SPOOF DETECTED! {liveness_reason}<br>Please show a REAL face, not a photo or screen.</div>', unsafe_allow_html=True)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, "SPOOF DETECTED", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        st.markdown(f'<div class="success-card">✅ REAL face verified! (Liveness: {liveness_score:.2%})</div>', unsafe_allow_html=True)
                        
                        # Now recognize
                        name, confidence = recognizer.recognize_face(face_img, threshold=0.45)
                        
                        if name and confidence > 0.45:
                            details = recognizer.student_details.get(name, {})
                            success = attendance_db.mark_attendance(name, details)
                            
                            if success:
                                st.markdown(f'<div class="success-card">✅ REAL: {name} - Attendance Marked! (Confidence: {confidence:.2%})</div>', unsafe_allow_html=True)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                cv2.putText(frame, f"LIVE: {name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            else:
                                st.info(f"📌 {name} already marked today")
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{name} (Done)", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            st.warning(f"⚠️ REAL face detected but NOT REGISTERED (Confidence: {confidence:.2%})")
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(frame, "LIVE - Not Registered", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Verification Result", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Today's Attendance
st.markdown("---")
st.markdown("### 📊 Today's Attendance (Real Faces Only)")

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