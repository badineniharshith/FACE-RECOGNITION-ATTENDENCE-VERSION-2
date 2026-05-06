import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Register New User", page_icon="📝")

st.title("📝 Face Registration")
st.markdown("### Register new user with face capture")

from utils.optimized_face import OptimizedFaceDetector, OptimizedFaceRecognizer

# Cache models for faster loading
@st.cache_resource
def load_models():
    detector = OptimizedFaceDetector()
    recognizer = OptimizedFaceRecognizer()
    return detector, recognizer

detector, recognizer = load_models()

# Registration form
with st.form("registration_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name *", placeholder="Enter full name")
        student_id = st.text_input("Student ID *", placeholder="Enter student ID")
        email = st.text_input("Email Address", placeholder="student@example.com")
    
    with col2:
        department = st.selectbox("Department *", ["Computer Science", "Engineering", "Business", "Medicine", "Law", "Arts", "Other"])
        year = st.selectbox("Year of Study *", ["1st Year", "2nd Year", "3rd Year", "4th Year", "Masters"])
        phone = st.text_input("Phone Number", placeholder="+91XXXXXXXXXX")
    
    st.markdown("### 📸 Capture Face")
    st.markdown("*Please look straight at camera and ensure good lighting*")
    
    picture = st.camera_input("Take a picture", key="reg_camera")
    
    submitted = st.form_submit_button("Register User", use_container_width=True)
    
    if submitted:
        if not name or not student_id:
            st.error("❌ Please fill in all required fields!")
        elif picture is None:
            st.error("❌ Please capture a face photo!")
        else:
            with st.spinner("🔄 Processing face registration..."):
                image = Image.open(picture)
                frame = np.array(image)
                
                # Detect faces
                faces = detector.detect_faces_optimized(frame, fast_mode=False)
                
                if len(faces) == 0:
                    st.error("❌ No face detected! Please ensure your face is clearly visible and well-lit.")
                elif len(faces) > 1:
                    st.warning("⚠️ Multiple faces detected! Please ensure only your face is in frame.")
                else:
                    face_img = faces[0]['face_img']
                    
                    # Check face quality
                    quality_pass, quality_msg = recognizer.check_face_quality(face_img)
                    
                    if not quality_pass:
                        st.warning(f"⚠️ {quality_msg}")
                    
                    # Register the face
                    department_full = f"{department} - {year}"
                    success = recognizer.register_face(face_img, name, student_id, department_full)
                    
                    if success:
                        st.success(f"✅ Successfully registered {name}!")
                        st.balloons()
                        
                        # Show preview
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Captured Image", use_container_width=True)
                        with col2:
                            st.image(face_img, caption="Detected Face", use_container_width=True)
                        
                        st.info(f"""
                        **Registration Details:**
                        - Student ID: {student_id}
                        - Department: {department_full}
                        - Email: {email if email else 'Not provided'}
                        - Phone: {phone if phone else 'Not provided'}
                        """)
                    else:
                        st.error("❌ Registration failed! Please try again with better lighting.")

# Display registered users
st.markdown("---")
st.markdown("### 📊 Registered Users")

if len(recognizer.names) > 0:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Registered", len(recognizer.names))
    with col2:
        depts = set([recognizer.student_details.get(name, {}).get('department', 'Unknown') for name in recognizer.names])
        st.metric("Departments", len(depts))
    with col3:
        st.metric("Active System", "✅")
    
    with st.expander("View All Registered Users"):
        for i, name in enumerate(recognizer.names, 1):
            details = recognizer.student_details.get(name, {})
            st.write(f"{i}. **{name}** - ID: {details.get('student_id', 'N/A')} - {details.get('department', 'N/A')}")
else:
    st.info("No users registered yet. Be the first to register!")

# Back button
if st.button("← Back to Home", use_container_width=True):
    st.switch_page("app.py")