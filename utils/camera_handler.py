import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

class CameraHandler:
    @staticmethod
    def get_camera():
        """Initialize camera with fallback options"""
        try:
            # Try to open camera
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                return cap
            else:
                st.warning("Camera not available in cloud environment. Using image upload mode.")
                return None
        except Exception as e:
            st.warning(f"Camera error: {e}. Using manual photo mode.")
            return None
    
    @staticmethod
    def process_uploaded_image(uploaded_file):
        """Process uploaded image file"""
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            return np.array(image)
        return None
