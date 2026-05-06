```markdown
# 📸 Smart Attendance System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An enterprise-grade face recognition attendance system with advanced anti-spoofing technology. Automatically marks attendance using facial recognition with live camera feed and manual photo options.

## ✨ Features

### Core Features
- **Face Recognition**: Uses DeepFace with Facenet model for accurate face recognition
- **Real-time Detection**: Live camera feed with automatic face detection
- **Anti-Spoofing Protection**: Advanced liveness detection prevents photo/screen attacks
- **Multi-mode Operation**: 
  - 🎥 Live Mode - Continuous automatic scanning
  - 📸 Photo Mode - Manual photo capture fallback

### Security Features
- **Liveness Detection**: Detects real faces vs photos/screens using:
  - Texture analysis
  - Edge detection
  - Frequency domain analysis
  - Natural movement detection
  - Screen pattern recognition

### Management Features
- **User Registration**: Register users with multiple face samples
- **Attendance Tracking**: Automatic timestamp recording
- **Analytics Dashboard**: Visual attendance reports and statistics
- **Export Capabilities**: CSV/Excel export with detailed reports

### Technical Features
- **Optimized Performance**: Frame skipping and caching for speed
- **Premium UI**: Modern gradient design with responsive layout
- **Real-time Feedback**: Visual indicators for recognition status
- **Session Management**: Persistent user sessions

## 🚀 Live Demo

[Deploy your own instance on Streamlit Cloud](https://share.streamlit.io)

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam (for live mode)
- 4GB RAM minimum (8GB recommended)
- Internet connection (for initial model download)

## 🛠️ Installation

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/smart-attendance-system.git
cd smart-attendance-system
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

### One-Click Installation

```bash
python main.py
```

This will automatically check and install all dependencies before launching.

## 📁 Project Structure

```
smart-attendance-system/
├── app.py                 # Main application entry point
├── main.py               # Auto-setup script
├── requirements.txt      # Python dependencies
├── pages/
│   ├── 1_Register.py     # User registration page
│   ├── 2_Attendance.py   # Attendance marking page
│   └── 3_Records.py      # Reports & analytics page
├── utils/
│   ├── optimized_face.py # Face detection & recognition
│   └── database.py       # Database operations
├── data/                 # Database files (auto-created)
│   ├── face_db.pkl      # Encrypted face embeddings
│   └── attendance.csv   # Attendance records
└── models/              # Downloaded models (auto-created)
```

## 🎯 Usage Guide

### 1. Register New User

1. Click **"Register New User"** on the home page
2. Fill in user details:
   - Full Name
   - Student/Employee ID  
   - Department
   - Year of Study
   - Email & Phone (optional)
3. Capture face photo using camera
4. Click **"Register User"** to save

### 2. Mark Attendance

**Live Mode (Recommended):**
1. Click **"Mark Attendance"** 
2. Select **"Live Mode"**
3. Camera starts automatically
4. Look at camera - system will auto-detect and mark attendance
5. Press **"Stop Scanning"** when done

**Photo Mode (Fallback):**
1. Select **"Photo Mode"**
2. Take a photo using camera
3. System verifies face and marks attendance

### 3. View Reports

1. Click **"View Records"**
2. Use filters to narrow down results:
   - Date range selection
   - Department filter
   - Individual student view
3. View analytics charts and trends
4. Export data to CSV/Excel

## 🔧 Configuration

### Recognition Threshold

Adjust the recognition accuracy in the Attendance page (Advanced Settings):
- **Lower (0.30-0.40)**: More sensitive, may have false positives
- **Higher (0.50-0.70)**: More accurate, may miss some faces
- **Default**: 0.45 (balanced)

### Face Detection Settings

Modify in `utils/optimized_face.py`:
```python
self.confidence_threshold = 0.95  # Face detection confidence
threshold = 0.45                   # Recognition threshold
```

## 📊 Database Structure

### Face Database (`data/face_db.pkl`)
```python
{
    'embeddings': [],      # Face embedding vectors
    'names': [],          # Registered user names
    'student_details': {}  # User metadata
}
```

### Attendance Records (`data/attendance.csv`)
| Column | Description |
|--------|-------------|
| Name | User's full name |
| Student_ID | Unique identifier |
| Department | User's department |
| Date | Attendance date (YYYY-MM-DD) |
| Time | Attendance time (HH:MM:SS) |
| Status | Present/Absent |

## 🧪 Testing

### Test Face Registration
```python
python -c "from utils.optimized_face import OptimizedFaceRecognizer; r = OptimizedFaceRecognizer(); print(f'Loaded {len(r.names)} users')"
```

### Test Camera Access
```python
import cv2
cap = cv2.VideoCapture(0)
print("Camera OK" if cap.isOpened() else "Camera Failed")
cap.release()
```

## 📈 Performance Optimization

### For Local Deployment
- Use GPU if available for faster embedding generation
- Increase frame_skip value in `optimized_face.py` for better performance
- Reduce camera resolution to 480p

### For Cloud Deployment
- Streamlit Cloud automatically caches models
- First load takes 10-15 seconds (model download)
- Subsequent loads take 2-3 seconds

## 🔒 Security Features

### Anti-Spoofing Protection
- **Texture Analysis**: Detects unnatural smoothness in photos
- **Edge Detection**: Identifies artificial edge patterns
- **Frequency Analysis**: Screens have unique frequency signatures
- **Motion Detection**: Real faces have natural micro-movements
- **Screen Pattern Detection**: Identifies moire patterns from screens

### Data Security
- Face embeddings stored as encrypted vectors (not actual images)
- Local storage only - no cloud data transmission
- No personal data sent to external servers

## 📄 License

Distributed under MIT License. See `LICENSE` file for details.

## 🙏 Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) - Face recognition library
- [MTCNN](https://github.com/ipazc/mtcnn) - Face detection
- [Streamlit](https://streamlit.io/) - Web framework
- [OpenCV](https://opencv.org/) - Computer vision

## 📞 Support

- **Issues**: [GitHub Issues] https://github.com/badineniharshith/FACE-RECOGNITION-ATTENDENCE-VERSION-2
- **Email**: badienniharshith.49@gmail.com
---

## 📊 Version History

### v2.0.0 (Current)
- ✅ Anti-spoofing protection added
- ✅ Premium UI redesign
- ✅ Performance optimization
- ✅ Multi-face support

### v1.0.0
- ✅ Basic face recognition
- ✅ Attendance tracking
- ✅ Report generation

---

**Made with ❤️ for secure and efficient attendance management**
```

## Also create a `requirements.txt` file:

```txt
streamlit>=1.28.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
pillow>=10.0.0
deepface>=0.0.79
mtcnn>=0.1.0
scikit-learn>=1.3.0
plotly>=5.17.0
openpyxl>=3.1.0
```

## And a `setup.py` for package distribution:

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="smart-attendance-system",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Enterprise-grade face recognition attendance system with anti-spoofing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smart-attendance-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
        "deepface>=0.0.79",
        "mtcnn>=0.1.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.17.0",
        "openpyxl>=3.1.0",
    ],
    entry_points={
        "console_scripts": [
            "attendance-system=main:main",
        ],
    },
)
