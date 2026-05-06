import cv2
import numpy as np
import pickle
import os
from datetime import datetime

class OptimizedFaceDetector:
    def __init__(self):
        self.detector = None
        self.load_detector()
        
    def load_detector(self):
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            self.confidence_threshold = 0.95
            print("Face detector loaded successfully")
        except Exception as e:
            print(f"Error loading detector: {e}")
            self.detector = None
    
    def detect_faces_optimized(self, frame, fast_mode=True):
        """Detect faces in frame"""
        if self.detector is None:
            return []
        
        try:
            # Convert to RGB (MTCNN expects RGB)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # For better detection in fast mode, resize if too large
            h, w = frame_rgb.shape[:2]
            original_size = (w, h)
            
            if fast_mode and w > 800:
                scale = 800 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_small = cv2.resize(frame_rgb, (new_w, new_h))
                detections = self.detector.detect_faces(frame_small)
                
                # Scale back coordinates
                for detection in detections:
                    detection['box'] = [
                        int(detection['box'][0] / scale),
                        int(detection['box'][1] / scale),
                        int(detection['box'][2] / scale),
                        int(detection['box'][3] / scale)
                    ]
            else:
                detections = self.detector.detect_faces(frame_rgb)
            
            faces = []
            h, w = frame.shape[:2]
            
            for detection in detections:
                if detection['confidence'] > self.confidence_threshold:
                    x, y, width, height = detection['box']
                    
                    # Ensure positive coordinates
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(w, x + width)
                    y2 = min(h, y + height)
                    
                    # Only add if face size is reasonable
                    if x2 > x1 and y2 > y1 and width > 40 and height > 40:
                        faces.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': detection['confidence'],
                            'face_img': frame[y1:y2, x1:x2]
                        })
            
            return faces
        except Exception as e:
            print(f"Detection error: {e}")
            return []

class OptimizedFaceRecognizer:
    def __init__(self, db_path="data/face_db.pkl"):
        self.db_path = db_path
        self.embeddings = []
        self.names = []
        self.student_details = {}
        self.load_database()
        
    def load_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data.get('embeddings', [])
                    self.names = data.get('names', [])
                    self.student_details = data.get('student_details', {})
                print(f"Loaded {len(self.names)} registered faces")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.embeddings = []
                self.names = []
                self.student_details = {}
    
    def save_database(self):
        try:
            data = {
                'embeddings': self.embeddings,
                'names': self.names,
                'student_details': self.student_details
            }
            with open(self.db_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Error saving: {e}")
            return False
    
    def get_face_embedding(self, face_img):
        try:
            from deepface import DeepFace
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            result = DeepFace.represent(face_rgb, model_name='Facenet', enforce_detection=False)
            
            # DeepFace returns a list for multiple faces, take first
            if isinstance(result, list):
                return result[0]['embedding']
            else:
                return result['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def register_face(self, face_img, name, student_id, department):
        embedding = self.get_face_embedding(face_img)
        if embedding is not None:
            self.embeddings.append(embedding)
            self.names.append(name)
            self.student_details[name] = {
                'student_id': student_id,
                'department': department,
                'registration_date': str(datetime.now())
            }
            self.save_database()
            return True
        return False
    
    def recognize_face(self, face_img, threshold=0.45):
        """Recognize face with improved threshold"""
        embedding = self.get_face_embedding(face_img)
        
        if embedding is None:
            return None, 0
        
        if len(self.embeddings) == 0:
            return None, 0
        
        # Calculate cosine similarity
        similarities = []
        for db_embedding in self.embeddings:
            similarity = np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        
        if max_similarity > threshold:
            best_match_idx = similarities.index(max_similarity)
            return self.names[best_match_idx], max_similarity
        
        return None, max_similarity