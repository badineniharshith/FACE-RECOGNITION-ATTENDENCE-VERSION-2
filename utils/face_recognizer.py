import cv2
import numpy as np
import pickle
import os
from datetime import datetime

class FaceRecognizer:
    def __init__(self, db_path="data/face_db.pkl"):
        self.db_path = db_path
        self.load_database()
    
    def load_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data.get('embeddings', [])
                    self.names = data.get('names', [])
                    self.student_details = data.get('student_details', {})
            except:
                self.embeddings = []
                self.names = []
                self.student_details = {}
        else:
            self.embeddings = []
            self.names = []
            self.student_details = {}
    
    def save_database(self):
        data = {
            'embeddings': self.embeddings,
            'names': self.names,
            'student_details': self.student_details
        }
        with open(self.db_path, 'wb') as f:
            pickle.dump(data, f)
    
    def get_face_embedding(self, face_img):
        try:
            from deepface import DeepFace
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            embedding = DeepFace.represent(face_rgb, model_name='Facenet', enforce_detection=False)
            return embedding[0]['embedding']
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
    
    def recognize_face(self, face_img, threshold=0.4):
        embedding = self.get_face_embedding(face_img)
        if embedding is None or len(self.embeddings) == 0:
            return None, 0
        
        similarities = []
        for db_embedding in self.embeddings:
            similarity = np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        if max_similarity > threshold:
            return self.names[similarities.index(max_similarity)], max_similarity
        return None, max_similarity