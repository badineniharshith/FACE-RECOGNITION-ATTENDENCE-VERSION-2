import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            self.confidence_threshold = 0.95
        except Exception as e:
            print(f"Error loading MTCNN: {e}")
            self.detector = None
    
    def detect_faces(self, frame):
        if self.detector is None:
            return []
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(frame_rgb)
            
            faces = []
            h, w = frame.shape[:2]
            
            for detection in detections:
                if detection['confidence'] > self.confidence_threshold:
                    x, y, width, height = detection['box']
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(w, x + width), min(h, y + height)
                    
                    if x2 > x1 and y2 > y1:
                        faces.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': detection['confidence'],
                            'face_img': frame[y1:y2, x1:x2]
                        })
            return faces
        except Exception as e:
            print(f"Detection error: {e}")
            return []