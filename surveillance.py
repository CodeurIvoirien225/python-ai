import cv2
import numpy as np
from collections import deque
import time

class BehaviorAnalyzer:
    def __init__(self):
        print("🔄 Initialisation de l'analyseur de comportement...")

        # Charger le classifieur Haar Cascade pour visage et yeux
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Historique et score
        self.gaze_history = deque(maxlen=30)
        self.movement_history = deque(maxlen=20)
        self.credibility_score = 100
        self.camera_blocked_frames = 0
        self.no_face_frames = 0
        self.last_detection_time = time.time()

        print("✅ Modèles OpenCV chargés avec succès")

    def analyze_behavior(self, frame):
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            camera_blocked = self.detect_camera_blocked(gray_frame)
            gaze_analysis = self.analyze_gaze(gray_frame)
            movement_analysis = self.analyze_movements(gray_frame)
            posture_analysis = self.analyze_posture(gray_frame)

            credibility_deduction = self.calculate_credibility_deduction(
                gaze_analysis, movement_analysis, posture_analysis, camera_blocked
            )

            self.credibility_score = max(20, self.credibility_score - credibility_deduction)

            return {
                'looking_away': gaze_analysis['looking_away'],
                'suspicious_movements': movement_analysis['suspicious_count'],
                'person_stood_up': posture_analysis['stood_up'],
                'camera_blocked': camera_blocked,
                'credibility_score': int(self.credibility_score),
                'gaze_direction': gaze_analysis['direction'],
                'head_movement': movement_analysis['head_movement'],
                'face_detected': gaze_analysis['face_detected'],
                'status': 'analyzed'
            }

        except Exception as e:
            print(f"❌ Erreur analyse: {e}")
            return {
                'looking_away': False,
                'suspicious_movements': 0,
                'person_stood_up': False,
                'camera_blocked': False,
                'credibility_score': 100,
                'gaze_direction': 'center',
                'head_movement': False,
                'face_detected': False,
                'status': 'error'
            }

    def detect_camera_blocked(self, gray_frame):
        try:
            brightness = np.mean(gray_frame)
            contrast = np.std(gray_frame)

            if brightness < 30 and contrast < 15:
                self.camera_blocked_frames += 1
            else:
                self.camera_blocked_frames = max(0, self.camera_blocked_frames - 0.5)

            return self.camera_blocked_frames > 5
        except Exception as e:
            print(f"❌ Erreur détection caméra bouchée: {e}")
            return False

    def analyze_gaze(self, gray_frame):
        try:
            faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            looking_away = False
            direction = "center"
            face_detected = len(faces) > 0

            if face_detected:
                self.no_face_frames = 0
                x, y, w, h = faces[0]
                roi_gray = gray_frame[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) == 1:
                    # Si un seul œil détecté, on suppose que la personne regarde sur le côté
                    looking_away = True
                    direction = "left"
                elif len(eyes) == 0:
                    looking_away = True
                    direction = "right"
            else:
                self.no_face_frames += 1

            self.gaze_history.append(looking_away)
            return {'looking_away': looking_away, 'direction': direction, 'face_detected': face_detected}

        except Exception as e:
            print(f"❌ Erreur analyse regard: {e}")
            return {'looking_away': False, 'direction': 'center', 'face_detected': False}

    def analyze_movements(self, gray_frame):
        # Détection simple de mouvements via différence de frames
        try:
            blurred = cv2.GaussianBlur(gray_frame, (21, 21), 0)
            if not hasattr(self, 'prev_frame'):
                self.prev_frame = blurred
                return {'suspicious_count': 0, 'head_movement': False}

            frame_delta = cv2.absdiff(self.prev_frame, blurred)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = np.sum(thresh) / 255

            self.prev_frame = blurred

            head_movement = motion_score > 5000  # seuil arbitraire
            suspicious_count = 1 if head_movement else 0

            self.movement_history.append(suspicious_count)
            return {'suspicious_count': suspicious_count, 'head_movement': head_movement}

        except Exception as e:
            print(f"❌ Erreur analyse mouvements: {e}")
            return {'suspicious_count': 0, 'head_movement': False}

    def analyze_posture(self, gray_frame):
        # Détection simple de posture : visage en haut/bas de l'image
        try:
            faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            stood_up = False
            if len(faces) > 0:
                x, y, w, h = faces[0]
                if y < 50:  # seuil arbitraire
                    stood_up = True
            return {'stood_up': stood_up}
        except Exception as e:
            print(f"❌ Erreur analyse posture: {e}")
            return {'stood_up': False}

    def calculate_credibility_deduction(self, gaze, movement, posture, camera_blocked):
        deduction = 0

        if gaze['looking_away']:
            deduction += 3
        if movement['suspicious_count'] > 0:
            deduction += movement['suspicious_count'] * 2
        if posture['stood_up']:
            deduction += 15
        if camera_blocked:
            deduction += 10
        if not gaze['face_detected'] and self.no_face_frames > 10:
            deduction += 5

        if deduction == 0 and self.credibility_score < 100:
            self.credibility_score = min(100, self.credibility_score + 0.5)

        return deduction
