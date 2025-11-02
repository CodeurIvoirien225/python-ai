import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class BehaviorAnalyzer:
    def __init__(self):
        print("🔄 Initialisation de l'analyseur de comportement...")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose

        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✅ Modèles MediaPipe chargés avec succès")
        except Exception as e:
            print(f"❌ Erreur chargement modèles: {e}")
            raise

        # Historique et score
        self.gaze_history = deque(maxlen=30)
        self.movement_history = deque(maxlen=20)
        self.credibility_score = 100
        self.camera_blocked_frames = 0
        self.no_face_frames = 0
        self.last_detection_time = time.time()

    def analyze_behavior(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            camera_blocked = self.detect_camera_blocked(frame)
            gaze_analysis = self.analyze_gaze(rgb_frame)
            movement_analysis = self.analyze_movements(rgb_frame)
            posture_analysis = self.analyze_posture(rgb_frame)

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

    def detect_camera_blocked(self, frame):
        """Détecte si la caméra est bouchée ou couverte"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)

            if brightness < 30 and contrast < 15:
                self.camera_blocked_frames += 1
            else:
                self.camera_blocked_frames = max(0, self.camera_blocked_frames - 0.5)

            return self.camera_blocked_frames > 5

        except Exception as e:
            print(f"❌ Erreur détection caméra bouchée: {e}")
            return False

    def analyze_gaze(self, frame):
        try:
            results = self.face_mesh.process(frame)
            looking_away = False
            direction = "center"
            face_detected = False

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                face_detected = True
                self.no_face_frames = 0

                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                nose_tip = face_landmarks.landmark[1]

                eye_center_x = (left_eye.x + right_eye.x) / 2
                horizontal_diff = nose_tip.x - eye_center_x

                if horizontal_diff > 0.1:
                    direction = "right"
                    looking_away = True
                elif horizontal_diff < -0.1:
                    direction = "left"
                    looking_away = True
            else:
                self.no_face_frames += 1

            self.gaze_history.append(looking_away)

            return {
                'looking_away': looking_away,
                'direction': direction,
                'face_detected': face_detected
            }

        except Exception as e:
            print(f"❌ Erreur analyse regard: {e}")
            return {'looking_away': False, 'direction': 'center', 'face_detected': False}

    def analyze_movements(self, frame):
        try:
            results = self.pose.process(frame)
            suspicious_count = 0
            head_movement = False

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
                right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]

                head_tilt = abs(left_ear.y - right_ear.y)
                if head_tilt > 0.1:
                    head_movement = True
                    suspicious_count += 1

                left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                mouth = landmarks[self.mp_pose.PoseLandmark.MOUTH_LEFT]

                if (abs(left_wrist.y - mouth.y) < 0.15 or
                    abs(right_wrist.y - mouth.y) < 0.15):
                    suspicious_count += 3

            self.movement_history.append(suspicious_count)

            return {'suspicious_count': suspicious_count, 'head_movement': head_movement}

        except Exception as e:
            print(f"❌ Erreur analyse mouvements: {e}")
            return {'suspicious_count': 0, 'head_movement': False}

    def analyze_posture(self, frame):
        try:
            results = self.pose.process(frame)
            stood_up = False

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                shoulder_height = (left_shoulder.y + right_shoulder.y) / 2

                if shoulder_height < 0.3:
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
