"""
ASL Fingerspelling Letter Recognition - Simplified Version
- Direct ONNX Runtime for 24 static letters (A-I, K-Y)
- Rule-based trajectory detection for motion letters J and Z
- No TensorRT dependency required
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import json
from pathlib import Path
from collections import deque, Counter
import time

# MediaPipe import with error handling
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("❌ MediaPipe not installed. Run: pip install mediapipe")
    exit(1)

# Verify MediaPipe has required modules
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except AttributeError:
    print("❌ MediaPipe 'solutions' module not found")
    exit(1)

# ONNX Runtime import (simpler alternative to TensorRT)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("❌ ONNX Runtime not installed. Run: pip install onnxruntime")
    exit(1)


class MotionTrajectory:
    """Track hand movement for motion letter recognition"""
    
    def __init__(self, max_length: int = 30):
        self.max_length = max_length
        self.points = []
        self.timestamps = []
        self.finger_tip_points = []
        self.finger_type = "unknown"
    
    def add_point(self, x: float, y: float, finger_tip_x: float = None, 
                  finger_tip_y: float = None, finger_type: str = "unknown"):
        """Add a point to the trajectory"""
        self.points.append((x, y))
        self.timestamps.append(time.time())
        
        if finger_tip_x is not None and finger_tip_y is not None:
            self.finger_tip_points.append((finger_tip_x, finger_tip_y))
        
        if finger_type != "unknown":
            self.finger_type = finger_type
        
        if len(self.points) > self.max_length:
            self.points.pop(0)
            self.timestamps.pop(0)
            if self.finger_tip_points:
                self.finger_tip_points.pop(0)
    
    def clear(self):
        self.points.clear()
        self.timestamps.clear()
        self.finger_tip_points.clear()
        self.finger_type = "unknown"
    
    def is_empty(self) -> bool:
        return len(self.points) == 0
    
    def get_total_distance(self) -> float:
        if len(self.points) < 2:
            return 0.0
        points_array = np.array(self.points)
        displacements = np.diff(points_array, axis=0)
        return np.sum(np.linalg.norm(displacements, axis=1))
    
    def get_features(self) -> dict:
        """Extract trajectory features for classification"""
        if len(self.points) < 3:
            return {
                'path_length': 0.0,
                'straightness': 0.0,
                'direction_changes': 0,
                'avg_angle': 0.0,
                'vertical_movement': 0.0,
                'horizontal_movement': 0.0
            }
        
        points_array = np.array(self.points)
        displacements = np.diff(points_array, axis=0)
        path_length = np.sum(np.linalg.norm(displacements, axis=1))
        
        start_end_dist = np.linalg.norm(points_array[-1] - points_array[0])
        straightness = start_end_dist / (path_length + 1e-6)
        
        angles = []
        for i in range(1, len(points_array) - 1):
            v1 = points_array[i] - points_array[i-1]
            v2 = points_array[i+1] - points_array[i]
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 0.001 and norm2 > 0.001:
                cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
                angles.append(np.arccos(cos_angle))
        
        total_displacement = points_array[-1] - points_array[0]
        
        return {
            'path_length': path_length,
            'straightness': straightness,
            'direction_changes': len([a for a in angles if a > np.pi/4]),
            'avg_angle': np.mean(angles) if angles else 0.0,
            'vertical_movement': abs(total_displacement[1]),
            'horizontal_movement': abs(total_displacement[0])
        }


class ASLLetterRecognizer:
    """Real-time ASL recognition with ONNX Runtime + Rule-based Motion"""
    
    def __init__(self, model_dir: str = "asl_data"):
        self.model_dir = Path(model_dir)

        # Load ONNX model and metadata
        self.onnx_session = self._load_onnx_model("asl_static_model.onnx")
        self.metadata = self._load_metadata_json("asl_static_metadata.json")
        
        # Store MediaPipe modules
        self.mp_hands = mp_hands
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        
        # Finger indices
        self.FINGER_INDICES = {
            'wrist': 0, 'thumb': 4, 'index': 8,
            'middle': 12, 'ring': 16, 'pinky': 20
        }
        
        # Recognition state
        self.prev_hand_landmarks = None
        self.motion_trajectory = MotionTrajectory(max_length=35)
        self.motion_detection_active = False
        self.motion_start_time = None
        self.motion_timeout = 1.2
        
        # Motion detection parameters
        self.movement_threshold = 0.02
        self.min_trajectory_distance = 0.1
        self.motion_cooldown_time = 1.5
        self.last_motion_detection = 0
        self.consecutive_movements = 0
        self.required_consecutive_movements = 3
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=7)
        self.current_prediction = None
        self.prediction_confidence = 0.0
        
        # Word building
        self.detected_word = ""
        self.last_letter = None
        self.last_detection_time = 0
        self.detection_cooldown = 1.5
        
        # Static letter stability
        self.stable_letter_count = 0
        self.stable_letter_threshold = 10

        # Motion letter hold
        self.last_motion_letter = None
        self.last_motion_confidence = 0.0
        self.motion_hold_time = 1.0

        if self.onnx_session is not None:
            print("✓ Loaded ONNX model (static letters)")
        print("✓ Using rule-based motion detection for J/Z")

    def _load_onnx_model(self, filename: str):
        """Load ONNX model with ONNX Runtime"""
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            print(f"⚠️ ONNX model not found: {filepath}")
            print("   Static letter recognition will not work.")
            return None
        
        try:
            session = ort.InferenceSession(str(filepath))
            print(f"✓ Loaded ONNX model: {filepath}")
            return session
        except Exception as e:
            print(f"❌ Error loading ONNX model: {e}")
            return None
    
    def _load_metadata_json(self, filename: str):
        """Load metadata from JSON file (instead of pickle)"""
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            print(f"⚠️ Metadata JSON not found: {filepath}")
            print("   Will use default label mapping (A-Y excluding J)")
            # Create default metadata
            labels = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]
            return {
                'labels': labels,
                'normalization_params': {
                    'mean': [0.0] * 42,
                    'std': [1.0] * 42
                }
            }
        
        try:
            with open(filepath, 'r') as f:
                metadata = json.load(f)
            print(f"✓ Loaded metadata: {filepath}")
            return metadata
        except Exception as e:
            print(f"⚠️ Error loading metadata: {e}")
            print("   Using default label mapping")
            labels = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]
            return {
                'labels': labels,
                'normalization_params': {
                    'mean': [0.0] * 42,
                    'std': [1.0] * 42
                }
            }
    
    def extract_static_features(self, hand_landmarks) -> Optional[np.ndarray]:
        """Extract 42 features from hand landmarks"""
        if hand_landmarks is None:
            return None
        
        try:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            
            landmarks = np.array(landmarks, dtype=np.float32)
            
            # Normalize relative to wrist
            wrist_x, wrist_y = landmarks[0], landmarks[1]
            landmarks[::2] -= wrist_x
            landmarks[1::2] -= wrist_y
            
            # Scale by hand size (wrist to middle finger tip)
            hand_size = np.sqrt(
                (landmarks[24] - landmarks[0])**2 + 
                (landmarks[25] - landmarks[1])**2
            ) + 1e-6
            landmarks /= hand_size
            
            # Apply normalization if available
            if self.metadata and 'normalization_params' in self.metadata:
                norm_params = self.metadata['normalization_params']
                mean = np.array(norm_params['mean'], dtype=np.float32)
                std = np.array(norm_params['std'], dtype=np.float32)
                landmarks = (landmarks - mean) / (std + 1e-6)
            
            return landmarks
        except Exception as e:
            print(f"⚠ Feature extraction error: {e}")
            return None
    
    def get_finger_tip_position(self, hand_landmarks, finger_type: str) -> Tuple[float, float]:
        """Get finger tip position"""
        if hand_landmarks is None:
            return (0, 0)
        finger_idx = self.FINGER_INDICES.get(finger_type, 0)
        finger = hand_landmarks.landmark[finger_idx]
        return (finger.x, finger.y)
    
    def detect_motion_start(self, hand_landmarks, prev_landmarks) -> bool:
        """Detect significant hand movement"""
        if hand_landmarks is None or prev_landmarks is None:
            return False
        
        current_time = time.time()
        if current_time - self.last_motion_detection < self.motion_cooldown_time:
            return False
        
        curr_wrist = hand_landmarks.landmark[0]
        prev_wrist = prev_landmarks.landmark[0]
        
        movement = np.sqrt(
            (curr_wrist.x - prev_wrist.x)**2 + 
            (curr_wrist.y - prev_wrist.y)**2
        )
        
        if movement > self.movement_threshold:
            self.consecutive_movements += 1
        else:
            self.consecutive_movements = max(0, self.consecutive_movements - 1)
        
        return self.consecutive_movements >= self.required_consecutive_movements
    
    def get_smoothed_prediction(self):
        """Get smoothed static prediction"""
        if len(self.prediction_history) == 0:
            return None, 0.0
        
        counter = Counter(self.prediction_history)
        most_common = counter.most_common(1)[0]
        letter, count = most_common
        confidence = count / len(self.prediction_history)
        
        return letter, confidence
    
    def add_letter_to_word(self, letter: str):
        """Add detected letter to word"""
        current_time = time.time()
        
        if (letter != self.last_letter or 
            (current_time - self.last_detection_time) > self.detection_cooldown):
            self.detected_word += letter
            self.last_letter = letter
            self.last_detection_time = current_time
            print(f"✓ Added: {letter} | Word: '{self.detected_word}'")
    
    def classify_motion_trajectory(self) -> Tuple[Optional[str], float]:
        """Rule-based classification for J and Z"""
        features = self.motion_trajectory.get_features()
        finger_type = self.motion_trajectory.finger_type
        
        if finger_type not in ["pinky", "index"]:
            return None, 0.0
        
        if features['path_length'] < 0.15:
            return None, 0.0
        
        confidence = 0.5
        
        if finger_type == "pinky":
            letter = 'J'
            confidence += 0.25
        elif finger_type == "index":
            letter = 'Z'
            confidence += 0.25
        else:
            return None, 0.0
        
        if features['path_length'] > 0.2:
            confidence += 0.1
        
        if letter == 'Z':
            if features['direction_changes'] >= 2:
                confidence += 0.1
            if features['straightness'] < 0.6:
                confidence += 0.05
        elif letter == 'J':
            if features['vertical_movement'] > features['horizontal_movement']:
                confidence += 0.1
            if features['straightness'] > 0.4:
                confidence += 0.05
        
        return letter, min(confidence, 1.0)
    
    def recognize_frame(self, hand_landmarks):
        """Recognize letter from current frame"""
        if hand_landmarks is None:
            return None, 0.0
        
        current_time = time.time()

        # Keep motion letter visible for a short time
        if (self.last_motion_letter is not None and 
            (current_time - self.last_motion_detection) < self.motion_hold_time):
            return self.last_motion_letter, self.last_motion_confidence
        
        # Check for motion start
        if self.prev_hand_landmarks is not None:
            if self.detect_motion_start(hand_landmarks, self.prev_hand_landmarks):
                if not self.motion_detection_active:
                    self.motion_detection_active = True
                    self.motion_start_time = current_time
                    self.motion_trajectory.clear()
                    self.prediction_history.clear()
                    print("🔄 Motion tracking started (J/Z)")
        
        # Handle motion timeout
        if self.motion_detection_active:
            if current_time - self.motion_start_time > self.motion_timeout:
                total_distance = self.motion_trajectory.get_total_distance()
                
                if total_distance > self.min_trajectory_distance:
                    letter, confidence = self.classify_motion_trajectory()
                    
                    if letter and confidence > 0.7:
                        print(f"✅ Motion detected: {letter} ({confidence:.0%})")
                        self.last_motion_letter = letter
                        self.last_motion_confidence = confidence
                        self.current_prediction = letter
                        self.prediction_confidence = confidence
                        self.last_motion_detection = current_time
                        
                        self.motion_detection_active = False
                        self.motion_trajectory.clear()
                        self.consecutive_movements = 0
                        
                        self.prev_hand_landmarks = hand_landmarks
                        return letter, confidence
                
                self.motion_detection_active = False
                self.motion_trajectory.clear()
                self.consecutive_movements = 0
                return None, 0.0
        
        # Track motion trajectory
        if self.motion_detection_active:
            wrist = hand_landmarks.landmark[0]
            pinky = hand_landmarks.landmark[self.FINGER_INDICES['pinky']]
            index = hand_landmarks.landmark[self.FINGER_INDICES['index']]
            
            if pinky.y < index.y:
                pinky_x, pinky_y = self.get_finger_tip_position(hand_landmarks, 'pinky')
                self.motion_trajectory.add_point(wrist.x, wrist.y, pinky_x, pinky_y, "pinky")
            else:
                index_x, index_y = self.get_finger_tip_position(hand_landmarks, 'index')
                self.motion_trajectory.add_point(wrist.x, wrist.y, index_x, index_y, "index")
        
        # Static letter recognition with ONNX Runtime
        if self.onnx_session is not None and not self.motion_detection_active:
            static_features = self.extract_static_features(hand_landmarks)
            
            if static_features is not None:
                try:
                    # Prepare input
                    input_name = self.onnx_session.get_inputs()[0].name
                    features = static_features.reshape(1, -1).astype(np.float32)
                    
                    # Run inference
                    outputs = self.onnx_session.run(None, {input_name: features})
                    prediction = outputs[0][0]
                    
                    pred_class = np.argmax(prediction)
                    confidence = float(prediction[pred_class])
                    
                    # Get letter from metadata
                    if self.metadata and 'labels' in self.metadata:
                        labels = self.metadata['labels']
                        if pred_class < len(labels):
                            letter = labels[pred_class]
                        else:
                            letter = chr(65 + pred_class)
                    else:
                        letter = chr(65 + pred_class)
                    
                    self.prediction_history.append(letter)
                    smoothed_pred, smoothed_conf = self.get_smoothed_prediction()
                    
                    if smoothed_conf > 0.80:
                        if smoothed_pred == self.current_prediction:
                            self.stable_letter_count += 1
                        else:
                            self.current_prediction = smoothed_pred
                            self.stable_letter_count = 1
                        
                        if self.stable_letter_count >= self.stable_letter_threshold:
                            self.prev_hand_landmarks = hand_landmarks
                            return smoothed_pred, smoothed_conf
                    else:
                        self.stable_letter_count = 0
                        self.current_prediction = None
                except Exception as e:
                    print(f"⚠ Prediction error: {e}")
        
        self.prev_hand_landmarks = hand_landmarks
        return None, 0.0
    
    def run_recognition(self, camera_id: int = 0):
        """Run real-time recognition"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return
        
        print("\n" + "="*70)
        print("ASL LETTER RECOGNITION - ONNX Runtime (Simplified)")
        print("="*70)
        print("\nRecognizes all 26 letters!")
        print(" • 24 static letters: ONNX Runtime")
        print(" • J (pinky) & Z (index): Rule-based trajectory detection")
        print("\nControls:")
        print(" SPACE: Add current letter")
        print(" BACKSPACE: Delete last letter")
        print(" C: Clear word")
        print(" Q: Quit")
        print("\n" + "="*70 + "\n")
        
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                h, w = image.shape[:2]
                
                current_letter = None
                confidence = 0.0
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                        
                        current_letter, confidence = self.recognize_frame(hand_landmarks)
                    
                    # Draw trajectory
                    if self.motion_detection_active and not self.motion_trajectory.is_empty():
                        points = self.motion_trajectory.points
                        for i in range(1, len(points)):
                            x1, y1 = int(points[i-1][0] * w), int(points[i-1][1] * h)
                            x2, y2 = int(points[i][0] * w), int(points[i][1] * h)
                            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        finger_points = self.motion_trajectory.finger_tip_points
                        if finger_points:
                            color = (255, 0, 255) if self.motion_trajectory.finger_type == "pinky" else (0, 255, 255)
                            for i in range(1, len(finger_points)):
                                x1, y1 = int(finger_points[i-1][0] * w), int(finger_points[i-1][1] * h)
                                x2, y2 = int(finger_points[i][0] * w), int(finger_points[i][1] * h)
                                cv2.line(image, (x1, y1), (x2, y2), color, 4)
                else:
                    self.prediction_history.clear()
                    self.stable_letter_count = 0
                    self.consecutive_movements = 0
                    if not self.motion_detection_active:
                        self.motion_trajectory.clear()
                
                # Draw UI
                cv2.putText(image, "ONNX Runtime", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
                
                # Prediction display
                if current_letter:
                    pred_box_h = 150
                    cv2.rectangle(image, (w//2 - 100, 20), (w//2 + 100, 20 + pred_box_h), (0, 0, 0), -1)
                    cv2.rectangle(image, (w//2 - 100, 20), (w//2 + 100, 20 + pred_box_h), (0, 255, 0), 3)
                    cv2.putText(image, current_letter, (w//2 - 50, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                    cv2.putText(image, f"{confidence:.0%}", (w//2 - 40, 155),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Motion indicator
                if self.motion_detection_active:
                    cv2.circle(image, (w - 40, 40), 20, (0, 165, 255), -1)
                    cv2.putText(image, "TRACKING", (w - 160, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    traj_len = len(self.motion_trajectory.points)
                    cv2.putText(image, f"Points: {traj_len}", (w - 160, 75),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Word display
                word_box_y = h - 120
                cv2.rectangle(image, (10, word_box_y), (w - 10, h - 10), (0, 0, 0), -1)
                cv2.rectangle(image, (10, word_box_y), (w - 10, h - 10), (255, 255, 255), 2)
                cv2.putText(image, "WORD:", (20, word_box_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                word_display = self.detected_word if self.detected_word else "(empty)"
                cv2.putText(image, word_display, (20, word_box_y + 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # Controls
                cv2.putText(image, "SPACE:Add | BKSP:Del | C:Clear | Q:Quit",
                           (20, h - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('ASL Letter Recognition', image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if current_letter and confidence > 0.75:
                        self.add_letter_to_word(current_letter)
                elif key == 8:  # Backspace
                    if self.detected_word:
                        self.detected_word = self.detected_word[:-1]
                        print(f"Removed letter | Word: '{self.detected_word}'")
                elif key == ord('c'):
                    self.detected_word = ""
                    self.last_letter = None
                    print("Cleared word")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.detected_word:
            print(f"\n✓ Final word: {self.detected_word}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ASL Letter Recognition - ONNX Runtime (Simplified)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python asl_recognize.py                # Default settings
  python asl_recognize.py --camera 1     # Use camera 1
        """
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='asl_data',
        help='Directory containing ONNX model and metadata JSON (default: asl_data)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    args = parser.parse_args()
    
    recognizer = ASLLetterRecognizer(args.model_dir)
    recognizer.run_recognition(args.camera)


if __name__ == '__main__':
    main()
