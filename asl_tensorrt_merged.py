"""
ASL Fingerspelling Letter Recognition - TensorRT ONNX (Jetson Optimized)
- TensorRT-ONNX for 24 static letters (A-I, K-Y) - optimized for Jetson
- Rule-based trajectory detection for motion letters J and Z
- Text-to-Speech support with Piper
"""

from typing import Optional, Tuple
import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from collections import deque, Counter
import time
import io
import wave
import threading

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
    print("Fix: pip uninstall mediapipe -y && pip install mediapipe==0.10.9")
    exit(1)

# TensorRT + PyCUDA import (Jetson optimized)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("❌ TensorRT or PyCUDA not installed. Install TensorRT SDK and pycuda.")
    exit(1)

# Optional TTS support
try:
    from piper import PiperVoice
    from piper.voice import SynthesisConfig
    import pyaudio
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("⚠ Piper TTS not available. Install with: pip install piper-tts pyaudio")


class AudioPlayer:
    """Simple audio player for Piper TTS output"""

    def __init__(self):
        if PIPER_AVAILABLE:
            self.audio = pyaudio.PyAudio()
            self.stream = None

    def play_wav_data(self, wav_data):
        """Play WAV data from memory"""
        if not PIPER_AVAILABLE:
            return

        try:
            wav_io = io.BytesIO(wav_data)
            with wave.open(wav_io, 'rb') as wf:
                if self.stream:
                    self.stream.close()

                self.stream = self.audio.open(
                    format=self.audio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )

                data = wf.readframes(1024)
                while data:
                    self.stream.write(data)
                    data = wf.readframes(1024)
        except Exception as e:
            print(f"Audio playback error: {e}")

    def close(self):
        """Clean up audio resources"""
        if PIPER_AVAILABLE:
            if self.stream:
                self.stream.close()
            self.audio.terminate()


class TTSEngine:
    """Text-to-Speech engine using Piper"""

    def __init__(self, voice_path: str = None):
        self.voice = None
        self.audio_player = AudioPlayer()
        self.speaking = False
        self.enabled = True

        if not PIPER_AVAILABLE:
            print("❌ Piper TTS not available")
            self.enabled = False
            return

        if voice_path and Path(voice_path).exists():
            try:
                print(f"Loading voice: {voice_path}")
                self.voice = PiperVoice.load(voice_path)
                print("✓ Voice loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load voice: {e}")
                self.enabled = False
        else:
            print("⚠ No voice model provided. TTS disabled.")
            print("Download a voice with: python -m piper.download_voices en_US-lessac-medium")
            self.enabled = False

    def speak(self, text: str, async_mode: bool = True):
        """Speak text using Piper TTS"""
        if not self.enabled or not self.voice or not text:
            return

        if async_mode:
            thread = threading.Thread(target=self._speak_sync, args=(text,))
            thread.daemon = True
            thread.start()
        else:
            self._speak_sync(text)

    def _speak_sync(self, text: str):
        """Synchronous speech synthesis"""
        try:
            self.speaking = True
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                self.voice.synthesize_wav(text, wav_file)
            wav_data = wav_io.getvalue()
            self.audio_player.play_wav_data(wav_data)
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            self.speaking = False

    def toggle(self):
        """Toggle TTS on/off"""
        self.enabled = not self.enabled
        return self.enabled

    def close(self):
        """Clean up resources"""
        self.audio_player.close()


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


class TensorRTONNXRuntime:
    """Simple TensorRT runtime wrapper for an ONNX MLP model."""

    def __init__(self, onnx_path: Path, max_workspace_size: int = 1 << 28):
        self.onnx_path = onnx_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._build_engine(max_workspace_size)
        if self.engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self._allocate_buffers()

    def _build_engine(self, max_workspace_size: int):
        if not self.onnx_path.exists():
            print(f"⚠️ ONNX model not found: {self.onnx_path}")
            return None

        print(f"✓ Loading ONNX model for TensorRT: {self.onnx_path}")

        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(self.logger) as builder, \
             builder.create_network(explicit_batch) as network, \
             trt.OnnxParser(network, self.logger) as parser:

            with open(self.onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    print("❌ Failed to parse ONNX file:")
                    for i in range(parser.num_errors):
                        print(parser.get_error(i))
                    return None

            config = builder.create_builder_config()
            config.max_workspace_size = max_workspace_size

            engine = builder.build_engine(network, config)
            if engine is None:
                print("❌ Failed to build TensorRT engine from ONNX")
            return engine

    def _allocate_buffers(self):
        """Allocate host and device buffers for all bindings."""
        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            shape = self.engine.get_binding_shape(binding)

            size = int(trt.volume(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            io_desc = {
                'name': binding,
                'index': idx,
                'host': host_mem,
                'device': device_mem,
                'shape': shape,
            }

            if self.engine.binding_is_input(binding):
                self.inputs.append(io_desc)
            else:
                self.outputs.append(io_desc)

        if len(self.inputs) != 1 or len(self.outputs) != 1:
            print("⚠️ Expected 1 input and 1 output for MLP ONNX model.")

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference for a single batch of features.

        Expects input_array with shape (1, n_features) and dtype float32.
        Returns output as NumPy array with original engine output shape.
        """
        if not self.inputs or not self.outputs:
            raise RuntimeError("TensorRT buffers not properly initialized")

        inp = self.inputs[0]
        flat_input = input_array.astype(inp['host'].dtype).ravel()
        if flat_input.size != inp['host'].size:
            raise ValueError(
                f"Input size mismatch for TensorRT engine: "
                f"expected {inp['host'].size}, got {flat_input.size}"
            )

        np.copyto(inp['host'], flat_input)

        # Transfer to device and execute
        cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        self.context.execute_v2(self.bindings)

        # Copy outputs back
        out = self.outputs[0]
        cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()

        return np.array(out['host']).reshape(out['shape'])


class ASLLetterRecognizer:
    """Real-time ASL recognition with TensorRT-ONNX + Rule-based Motion"""
    
    def __init__(self, model_dir: str = "asl_data", voice_path: str = None):
        self.model_dir = Path(model_dir)

        # Initialize TTS
        self.tts = TTSEngine(voice_path)

        # Load TensorRT engine from ONNX model
        onnx_model_name = "asl_static_model.onnx"
        self.trt_runtime = self._load_trt_engine(onnx_model_name)
        self.static_metadata = self._load_metadata("asl_static_metadata.pkl")
        
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

        if self.trt_runtime is not None:
            print("✓ Loaded TensorRT-ONNX model (Jetson optimized)")
        print("✓ Using rule-based motion detection for J/Z")

    def _load_trt_engine(self, filename: str) -> Optional[TensorRTONNXRuntime]:
        """Load TensorRT engine from ONNX file in model_dir."""
        filepath = self.model_dir / filename

        try:
            runtime = TensorRTONNXRuntime(filepath)
            return runtime
        except Exception as e:
            print(f"❌ Error loading TensorRT-ONNX model: {e}")
            return None
    
    def _load_metadata(self, filename: str):
        """Load metadata pickle file"""
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            print(f"⚠️ Metadata not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            return None
    
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
            
            # Scale by hand size
            hand_size = np.sqrt(
                (landmarks[18*2] - landmarks[0])**2 + 
                (landmarks[18*2+1] - landmarks[1])**2
            ) + 1e-6
            landmarks /= hand_size
            
            # Apply training normalization
            if self.static_metadata and 'normalization_params' in self.static_metadata:
                norm_params = self.static_metadata['normalization_params']
                mean = np.asarray(norm_params['mean'], dtype=np.float32)
                std = np.asarray(norm_params['std'], dtype=np.float32)
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
                    else:
                        print(f"❌ Motion uncertain (conf={confidence:.0%})")
                else:
                    print(f"❌ Motion too small (dist={total_distance:.3f})")
                
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
        
        # Static letter recognition with TensorRT-ONNX
        if self.trt_runtime is not None and not self.motion_detection_active:
            static_features = self.extract_static_features(hand_landmarks)
            
            if static_features is not None:
                try:
                    # Predict with TensorRT-ONNX
                    features = static_features.reshape(1, -1).astype(np.float32)
                    prediction = self.trt_runtime.infer(features)[0]
                    
                    pred_class = np.argmax(prediction)
                    confidence = float(prediction[pred_class])
                    
                    # Get letter from label encoder
                    if self.static_metadata and 'label_encoder' in self.static_metadata:
                        label_encoder = self.static_metadata['label_encoder']
                        letter = label_encoder.classes_[pred_class]
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
        print("ASL LETTER RECOGNITION - TensorRT ONNX (JETSON OPTIMIZED)")
        print("="*70)
        print("\nRecognizes all 26 letters!")
        print(" • 24 static letters: TensorRT-ONNX MLP (Jetson optimized)")
        print(" • J (pinky) & Z (index): Rule-based trajectory detection")
        print("\nControls:")
        print(" SPACE: Add current letter")
        print(" BACKSPACE: Delete last letter")
        print(" C: Clear word")
        print(" ENTER: Speak word")
        print(" T: Toggle TTS")
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
                                x2