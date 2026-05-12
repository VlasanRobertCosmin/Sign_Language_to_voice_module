import os
import numpy as np
import cv2
import time
from collections import deque
import math
import socket
import threading
import json

try:
    from ctypes import *
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass

SPEECH_AVAILABLE = False
try:
    import speech_recognition as sr
    sr.Microphone()
    SPEECH_AVAILABLE = True
    print("Speech recognition: ENABLED")
except Exception:
    print("Speech recognition: DISABLED")
    print("Use ENTER key to type text instead")

CACHE_FILE = "data/asl_signs_cache.npz"

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
FPS = 30
ANIMATION_SPEED = 0.5

UDP_ENABLED = True
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
UDP_BUFFER_SIZE = 8192
UDP_MIN_CONFIDENCE = 0.30

SKIN_COLOR = (180, 160, 150)
SKIN_DARK = (140, 120, 110)
SHIRT_COLOR = (180, 100, 80)
HAIR_COLOR = (50, 35, 25)
BG_COLOR = (45, 42, 38)


class SignDatabase:
    def __init__(self, cache_file):
        self.signs = {}
        self.classes = []
        self.load_data(cache_file)

    def interpolate_frames(self, frames, factor=3):
        if len(frames) < 2:
            return frames

        interpolated = []
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])
            for j in range(1, factor):
                t = j / factor
                interp = frames[i] * (1 - t) + frames[i + 1] * t
                interpolated.append(interp)

        interpolated.append(frames[-1])
        return np.array(interpolated)

    def load_data(self, cache_file):
        if not os.path.exists(cache_file):
            print(f"Cache not found: {cache_file}")
            self._create_demo()
            return

        print(f"Loading {cache_file}...")
        cache = np.load(cache_file, allow_pickle=True)
        X = cache["X"]
        y = cache["y"]

        self.classes = sorted(list(set(y)))

        for i, label in enumerate(y):
            label = str(label).lower().strip()

            if label not in self.signs:
                seq = X[i]
                valid_frames = []

                for f in range(len(seq)):
                    if np.abs(seq[f]).sum() > 0.1:
                        valid_frames.append(seq[f])

                if valid_frames:
                    frames = np.array(valid_frames)
                    frames = self.interpolate_frames(frames, factor=3)
                    self.signs[label] = frames
                else:
                    self.signs[label] = seq[:10]

        print(f"Loaded {len(self.signs)} signs")

    def _create_demo(self):
        for word in ["hello", "thank", "you"]:
            self.signs[word] = np.random.rand(30, 225).astype(np.float32) * 0.5
            self.classes.append(word)

    def get_sign(self, word):
        return self.signs.get(word.lower().strip())

    def find_similar(self, word):
        word = word.lower().strip()

        if word in self.signs:
            return [word]

        matches = [s for s in self.classes if word in s or s in word]

        if not matches and len(word) >= 2:
            matches = [s for s in self.classes if s[:2] == word[:2]]

        return matches[:3]


class AvatarRenderer:
    def __init__(self, width=700, height=650):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2 + 50
        self.scale = 350

        self.prev_landmarks = None
        self.smooth = 0.25

        self.finger_chains = [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20],
        ]

    def smooth_landmarks(self, landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks.copy()
            return landmarks

        smoothed = self.smooth * landmarks + (1 - self.smooth) * self.prev_landmarks
        self.prev_landmarks = smoothed.copy()
        return smoothed

    def draw(self, frame, raw_landmarks):
        landmarks = self.smooth_landmarks(raw_landmarks)

        left_hand = landmarks[0:63].reshape(21, 3)
        right_hand = landmarks[63:126].reshape(21, 3)
        pose = landmarks[126:225].reshape(33, 3)

        left_visible = np.abs(left_hand).sum() > 0.5
        right_visible = np.abs(right_hand).sum() > 0.5

        self.draw_body(frame, pose)
        self.draw_arm_and_hand(frame, pose, right_hand, is_left=False, hand_visible=right_visible)
        self.draw_arm_and_hand(frame, pose, left_hand, is_left=True, hand_visible=left_visible)
        self.draw_head(frame)

        return frame

    def pose_to_screen(self, pose_point):
        x = pose_point[0]
        y = pose_point[1]

        screen_x = self.center_x + (x - 0.5) * self.scale
        screen_y = self.center_y + (y - 0.5) * self.scale * 0.8

        return int(screen_x), int(screen_y)

    def draw_body(self, frame, pose):
        left_shoulder = self.pose_to_screen(pose[11])
        right_shoulder = self.pose_to_screen(pose[12])

        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) // 2,
            (left_shoulder[1] + right_shoulder[1]) // 2
        )

        body_width = abs(left_shoulder[0] - right_shoulder[0]) + 40
        body_top = shoulder_center[1]
        body_bottom = shoulder_center[1] + 200

        pts = np.array([
            [shoulder_center[0] - body_width // 2, body_top],
            [shoulder_center[0] + body_width // 2, body_top],
            [shoulder_center[0] + body_width // 2 + 30, body_bottom],
            [shoulder_center[0] - body_width // 2 - 30, body_bottom],
        ], np.int32)

        cv2.fillPoly(frame, [pts], SHIRT_COLOR)
        cv2.polylines(frame, [pts], True, (120, 70, 50), 3)

        neck_top = shoulder_center[1] - 30
        cv2.rectangle(
            frame,
            (shoulder_center[0] - 25, neck_top),
            (shoulder_center[0] + 25, body_top + 10),
            SKIN_COLOR,
            -1
        )

        self.head_center = (shoulder_center[0], neck_top - 60)

    def draw_head(self, frame):
        cx, cy = self.head_center
        radius = 55

        cv2.circle(frame, (cx, cy), radius, SKIN_COLOR, -1)
        cv2.circle(frame, (cx, cy), radius, SKIN_DARK, 2)

        hair_pts = []
        for angle in range(160, 381, 8):
            rad = math.radians(angle)
            x = cx + int((radius + 5) * math.cos(rad))
            y = cy + int((radius + 5) * math.sin(rad))
            hair_pts.append([x, y])

        if hair_pts:
            cv2.fillPoly(frame, [np.array(hair_pts, np.int32)], HAIR_COLOR)

        eye_y = cy - 5
        eye_spacing = 22

        cv2.ellipse(frame, (cx - eye_spacing, eye_y), (12, 8), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (cx + eye_spacing, eye_y), (12, 8), 0, 0, 360, (255, 255, 255), -1)

        cv2.circle(frame, (cx - eye_spacing, eye_y), 5, (40, 30, 20), -1)
        cv2.circle(frame, (cx + eye_spacing, eye_y), 5, (40, 30, 20), -1)

        cv2.line(frame, (cx - eye_spacing - 12, eye_y - 15),
                 (cx - eye_spacing + 10, eye_y - 12), HAIR_COLOR, 3)
        cv2.line(frame, (cx + eye_spacing - 10, eye_y - 12),
                 (cx + eye_spacing + 12, eye_y - 15), HAIR_COLOR, 3)

        cv2.line(frame, (cx, cy + 5), (cx + 8, cy + 20), SKIN_DARK, 2)
        cv2.ellipse(frame, (cx, cy + 35), (15, 8), 0, 10, 170, (120, 80, 80), 2)

    def draw_arm_and_hand(self, frame, pose, hand, is_left, hand_visible):
        shoulder_idx = 11 if is_left else 12
        elbow_idx = 13 if is_left else 14
        wrist_idx = 15 if is_left else 16

        shoulder = self.pose_to_screen(pose[shoulder_idx])
        elbow = self.pose_to_screen(pose[elbow_idx])
        wrist = self.pose_to_screen(pose[wrist_idx])

        wrist = (
            max(50, min(self.width - 50, wrist[0])),
            max(50, min(self.height - 50, wrist[1]))
        )
        elbow = (
            max(50, min(self.width - 50, elbow[0])),
            max(50, min(self.height - 50, elbow[1]))
        )

        cv2.line(frame, shoulder, elbow, SKIN_COLOR, 22)
        cv2.line(frame, shoulder, elbow, SKIN_DARK, 2)
        cv2.line(frame, elbow, wrist, SKIN_COLOR, 18)
        cv2.line(frame, elbow, wrist, SKIN_DARK, 2)

        cv2.circle(frame, elbow, 12, SKIN_COLOR, -1)
        cv2.circle(frame, elbow, 12, SKIN_DARK, 2)

        if hand_visible:
            self.draw_hand(frame, hand)
        else:
            cv2.circle(frame, wrist, 25, SKIN_COLOR, -1)
            cv2.circle(frame, wrist, 25, SKIN_DARK, 2)

    def draw_hand(self, frame, hand_landmarks):
        points = []

        for i in range(21):
            x = hand_landmarks[i][0]
            y = hand_landmarks[i][1]

            screen_x = self.center_x + (x - 0.5) * self.scale * 1.5
            screen_y = self.center_y + (y - 0.5) * self.scale * 1.2

            points.append((int(screen_x), int(screen_y)))

        palm_pts = [points[0], points[5], points[9], points[13], points[17]]

        cv2.fillPoly(frame, [np.array(palm_pts, np.int32)], SKIN_COLOR)
        cv2.polylines(frame, [np.array(palm_pts, np.int32)], True, SKIN_DARK, 2)

        for finger in self.finger_chains:
            finger_pts = [points[i] for i in finger]

            for i in range(len(finger_pts) - 1):
                thickness = max(5, 14 - i * 3)
                cv2.line(frame, finger_pts[i], finger_pts[i + 1], SKIN_COLOR, thickness)
                cv2.line(frame, finger_pts[i], finger_pts[i + 1], SKIN_DARK, 2)

            for i, pt in enumerate(finger_pts):
                radius = max(4, 10 - i * 2)
                cv2.circle(frame, pt, radius, SKIN_COLOR, -1)
                cv2.circle(frame, pt, radius, SKIN_DARK, 1)

            cv2.circle(frame, finger_pts[-1], 6, SKIN_DARK, 2)

        cv2.circle(frame, points[0], 12, SKIN_COLOR, -1)


class SpeechRecognizer:
    def __init__(self):
        self.recognizer = None

        if SPEECH_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

    def listen(self, timeout=5):
        if not self.recognizer:
            return None, "No microphone"

        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)

            text = self.recognizer.recognize_google(audio)
            return text.lower(), None

        except sr.WaitTimeoutError:
            return None, "No speech detected"
        except sr.UnknownValueError:
            return None, "Could not understand"
        except Exception as e:
            return None, str(e)


class UDPReceiver:
    def __init__(self, ip="0.0.0.0", port=5005):
        self.ip = ip
        self.port = port
        self.sock = None
        self.running = False
        self.thread = None
        self.callback = None

    def start(self, callback):
        self.callback = callback
        self.running = True

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(0.5)

        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

        print(f"UDP VR Receiver listening on {self.ip}:{self.port}")

    def stop(self):
        self.running = False

        if self.sock:
            try:
                self.sock.close()
            except:
                pass

    def _listen_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(UDP_BUFFER_SIZE)
                message = data.decode("utf-8").strip()

                text_input = None
                confidence = 1.0

                try:
                    payload = json.loads(message)

                    message_type = payload.get("type", "text")
                    confidence = float(payload.get("confidence", 1.0))

                    if message_type == "text":
                        text_input = payload.get("text", "")
                    elif message_type == "sign":
                        text_input = payload.get("label", "")
                    else:
                        text_input = payload.get("text", "") or payload.get("label", "")

                except json.JSONDecodeError:
                    text_input = message

                if not text_input:
                    continue

                if confidence < UDP_MIN_CONFIDENCE:
                    continue

                if self.callback:
                    self.callback(text_input, confidence)

            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                print(f"UDP receive error: {e}")


class SpeechToSignApp:
    def __init__(self):
        self.db = SignDatabase(CACHE_FILE)
        self.avatar = AvatarRenderer(WINDOW_WIDTH - 280, WINDOW_HEIGHT)
        self.speech = SpeechRecognizer()
        self.udp = UDPReceiver(UDP_IP, UDP_PORT) if UDP_ENABLED else None

        self.current_word = ""
        self.current_sign = None
        self.frame_idx = 0
        self.is_playing = False
        self.word_queue = deque()
        self.status = "Ready"

        self.neutral = self._create_neutral()
        self.current_landmarks = self.neutral.copy()

        self.last_udp_text = ""
        self.last_udp_time = 0.0

    def _create_neutral(self):
        lm = np.zeros(225, dtype=np.float32)
        pose_start = 126

        lm[pose_start + 11 * 3] = 0.7
        lm[pose_start + 11 * 3 + 1] = 0.4

        lm[pose_start + 12 * 3] = 0.3
        lm[pose_start + 12 * 3 + 1] = 0.4

        lm[pose_start + 13 * 3] = 0.8
        lm[pose_start + 13 * 3 + 1] = 0.6

        lm[pose_start + 14 * 3] = 0.2
        lm[pose_start + 14 * 3 + 1] = 0.6

        lm[pose_start + 15 * 3] = 0.85
        lm[pose_start + 15 * 3 + 1] = 0.8

        lm[pose_start + 16 * 3] = 0.15
        lm[pose_start + 16 * 3 + 1] = 0.8

        return lm

    def enqueue_word(self, word, source="text"):
        word = word.lower().strip()

        if not word:
            return False

        if self.db.get_sign(word) is not None:
            self.word_queue.append(word)
            print(f"✓ [{source}] Found: {word}")
            return True

        similar = self.db.find_similar(word)

        if similar:
            self.word_queue.append(similar[0])
            print(f"~ [{source}] Using '{similar[0]}' for '{word}'")
            return True

        print(f"✗ [{source}] Not found: {word}")
        return False

    def process_text(self, text, source="text"):
        words = text.lower().split()
        found = False

        for word in words:
            word = "".join(c for c in word if c.isalpha())

            if not word:
                continue

            ok = self.enqueue_word(word, source=source)
            found = found or ok

        return found

    def handle_udp_input(self, text_input, confidence=1.0):
        now = time.time()

        clean_text = text_input.lower().strip()

        if clean_text == self.last_udp_text and (now - self.last_udp_time) < 1.0:
            return

        self.last_udp_text = clean_text
        self.last_udp_time = now

        print(f"\nVR/UDP received: '{clean_text}' ({confidence * 100:.1f}%)")

        added = self.process_text(clean_text, source="vr-client")

        if added:
            self.status = f"VR input: {clean_text[:20].upper()}"

            if not self.is_playing:
                self.play_next()

    def play_next(self):
        if self.word_queue:
            self.current_word = self.word_queue.popleft()
            self.current_sign = self.db.get_sign(self.current_word)
            self.frame_idx = 0
            self.is_playing = True
            self.status = f"Signing: {self.current_word.upper()}"

            print(f"Playing: {self.current_word} ({len(self.current_sign)} frames)")

    def update(self):
        if self.is_playing and self.current_sign is not None:
            if self.frame_idx < len(self.current_sign):
                self.current_landmarks = self.current_sign[self.frame_idx]
                self.frame_idx += 1
            else:
                self.is_playing = False

                if self.word_queue:
                    self.play_next()
                else:
                    self.status = "Ready"
                    self.current_word = ""
        else:
            self.current_landmarks = 0.9 * self.current_landmarks + 0.1 * self.neutral

    def draw_ui(self, frame):
        h, w = frame.shape[:2]
        panel_x = w - 260

        cv2.rectangle(frame, (panel_x, 0), (w, h), (35, 32, 30), -1)
        cv2.line(frame, (panel_x, 0), (panel_x, h), (60, 55, 50), 2)

        cv2.putText(frame, "TEXT TO SIGN", (panel_x + 15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        cv2.line(frame, (panel_x + 15, 50), (w - 15, 50), (60, 55, 50), 1)

        cv2.putText(frame, "Now signing:", (panel_x + 15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        word = self.current_word.upper() if self.current_word else "---"

        cv2.putText(frame, word, (panel_x + 15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 180), 2)

        if self.is_playing and self.current_sign is not None:
            progress = self.frame_idx / len(self.current_sign)
            bar_w = 220

            cv2.rectangle(frame, (panel_x + 15, 145),
                          (panel_x + 15 + bar_w, 160), (50, 50, 50), -1)

            cv2.rectangle(frame, (panel_x + 15, 145),
                          (panel_x + 15 + int(bar_w * progress), 160),
                          (100, 255, 180), -1)

        cv2.putText(frame, f"Queue ({len(self.word_queue)}):", (panel_x + 15, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        y = 225

        for item in list(self.word_queue)[:5]:
            cv2.putText(frame, f"- {item}", (panel_x + 25, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            y += 22

        cv2.line(frame, (panel_x + 15, 350), (w - 15, 350), (60, 55, 50), 1)

        controls = [
            "SPACE  Voice input",
            "ENTER  Type text",
            "R      Replay",
            "C      Clear",
            "L      List signs",
            "Q      Quit",
            "",
            f"UDP port: {UDP_PORT}"
        ]

        y = 380

        for ctrl in controls:
            cv2.putText(frame, ctrl, (panel_x + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
            y += 25

        cv2.putText(frame, self.status[:30], (panel_x + 15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 180), 1)

    def run(self):
        print("\n" + "=" * 55)
        print("VOICE / TEXT TO SIGN - VR SERVER")
        print("=" * 55)
        print(f"Signs loaded: {len(self.db.classes)}")
        print(f"UDP listening on: {UDP_IP}:{UDP_PORT}")
        print("Use Unity or vr_text_client_test.py to send text.")
        print("=" * 55 + "\n")

        if self.udp:
            self.udp.start(self.handle_udp_input)

        cv2.namedWindow("Speech to Sign VR Server", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Speech to Sign VR Server", WINDOW_WIDTH, WINDOW_HEIGHT)

        frame_time = 1.0 / FPS
        last_update = time.time()

        while True:
            frame = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), BG_COLOR, dtype=np.uint8)

            now = time.time()

            if now - last_update >= frame_time / ANIMATION_SPEED:
                self.update()
                last_update = now

            avatar_area = frame[:, :WINDOW_WIDTH - 260]
            self.avatar.draw(avatar_area, self.current_landmarks)
            self.draw_ui(frame)

            cv2.imshow("Speech to Sign VR Server", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord(" "):
                self.status = "Listening..."
                text, err = self.speech.listen()

                if text:
                    print(f"\nHeard: '{text}'")

                    if self.process_text(text, source="speech") and not self.is_playing:
                        self.play_next()
                else:
                    self.status = err or "Try again"

            elif key == 13:
                print("\nType text: ", end="", flush=True)
                text = input()

                if text:
                    print(f"Processing: '{text}'")

                    if self.process_text(text, source="typed") and not self.is_playing:
                        self.play_next()

            elif key == ord("r") and self.current_sign is not None:
                self.frame_idx = 0
                self.is_playing = True

            elif key == ord("c"):
                self.word_queue.clear()
                self.is_playing = False
                self.current_word = ""
                self.status = "Cleared"

            elif key == ord("l"):
                print("\nAvailable signs:")

                for i, sign in enumerate(sorted(self.db.classes)):
                    print(f"{sign:15}", end="")

                    if (i + 1) % 6 == 0:
                        print()

                print("\n")

            if not self.is_playing and self.word_queue:
                self.play_next()

        if self.udp:
            self.udp.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = SpeechToSignApp()
    app.run()