import os
import threading
import time
from collections import deque

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3

MODEL_FILE = "models/sign_model.joblib"
HAND_MODEL_PATH = "hand_landmarker.task"  # Ensure this file exists in your project directory

if not os.path.exists(MODEL_FILE):
    print("Model not found. Run train_model.py first.")
    exit()

if not os.path.exists(HAND_MODEL_PATH):
    print(
        f"Hand landmarker model not found. Download 'hand_landmarker.task' and save it at '{HAND_MODEL_PATH}'."
    )
    exit()

model = joblib.load(MODEL_FILE)

prediction_history = deque(maxlen=10)
current_text = ""
last_added = ""
last_added_time = 0


def normalize_landmarks(hand_landmarks):
    points = []
    wrist = hand_landmarks[0]

    for landmark in hand_landmarks:
        points.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z,
        ])

    max_value = max(abs(value) for value in points)

    if max_value == 0:
        return points

    return [value / max_value for value in points]


def most_common_prediction(history):
    if not history:
        return ""

    return max(set(history), key=history.count)


def speak_text(text):
    """Speaks text in a separate background thread to prevent GUI freezing
    and pyttsx3 loop locking issues.
    """
    if not text.strip():
        return

    def _say():
        try:
            tts = pyttsx3.init()
            tts.say(text)
            tts.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

    threading.Thread(target=_say, daemon=True).start()


# Initialize Modern MediaPipe HandLandmarker Tasks API
base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
detector = vision.HandLandmarker.create_from_options(options)

# Connections array for drawing hand skeleton connections manually
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
]


def draw_landmarks_on_image(image, landmarks):
    """Draws keypoints and skeleton connections using standard OpenCV."""
    h, w, _ = image.shape
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Draw lines connecting landmarks
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(image, coords[start_idx], coords[end_idx], (0, 255, 0), 2)

    # Draw landmark joints
    for coord in coords:
        cv2.circle(image, coord, 5, (0, 0, 255), -1)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

WINDOW_NAME = "Sign-o-lator"

# Set up full screen window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Controls:")
print("ENTER = add character")
print("S = speak current text")
print("BACKSPACE = delete last character")
print("SPACE = add space")
print("C = clear text")
print("F = toggle full screen")
print("Q / ESC = quit")

is_fullscreen = True

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image format required by Tasks API
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    prediction = ""

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        features = normalize_landmarks(hand_landmarks)

        prediction = str(model.predict([features])[0])
        prediction_history.append(prediction)

        stable_prediction = most_common_prediction(list(prediction_history))

        # Draw hand keypoints and connections on frame
        draw_landmarks_on_image(frame, hand_landmarks)

        cv2.putText(
            frame,
            f"Prediction: {stable_prediction}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )
    else:
        stable_prediction = ""
        prediction_history.clear()

        cv2.putText(
            frame,
            "No hand detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

    cv2.putText(
        frame,
        f"Text: {current_text}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    # --- Two-Line Instructions at the bottom ---
    h = frame.shape[0]

    cv2.putText(
        frame,
        "ENTER = add | SPACE = space | BACKSPACE = delete | C = clear",
        (20, h - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "S = speak | F = toggle fullscreen | Q / ESC = quit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2,
    )

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:  # Q or ESC
        break

    elif key == ord("f"):  # Toggle Fullscreen
        is_fullscreen = not is_fullscreen
        prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)

    elif key == 13:  # ENTER
        now = time.time()

        if stable_prediction and not (
            stable_prediction == last_added and now - last_added_time < 1
        ):
            current_text += stable_prediction
            last_added = stable_prediction
            last_added_time = now

    elif key == 8:  # BACKSPACE
        current_text = current_text[:-1]

    elif key == 32:  # SPACE
        current_text += " "

    elif key == ord("c"):
        current_text = ""

    elif key == ord("s"):
        speak_text(current_text)

cap.release()
cv2.destroyAllWindows()
detector.close()