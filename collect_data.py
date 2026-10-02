import csv
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_FILE = "data/sign_landmarks_cleaned.csv"
MODEL_PATH = "hand_landmarker.task"

# Allowed labels for dataset collection
ALLOWED_LABELS = set("ABCDEFGHIKLMNOPQRSTUVWXY123")

os.makedirs("data", exist_ok=True)

# Complete hand landmark connections (matching standard MediaPipe skeleton)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles
    (5, 9), (9, 13), (13, 17)
]


def draw_hand_landmarks(image, landmarks):
    """Draw landmarks and connection lines on the OpenCV frame."""
    h, w, _ = image.shape

    # Draw connection lines
    for start_idx, end_idx in HAND_CONNECTIONS:
        pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

    # Draw landmark points
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)


def normalize_landmarks(hand_landmarks):
    """Normalize landmarks relative to wrist (index 0) and max distance scale."""
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


def save_sample(label, features):
    """Append label and feature vector to dataset CSV."""
    file_exists = os.path.exists(DATA_FILE)

    with open(DATA_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            header = ["label"] + [f"feature_{i}" for i in range(len(features))]
            writer.writerow(header)

        writer.writerow([label] + features)


# Initialize Hand Landmarker for VIDEO stream tracking
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

print("Controls:")
print("Press any allowed letter/number key to save a sample.")
print("Press / to quit.")

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()

        if not success:
            print("Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Pass current timestamp in milliseconds for VIDEO mode tracking
        frame_timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        current_features = None

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            current_features = normalize_landmarks(hand_landmarks)
            draw_hand_landmarks(frame, hand_landmarks)

        cv2.putText(
            frame,
            "Show sign, press character to save. / = quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Collect Sign Data", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("/"):
            break

        # Check key range before calling chr()
        if 0 <= key < 128:
            label = chr(key).upper()
            if label in ALLOWED_LABELS:
                if current_features is not None:
                    save_sample(label, current_features)
                    print(f"Saved sample for {label}")
                else:
                    print("No hand detected. Try again.")

cap.release()
cv2.destroyAllWindows()