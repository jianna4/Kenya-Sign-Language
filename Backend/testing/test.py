import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

# ==============================
# CONFIGURATION
# ==============================
MODEL_PATH = r"F:\projects\KSL\test\KSL_model_a.keras"
LABELS = ["A", "this", "MOTHER", "MY", "is"]
MAX_FRAMES = 40
GESTURE_COOLDOWN = 5.0
CONFIDENCE_THRESHOLD = 0.5
FPS = 30

# ==============================
# LOAD MODEL
# ==============================
model = load_model(MODEL_PATH)
print("✅ Model loaded!")

# ==============================
# MEDIAPIPE HANDS
# ==============================
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ==============================
# CAMERA SETUP
# ==============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

# ==============================
# STATE VARIABLES
# ==============================
last_prediction = ""
last_confidence = 0.0
last_prediction_time = 0
sequence = []
is_capturing = False
hands_detected_count = 0  # To avoid false triggers

try:
    while True:
        current_time = time.time()
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)

        left_hand = None
        right_hand = None
        hands_present = False

        if results.multi_hand_landmarks and results.multi_handedness:
            hands_present = True
            for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_info.classification[0].label
                vec = np.array([[p.x, p.y, p.z] for p in hand_lm.landmark]).flatten()

                if label == "Left":
                    left_hand = vec
                else:
                    right_hand = vec

                mp_draw.draw_landmarks(
                    frame,
                    hand_lm,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        if left_hand is None:
            left_hand = np.zeros(63)
        if right_hand is None:
            right_hand = np.zeros(63)

        combined = np.concatenate([left_hand, right_hand])

        # ==============================
        # COOLDOWN PHASE
        # ==============================
        if current_time - last_prediction_time < GESTURE_COOLDOWN:
            cv2.putText(frame, "WAITING...", (420, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
            if last_prediction:
                txt = f"Last: {last_prediction} ({last_confidence:.2f})"
                cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("KSL Live Recognition", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # ==============================
        # READY OR CAPTURING?
        # ==============================
        if not is_capturing:
            # In READY state — look for sustained hand presence
            if hands_present:
                hands_detected_count += 1
            else:
                hands_detected_count = 0

            # Require 2 consecutive frames with hands to start
            if hands_detected_count >= 2:
                is_capturing = True
                sequence = []
                hands_detected_count = 0
                print("▶️ Auto-start: Hands detected — capturing...")

            # Show READY message
            cv2.putText(frame, "READY! Show your gesture", (320, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            if last_prediction:
                cv2.putText(frame, f"Last: {last_prediction}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        else:
            # CAPTURING
            sequence.append(combined)
            cv2.putText(frame, "RECORDING...", (420, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, f"Frame: {len(sequence)}/{MAX_FRAMES}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if len(sequence) >= MAX_FRAMES:
                is_capturing = False
                seq_input = np.expand_dims(np.array(sequence[:MAX_FRAMES]), axis=0)
                preds = model.predict(seq_input, verbose=0)[0]
                confidence = np.max(preds)
                pred_label = LABELS[np.argmax(preds)]

                if confidence >= CONFIDENCE_THRESHOLD:
                    last_prediction = pred_label
                    last_confidence = confidence
                    print(f"✅ Predicted: {pred_label} | Confidence: {confidence:.2f}")
                else:
                    last_prediction = "Uncertain"
                    last_confidence = confidence
                    print(f"⚠️ Low confidence: {confidence:.2f}")

                last_prediction_time = time.time()

        cv2.imshow("KSL Live Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

        # Optional: maintain ~30 FPS
        time.sleep(max(0, 1/FPS - (time.time() - current_time)))

except Exception as e:
    print(f"Error: {e}")

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()
print("\n👋 Session ended.")