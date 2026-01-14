import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ================================
# CONFIGURATION
# ================================
WORDS = [ "this"]
NUM_SAMPLES = 50
DURATION = 4
FPS = 30
SAVE_DIR = "ksl_video_data_this"

os.makedirs(SAVE_DIR, exist_ok=True)

# ================================
# MEDIAPIPE HANDS
# ================================
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ================================
# CAMERA SETUP
# ================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

# ================================
# MAIN LOOP
# ================================
for word in WORDS:
    print(f"\n👉 Sign: '{word}'")

    for i in range(NUM_SAMPLES):
        sequence = []

        # ---------- GET READY ----------
        ready_start = time.time()
        while time.time() - ready_start < 2:
            ret, frame = cap.read()
            if not ret:
                continue

            cv2.putText(
                frame,
                "GET READY",
                (400, 360),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 255),
                4
            )

            cv2.imshow("KSL Recorder", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # ---------- RECORDING ----------
        record_start = time.time()
        while time.time() - record_start < DURATION:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)

            if results.multi_hand_landmarks and results.multi_handedness:
                left_hand = None
                right_hand = None

                for hand_lm, hand_info in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    label = hand_info.classification[0].label

                    vec = np.array(
                        [[p.x, p.y, p.z] for p in hand_lm.landmark]
                    ).flatten()

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

                sequence.append(np.concatenate([left_hand, right_hand]))

            # UI text
            cv2.putText(
                frame,
                "RECORDING",
                (400, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                4
            )

            cv2.putText(
                frame,
                f"{word} ({i+1}/{NUM_SAMPLES})",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("KSL Recorder", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(1 / FPS)

        # ---------- SAVE ----------
        if len(sequence) > 0:
            sequence = np.array(sequence)
            np.save(f"{SAVE_DIR}/{word}_{i}.npy", sequence)
            print(f"  ✅ Saved {word}_{i}.npy ({sequence.shape})")
        else:
            print("  ❌ No hand detected")

# ================================
# CLEANUP
# ================================
cap.release()
cv2.destroyAllWindows()
print("\n🎉 Recording complete!")
