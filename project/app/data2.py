import cv2
import mediapipe as mp
import numpy as np
import os

# ================================
# CONFIGURATION
# ================================
WORDS = ["water", "help", "doctor", "school", "name"]  # the KSL words you want to capture
NUM_SAMPLES = 10                                        # how many samples per word
SAVE_DIR = "ksl_data"                                   # folder to save .npy files
os.makedirs(SAVE_DIR, exist_ok=True)                    # create folder if it doesn't exist

# ================================
# INITIALIZE MEDIAPIPE
# ================================
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,         # each frame treated independently
    max_num_hands=1,                # only track one hand
    min_detection_confidence=0.6    # threshold for hand detection
)
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ================================
# OPEN WEBCAM
# ================================
cap = cv2.VideoCapture(0)

# ================================
# LOOP THROUGH EACH WORD
# ================================
for word in WORDS:
    print(f"\n👉 Sign: '{word}' — Press SPACE to capture (do {NUM_SAMPLES} times)")
    
    for i in range(NUM_SAMPLES):
        while True:
            # 1️⃣ Capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame. Check webcam.")
                continue

            # 2️⃣ Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)

            # 3️⃣ Draw green hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            # 4️⃣ Display current word + sample count on top
            cv2.putText(frame,
                        f"Sign: {word} ({i+1}/{NUM_SAMPLES})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),  # green text
                        2)

            # 5️⃣ Show frame in a window
            cv2.imshow("KSL Recorder", frame)

            # 6️⃣ Capture frame when SPACE is pressed
            key = cv2.waitKey(1)
            if key == 32:  # SPACE key
                break

        # ================================
        # PROCESS FRAME AND SAVE LANDMARKS
        # ================================
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]  # first hand
            # Convert landmarks to 1D vector: 21 points × 3 coordinates = 63D
            vec = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            # Save as .npy file
            np.save(f"{SAVE_DIR}/{word}_{i}.npy", vec)
            print(f"  ✅ Saved {word}_{i}.npy")
        else:
            print("  ❌ No hand detected — try again")

# ================================
# CLEANUP
# ================================
cap.release()
cv2.destroyAllWindows()
print("\n🎉 All words captured successfully!")
