import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ================================
# CONFIGURATION
# ================================
WORDS = ["water", "help", "doctor", "school", "name"]  # KSL words
NUM_SAMPLES = 5                                        # repetitions per word
DURATION = 2                                           # seconds to record each repetition
FPS = 20                                               # approximate frames per second
SAVE_DIR = "ksl_video_data"                            # folder to save video-like sequences
os.makedirs(SAVE_DIR, exist_ok=True)

# ================================
# INITIALIZE MEDIAPIPE
# ================================
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,       # continuous tracking for video
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ================================
# OPEN WEBCAM
# ================================
cap = cv2.VideoCapture(0)

# ================================
# LOOP THROUGH WORDS
# ================================
for word in WORDS:
    print(f"\n👉 Sign: '{word}' — recording {NUM_SAMPLES} times, {DURATION}s each")

    for i in range(NUM_SAMPLES):
        print(f"  🔴 Repetition {i+1}/{NUM_SAMPLES} — get ready!")
        time.sleep(2)  # short delay to prepare

        sequence = []  # store all frames for this repetition
        start_time = time.time()

        while time.time() - start_time < DURATION:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                continue

            # Convert BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)

            # Draw green landmarks if hand detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                    # Extract 63D vector and store
                    vec = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark]).flatten()
                    sequence.append(vec)

            # Overlay text
            cv2.putText(frame,
                        f"Sign: {word} ({i+1}/{NUM_SAMPLES})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            # Show webcam feed
            cv2.imshow("KSL Video Recorder", frame)

            # Optional: stop early with ESC
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                print("⏹ Recording stopped by user")
                break

        # Save sequence as .npy if at least 1 frame was captured
        if len(sequence) > 0:
            sequence = np.array(sequence)  # shape: (num_frames, 63)
            np.save(f"{SAVE_DIR}/{word}_{i}.npy", sequence)
            print(f"  ✅ Saved {word}_{i}.npy — {len(sequence)} frames")
        else:
            print("  ❌ No hand frames captured — try again")

# ================================
# CLEANUP
# ================================
cap.release()
cv2.destroyAllWindows()
print("\n🎉 All words recorded successfully!")
