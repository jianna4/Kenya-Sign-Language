import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Initialize MediaPipe for version 0.10.7
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

WORDS = ["this"]
NUM_SAMPLES = 210
DURATION = 4
FPS = 30
SAVE_DIR = "this"

os.makedirs(SAVE_DIR, exist_ok=True) 

# CAP_DSHOW improves camera stability on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Webcam not accessible")
    exit()

print("Press ESC to exit at any time")
print(f"Starting recording for word: {WORDS[0]}")
print(f"Total samples to collect: {NUM_SAMPLES}")
print(f"Each sample will be {DURATION} seconds at {FPS} FPS = {DURATION * FPS} frames")

for word in WORDS:
    print(f"\n📝 Sign: '{word}'")

    for i in range(NUM_SAMPLES):
        sequence = []
        frame_count = 0
        expected_frames = DURATION * FPS

        # ---------- GET READY ----------
        ready_start = time.time()
        while time.time() - ready_start < 2:
            ret, frame = cap.read()
            if not ret:
                continue

            # Display ready screen
            cv2.putText(
                frame,
                f"Sample {i+1}/{NUM_SAMPLES}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            cv2.putText(
                frame,
                "GET READY",
                (400, 360),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 255),
                4
            )
            
            cv2.putText(
                frame,
                f"Perform: {word}",
                (400, 420),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.imshow("KSL Recorder", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                cap.release()
                cv2.destroyAllWindows()
                exit()

        print(f"  Recording sample {i+1}...", end="", flush=True)

        # ---------- RECORDING ----------
        record_start = time.time()
        last_frame_time = time.time()
        
        while time.time() - record_start < DURATION:
            ret, frame = cap.read()
            if not ret:
                continue

            # Control frame rate
            current_time = time.time()
            if current_time - last_frame_time < 1/FPS:
                time.sleep(0.001)
                continue
            last_frame_time = current_time

            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Process hand landmarks
            if results.multi_hand_landmarks and results.multi_handedness:
                left_hand = None
                right_hand = None

                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    # Get hand label (Left or Right)
                    hand_label = handedness.classification[0].label

                    # Extract landmarks (x, y, z for 21 landmarks = 63 values)
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    landmarks_array = np.array(landmarks)

                    if hand_label == "Left":
                        left_hand = landmarks_array
                    else:
                        right_hand = landmarks_array

                    # Draw landmarks on frame
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                # Fill in missing hand with zeros
                if left_hand is None:
                    left_hand = np.zeros(63)
                if right_hand is None:
                    right_hand = np.zeros(63)

                # Combine both hands (126 values total)
                combined_landmarks = np.concatenate([left_hand, right_hand])
                sequence.append(combined_landmarks)
                frame_count += 1
            else:
                # If no hands detected, append zeros for both hands
                sequence.append(np.zeros(126))
                frame_count += 1

            # Add UI text
            cv2.putText(
                frame,
                "🔴 RECORDING",
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
            
            # Show progress
            progress = int((frame_count / expected_frames) * 100)
            cv2.putText(
                frame,
                f"Frames: {frame_count}/{expected_frames} ({progress}%)",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Show hand detection status
            if results.multi_hand_landmarks:
                cv2.putText(
                    frame,
                    "✓ Hands detected",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "✗ No hands detected",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("KSL Recorder", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # ---------- SAVE ----------
        if len(sequence) > 0:
            sequence = np.array(sequence)
            
            # Ensure we have the right number of frames
            if len(sequence) < expected_frames:
                # Pad with zeros if we missed some frames
                padding = np.zeros((expected_frames - len(sequence), 126))
                sequence = np.vstack([sequence, padding])
                print(f" padded {expected_frames - len(sequence)} frames", end="")
            elif len(sequence) > expected_frames:
                # Trim if we have too many frames
                sequence = sequence[:expected_frames]
                print(f" trimmed {len(sequence) - expected_frames} frames", end="")
            
            # Save the sequence
            save_path = f"{SAVE_DIR}/{word}_{i}.npy"
            np.save(save_path, sequence)
            print(f" ✅ Saved - Shape: {sequence.shape}")
        else:
            print(f" ❌ No data captured")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n🎉 Recording complete!")
print(f"Data saved in: {os.path.abspath(SAVE_DIR)}")