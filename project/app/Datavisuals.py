import numpy as np
import cv2
import mediapipe as mp
import time

# Load saved sequence
sequence = np.load(r"F:\projects\KSL\KSL_Backend\project\app\dataset\mother\mother_2.npy")

# MediaPipe connections
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# Window
cv2.namedWindow("Hand Replay", cv2.WINDOW_NORMAL)

for frame_data in sequence:
    # Split left & right hand
    left = frame_data[:63].reshape(21, 3)
    right = frame_data[63:].reshape(21, 3)

    # Create blank canvas
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    def draw_hand(hand):
        for (start, end) in HAND_CONNECTIONS:
            x1, y1 = int(hand[start][0] * 1280), int(hand[start][1] * 720)
            x2, y2 = int(hand[end][0] * 1280), int(hand[end][1] * 720)

            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for x, y, _ in hand:
            cv2.circle(
                canvas,
                (int(x * 1280), int(y * 720)),
                5,
                (0, 0, 255),
                -1
            )

    if not np.all(left == 0):
        draw_hand(left)

    if not np.all(right == 0):
        draw_hand(right)

    cv2.imshow("Hand Replay", canvas)

    if cv2.waitKey(100) & 0xFF == 27:
        break

cv2.destroyAllWindows()
