# Create a test file: test_install.py
import cv2
import mediapipe as mp
import numpy as np
print("All packages installed successfully!")

# test_install.py
import cv2
import mediapipe as mp
import numpy as np

print("All packages installed successfully!")

import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create hand detector
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the program")
print("Show your hand to the camera...")

while cap.isOpened():
    # Read frame
    success, frame = cap.read()
    if not success:
        print("Failed to get frame from webcam")
        continue

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    results = hands.process(rgb_frame)

    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Display the frame
    cv2.imshow('Hand Detection - Press Q to quit', frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Program ended successfully!")
