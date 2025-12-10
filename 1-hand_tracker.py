# hand_tracker.py - DAY 2: Basic Hand Detection
import cv2
import mediapipe as mp

print("=== HAND TRACKING PROGRAM ===")
print("Initializing...")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create hand detector with medium confidence
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2  # Detect up to 2 hands
)

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    print("1. Check if webcam is connected")
    print("2. Another program might be using the webcam")
    exit()

print("Webcam opened successfully!")
print("\nINSTRUCTIONS:")
print("1. Show your hand to the camera")
print("2. Move your hand around")
print("3. Press 'Q' on keyboard to quit")
print("-" * 40)

while True:
    # Read a frame from webcam
    success, frame = cap.read()

    if not success:
        print("Failed to get frame from webcam")
        break

    # Flip horizontally for mirror effect (feels more natural)
    frame = cv2.flip(frame, 1)

    # Convert BGR (OpenCV default) to RGB (MediaPipe needs RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # If hands are detected, draw landmarks
    if results.multi_hand_landmarks:
        print(f"Hands detected: {len(results.multi_hand_landmarks)}")

        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the 21 hand landmarks and connections
            mp_drawing.draw_landmarks(
                frame,  # Image to draw on
                hand_landmarks,  # Detected hand landmarks
                mp_hands.HAND_CONNECTIONS,  # Draw connections between landmarks
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # Landmark style
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # Connection style
            )
    else:
        print("No hand detected - show your hand to the camera")

    # Display the frame in a window
    cv2.imshow('Hand Tracking - Press Q to Quit', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nExiting program...")
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
print("Program ended cleanly!")
print("Great job! Move to Day 3: Finger Tracking next.")
