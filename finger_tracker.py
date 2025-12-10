# finger_tracker.py - DAY 3: Finger Tracking with Trail
import cv2
import mediapipe as mp
import numpy as np

print("=== FINGER TRACKING WITH TRAIL ===")
print("Initializing...")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create hand detector
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1  # Only track one hand for simplicity
)

# Open webcam
cap = cv2.VideoCapture(0)

# List to store finger trail points
trail_points = []
max_trail_length = 50  # How many points to keep in trail

# Drawing canvas (for virtual drawing feature)
canvas = None
drawing_mode = False
draw_color = (0, 255, 0)  # Green

print("\nINSTRUCTIONS:")
print("1. Show your INDEX FINGER to the camera")
print("2. See the blue trail follow your finger")
print("3. Press 'D' to toggle drawing mode ON/OFF")
print("4. Press 'C' to clear drawing")
print("5. Press 'Q' to quit")
print("-" * 40)

while True:
    # Read frame
    success, frame = cap.read()
    if not success:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Initialize canvas (same size as frame)
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Get frame dimensions
    height, width, _ = frame.shape

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process for hand detection
    results = hands.process(rgb_frame)

    # Reset trail if no hand detected
    if not results.multi_hand_landmarks:
        if len(trail_points) > 0:
            trail_points.clear()
            print("No hand - trail cleared")

    # If hand detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw all hand landmarks (optional - can comment out)
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(50, 50, 50), thickness=1)
            )

            # Get INDEX FINGER TIP (Landmark 8)
            index_tip = hand_landmarks.landmark[8]

            # Convert to pixel coordinates
            finger_x = int(index_tip.x * width)
            finger_y = int(index_tip.y * height)

            # Add to trail points
            trail_points.append((finger_x, finger_y))

            # Keep only last N points (for trail effect)
            if len(trail_points) > max_trail_length:
                trail_points.pop(0)

            # Draw a bright circle on index finger tip
            cv2.circle(frame, (finger_x, finger_y), 15, (0, 255, 255), -1)  # Yellow circle
            cv2.circle(frame, (finger_x, finger_y), 15, (0, 0, 0), 2)  # Black border

            # Draw the trail (connect all points with lines)
            for i in range(1, len(trail_points)):
                # Draw thicker lines for recent points, thinner for older ones
                thickness = max(1, 5 - i // 10)
                # Draw line between consecutive points
                cv2.line(frame, trail_points[i - 1], trail_points[i],
                         (255, 0, 0), thickness)  # Blue trail

            # If drawing mode is ON, draw on canvas
            if drawing_mode:
                cv2.circle(canvas, (finger_x, finger_y), 10, draw_color, -1)

            # Draw info text near finger
            cv2.putText(frame, f"X: {finger_x}, Y: {finger_y}",
                        (finger_x + 20, finger_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Combine frame with canvas (for drawing)
    if drawing_mode:
        # Blend canvas with frame
        frame = cv2.addWeighted(frame, 0.8, canvas, 0.2, 0)

    # Display trail length info
    cv2.putText(frame, f"Trail Points: {len(trail_points)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display mode info
    mode_text = "DRAWING: ON (Press 'D' to turn off)" if drawing_mode else "DRAWING: OFF (Press 'D' to turn on)"
    mode_color = (0, 255, 0) if drawing_mode else (0, 0, 255)
    cv2.putText(frame, mode_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    # Display instructions
    cv2.putText(frame, "Press 'D': Toggle Drawing", (width - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    cv2.putText(frame, "Press 'C': Clear Drawing", (width - 250, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    cv2.putText(frame, "Press 'Q': Quit", (width - 250, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    # Show the frame
    cv2.imshow('Finger Tracking with Trail - Press Q to Quit', frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\nExiting program...")
        break
    elif key == ord('d'):  # Toggle drawing mode
        drawing_mode = not drawing_mode
        status = "ON" if drawing_mode else "OFF"
        print(f"Drawing mode: {status}")
    elif key == ord('c'):  # Clear canvas
        canvas = np.zeros_like(frame)
        trail_points.clear()
        print("Canvas cleared!")
    elif key == ord('1'):  # Change to red
        draw_color = (0, 0, 255)
        print("Color: RED")
    elif key == ord('2'):  # Change to green
        draw_color = (0, 255, 0)
        print("Color: GREEN")
    elif key == ord('3'):  # Change to blue
        draw_color = (255, 0, 0)
        print("Color: BLUE")

# Release resources
cap.release()
cv2.destroyAllWindows()
print("=" * 40)
print("Day 3 Complete! Great job!")
print("Next: Add virtual buttons and interaction")
print("=" * 40)
