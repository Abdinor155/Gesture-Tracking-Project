# gesture_tracker.py - DAY 5: Advanced Features with Gesture Recognition
import cv2
import mediapipe as mp
import numpy as np
import time

print("=" * 60)
print("ADVANCED GESTURE RECOGNITION PROGRAM")
print("=" * 60)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    max_num_hands=1
)

# Open webcam
cap = cv2.VideoCapture(0)

# State variables
canvas = None
trail_points = []
mode = "idle"  # idle, draw, erase, select
current_color = (0, 255, 0)  # Green
brush_size = 10
last_gesture_time = 0
recognized_gesture = "None"
score = 0

# Drawing tools
tools = {
    "brush": {"size": 10, "color": (0, 255, 0), "active": True},
    "eraser": {"size": 20, "color": (0, 0, 0), "active": False},
    "spray": {"size": 15, "color": (255, 0, 0), "active": False}
}

# Gesture actions
gesture_actions = {
    "open_hand": "DRAW",
    "fist": "ERASE",
    "peace": "GREEN",
    "ok": "RED",
    "rock": "BLUE",
    "thumbs_up": "CLEAR",
    "thumbs_down": "UNDO"
}

print("\n" + "=" * 60)
print("GESTURE CONTROLS:")
print("Open Hand (5 fingers)  → Drawing mode")
print("Fist                    → Eraser mode")
print("Peace Sign (✌️)         → Green color")
print("OK Sign (👌)           → Red color")
print("Rock (🤘)              → Blue color")
print("Thumbs Up (👍)         → Clear canvas")
print("Thumbs Down (👎)       → Undo last action")
print("=" * 60)
print("\nPress 'Q' to quit")
print("Press 'S' to save your drawing")
print("=" * 60)


def count_fingers(landmarks, height, width):
    """Count how many fingers are raised"""
    finger_tips = [4, 8, 12, 16, 20]  # Thumb to Pinky
    finger_pips = [3, 6, 10, 14, 18]  # Lower joints

    finger_count = 0
    finger_positions = []

    # Thumb is special (compares x coordinate)
    thumb_tip = landmarks.landmark[4]
    thumb_ip = landmarks.landmark[3]

    if thumb_tip.x < thumb_ip.x:  # Thumb is extended (left hand)
        finger_count += 1

    # Other four fingers (compare y coordinate)
    for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
        tip_point = landmarks.landmark[tip]
        pip_point = landmarks.landmark[pip]

        if tip_point.y < pip_point.y:  # Finger is extended
            finger_count += 1
            finger_positions.append(tip)

    return finger_count, finger_positions


def recognize_gesture(finger_count, finger_positions, landmarks, height, width):
    """Recognize specific hand gestures"""

    # Get specific landmarks
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    middle_tip = landmarks.landmark[12]
    ring_tip = landmarks.landmark[16]
    pinky_tip = landmarks.landmark[20]

    # Calculate distances
    def distance(p1, p2):
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    # Open hand (all fingers extended)
    if finger_count == 5:
        return "open_hand"

    # Fist (no fingers extended)
    elif finger_count == 0:
        return "fist"

    # Peace sign (index and middle finger up)
    elif finger_count == 2 and 8 in finger_positions and 12 in finger_positions:
        # Check if ring and pinky are down
        ring_up = ring_tip.y < landmarks.landmark[14].y
        pinky_up = pinky_tip.y < landmarks.landmark[18].y
        if not ring_up and not pinky_up:
            return "peace"

    # OK sign (thumb and index touching)
    elif finger_count == 3 and distance(thumb_tip, index_tip) < 0.05:
        return "ok"

    # Rock sign (index and pinky up)
    elif finger_count == 2 and 8 in finger_positions and 20 in finger_positions:
        return "rock"

    # Thumbs up (only thumb up)
    elif finger_count == 1 and thumb_tip.x < landmarks.landmark[3].x:
        # Check other fingers are down
        other_fingers_down = True
        for tip in [8, 12, 16, 20]:
            if landmarks.landmark[tip].y < landmarks.landmark[tip - 2].y:
                other_fingers_down = False
                break
        if other_fingers_down:
            return "thumbs_up"

    # Thumbs down (only thumb down orientation)
    elif finger_count == 4 and thumb_tip.x > landmarks.landmark[3].x:
        return "thumbs_down"

    return "unknown"


def draw_gesture_info(frame, gesture, finger_count, height, width):
    """Display gesture information on frame"""

    # Create info panel
    info_panel = np.zeros((150, 400, 3), dtype=np.uint8)

    # Gesture name
    gesture_name = gesture.replace("_", " ").title()
    cv2.putText(info_panel, f"GESTURE: {gesture_name}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Finger count
    cv2.putText(info_panel, f"FINGERS: {finger_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Action
    if gesture in gesture_actions:
        action = gesture_actions[gesture]
        cv2.putText(info_panel, f"ACTION: {action}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Place panel in top-right corner
    frame[10:160, width - 410:width - 10] = info_panel


def draw_score_board(frame, score, height, width):
    """Display score/achievement board"""
    score_panel = np.zeros((100, 300, 3), dtype=np.uint8)

    cv2.putText(score_panel, "ACHIEVEMENTS", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(score_panel, f"SCORE: {score}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Place at top-left
    frame[10:110, 10:310] = score_panel


def handle_gesture_action(gesture):
    """Handle actions based on recognized gesture"""
    global mode, current_color, canvas, trail_points, score

    if gesture == "open_hand":
        mode = "draw"
        print("✓ Gesture: OPEN HAND → Drawing mode")
        return 10

    elif gesture == "fist":
        mode = "erase"
        print("✓ Gesture: FIST → Eraser mode")
        return 10

    elif gesture == "peace":
        current_color = (0, 255, 0)  # Green
        mode = "draw"
        print("✓ Gesture: PEACE → Green color")
        return 20

    elif gesture == "ok":
        current_color = (0, 0, 255)  # Red
        mode = "draw"
        print("✓ Gesture: OK → Red color")
        return 20

    elif gesture == "rock":
        current_color = (255, 0, 0)  # Blue
        mode = "draw"
        print("✓ Gesture: ROCK → Blue color")
        return 20

    elif gesture == "thumbs_up":
        if canvas is not None:
            canvas = np.zeros_like(canvas)
            trail_points.clear()
            print("✓ Gesture: THUMBS UP → Canvas cleared")
            return 30

    elif gesture == "thumbs_down":
        # Simple undo: clear last 50 points
        if len(trail_points) > 50:
            for _ in range(50):
                if trail_points:
                    trail_points.pop()
            print("✓ Gesture: THUMBS DOWN → Undo last drawing")
            return 15

    return 0


# Main loop
print("\nStarting camera... Show your hand and try different gestures!")
last_gesture = "None"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Initialize canvas
    if canvas is None:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 100, 0), thickness=2)
            )

            # Get index finger tip for drawing
            index_tip = hand_landmarks.landmark[8]
            finger_x = int(index_tip.x * width)
            finger_y = int(index_tip.y * height)

            # Count fingers and recognize gesture
            finger_count, finger_positions = count_fingers(hand_landmarks, height, width)
            gesture = recognize_gesture(finger_count, finger_positions, hand_landmarks, height, width)

            # Update recognized gesture
            if gesture != "unknown":
                recognized_gesture = gesture

            # Draw gesture info
            draw_gesture_info(frame, gesture, finger_count, height, width)

            # Handle gesture action (with delay to avoid rapid triggers)
            if current_time - last_gesture_time > 1.5:  # 1.5 second delay between gestures
                points_earned = handle_gesture_action(gesture)
                if points_earned > 0:
                    score += points_earned
                    last_gesture_time = current_time
                    last_gesture = gesture

            # Draw on canvas based on mode
            if mode == "draw":
                cv2.circle(canvas, (finger_x, finger_y), brush_size, current_color, -1)
                # Add sparkle effect
                cv2.circle(frame, (finger_x, finger_y), 5, (255, 255, 255), -1)

            elif mode == "erase":
                cv2.circle(canvas, (finger_x, finger_y), brush_size * 2, (0, 0, 0), -1)

            # Add to trail
            trail_points.append((finger_x, finger_y))
            if len(trail_points) > 100:
                trail_points.pop(0)

            # Draw index finger with glow effect
            cv2.circle(frame, (finger_x, finger_y), 20, (255, 255, 0), 2)
            cv2.circle(frame, (finger_x, finger_y), 15, (0, 255, 255), -1)
            cv2.circle(frame, (finger_x, finger_y), 8, (255, 255, 255), -1)

    # Draw trail with gradient
    for i in range(1, len(trail_points)):
        alpha = i / len(trail_points)
        trail_color = (
            int(255 * (1 - alpha)),
            int(128 * alpha),
            255
        )
        thickness = max(1, int(4 * alpha))
        cv2.line(frame, trail_points[i - 1], trail_points[i], trail_color, thickness)

    # Combine with canvas
    frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    # Draw score board
    draw_score_board(frame, score, height, width)

    # Draw mode indicator
    mode_panel = np.zeros((80, 300, 3), dtype=np.uint8)
    mode_color = current_color if mode == "draw" else (100, 100, 100)
    mode_text = f"MODE: {mode.upper()}"
    cv2.putText(mode_panel, mode_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2)
    frame[height - 100:height - 20, 10:310] = mode_panel

    # Draw instructions at bottom
    cv2.putText(frame, "Try: OPEN HAND, FIST, PEACE, OK, ROCK, THUMBS UP/DOWN",
                (width // 2 - 300, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    # Show frame
    cv2.imshow('Advanced Gesture Recognition - Press Q to quit', frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save drawing
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"drawing_{timestamp}.png"
        cv2.imwrite(filename, canvas)
        print(f"✓ Drawing saved as: {filename}")
        score += 50

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("DAY 5 COMPLETE! Outstanding work!")
print(f"Final Score: {score}")
print("=" * 60)
print("\nWhat you've accomplished:")
print("1. Hand detection with 21 landmarks")
print("2. Finger tracking with colorful trail")
print("3. Virtual button interaction")
print("4. GESTURE RECOGNITION (7 different gestures!)")
print("5. Score/achievement system")
print("=" * 60)
print("\nNext: Record your 1-minute demo video!")
