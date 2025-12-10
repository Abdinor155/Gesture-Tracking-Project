# button_tracker.py - DAY 4: Virtual Button Interaction
import cv2
import mediapipe as mp
import numpy as np

print("=== VIRTUAL BUTTON INTERACTION ===")
print("Initializing...")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Open webcam
cap = cv2.VideoCapture(0)

# List to store finger trail points
trail_points = []
max_trail_length = 30

# Drawing canvas
canvas = None
drawing_mode = False
draw_color = (0, 255, 0)  # Green
brush_size = 10

# VIRTUAL BUTTONS
buttons = [
    # Each button: [x1, y1, x2, y2, label, color, is_active, action]
    [50, 50, 150, 120, "DRAW", (0, 255, 0), True, "toggle_draw"],
    [170, 50, 270, 120, "ERASE", (0, 0, 255), False, "toggle_erase"],
    [290, 50, 340, 120, "R", (0, 0, 255), False, "color_red"],
    [360, 50, 410, 120, "G", (0, 255, 0), False, "color_green"],
    [430, 50, 480, 120, "B", (255, 0, 0), False, "color_blue"],
    [500, 50, 600, 120, "CLEAR", (100, 100, 100), False, "clear_all"],
    [620, 50, 720, 120, "TRAIL", (255, 255, 0), True, "toggle_trail"]
]

# State variables
show_trail = True
current_action = "draw"  # draw, erase
last_button_click_time = 0
click_delay = 500  # milliseconds between clicks

print("\n" + "=" * 50)
print("INSTRUCTIONS:")
print("1. Move your INDEX FINGER to control")
print("2. PINCH thumb and index finger to 'click' buttons")
print("3. Buttons will change color when clicked")
print("4. Press 'Q' to quit")
print("=" * 50)


def draw_buttons(frame):
    """Draw all virtual buttons on the frame"""
    for btn in buttons:
        x1, y1, x2, y2, label, color, is_active, _ = btn

        # Draw button rectangle
        if is_active:
            # Active button: filled with brighter color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
        else:
            # Inactive button: outline only
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # Draw button label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.7, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        text_color = (255, 255, 255) if is_active else (200, 200, 200)
        cv2.putText(frame, label, (text_x, text_y), font, 0.7, text_color, 2)


def is_pinch_gesture(landmarks, width, height, threshold=40):
    """Check if thumb and index finger are pinched together"""
    # Get thumb tip (landmark 4) and index tip (landmark 8)
    thumb = landmarks.landmark[4]
    index = landmarks.landmark[8]

    # Convert to pixel coordinates
    thumb_x, thumb_y = int(thumb.x * width), int(thumb.y * height)
    index_x, index_y = int(index.x * width), int(index.y * height)

    # Calculate distance between thumb and index
    distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

    return distance < threshold, (thumb_x, thumb_y), (index_x, index_y)


def check_button_click(x, y):
    """Check if coordinates are inside any button"""
    for i, btn in enumerate(buttons):
        x1, y1, x2, y2, _, _, _, action = btn
        if x1 < x < x2 and y1 < y < y2:
            return i, action
    return None, None


def handle_button_action(action_index):
    """Handle button click actions"""
    global drawing_mode, draw_color, current_action, canvas, show_trail, trail_points

    if action_index is None:
        return

    # Deactivate all buttons first
    for btn in buttons:
        btn[6] = False  # Set is_active to False

    # Activate clicked button and perform action
    buttons[action_index][6] = True

    action = buttons[action_index][7]

    if action == "toggle_draw":
        drawing_mode = True
        current_action = "draw"
        draw_color = buttons[action_index][5]  # Use button color
        print("✓ Drawing mode ON")

    elif action == "toggle_erase":
        drawing_mode = True
        current_action = "erase"
        draw_color = (0, 0, 0)  # Black for erasing
        print("✓ Erase mode ON")

    elif action == "color_red":
        drawing_mode = True
        current_action = "draw"
        draw_color = (0, 0, 255)  # Red
        print("✓ Color: RED")

    elif action == "color_green":
        drawing_mode = True
        current_action = "draw"
        draw_color = (0, 255, 0)  # Green
        print("✓ Color: GREEN")

    elif action == "color_blue":
        drawing_mode = True
        current_action = "draw"
        draw_color = (255, 0, 0)  # Blue
        print("✓ Color: BLUE")

    elif action == "clear_all":
        canvas = None
        trail_points.clear()
        print("✓ Canvas cleared!")

    elif action == "toggle_trail":
        show_trail = not show_trail
        print(f"✓ Trail: {'ON' if show_trail else 'OFF'}")


# Main loop
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

    # Draw buttons
    draw_buttons(frame)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    finger_x, finger_y = None, None
    is_pinching = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks (lightly)
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(50, 50, 50), thickness=1)
            )

            # Get index finger tip
            index_tip = hand_landmarks.landmark[8]
            finger_x = int(index_tip.x * width)
            finger_y = int(index_tip.y * height)

            # Check for pinch gesture (for clicking)
            pinch, thumb_pos, index_pos = is_pinch_gesture(hand_landmarks, width, height)

            # Draw pinch line if pinching
            if pinch:
                cv2.line(frame, thumb_pos, index_pos, (0, 255, 255), 3)
                is_pinching = True

                # Check if pinching over a button
                button_index, action = check_button_click(finger_x, finger_y)
                if button_index is not None:
                    handle_button_action(button_index)

            # Draw index finger circle
            finger_color = (0, 255, 255) if pinch else (0, 200, 255)
            cv2.circle(frame, (finger_x, finger_y), 12, finger_color, -1)
            cv2.circle(frame, (finger_x, finger_y), 12, (0, 0, 0), 2)

            # Add to trail if showing trail
            if show_trail:
                trail_points.append((finger_x, finger_y))
                if len(trail_points) > max_trail_length:
                    trail_points.pop(0)

            # Draw on canvas if in drawing mode
            if drawing_mode and finger_x is not None and finger_y is not None:
                if current_action == "erase":
                    cv2.circle(canvas, (finger_x, finger_y), brush_size * 2, (0, 0, 0), -1)
                else:
                    cv2.circle(canvas, (finger_x, finger_y), brush_size, draw_color, -1)

    # Draw trail
    if show_trail:
        for i in range(1, len(trail_points)):
            alpha = i / len(trail_points)  # Fade out older points
            color = (int(255 * alpha), int(255 * alpha), 255)  # Blue fading
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, trail_points[i - 1], trail_points[i], color, thickness)

    # Combine canvas with frame
    frame = cv2.addWeighted(frame, 0.8, canvas, 0.2, 0)

    # Display status panel
    status_panel = np.zeros((100, width, 3), dtype=np.uint8)

    # Current mode
    mode_text = f"MODE: {current_action.upper()}"
    mode_color = draw_color if current_action == "draw" else (0, 0, 0)
    cv2.putText(status_panel, mode_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.circle(status_panel, (150, 30), 10, mode_color, -1)

    # Pinch status
    pinch_status = "PINCHING: YES" if is_pinching else "PINCHING: NO"
    pinch_color = (0, 255, 0) if is_pinching else (0, 0, 255)
    cv2.putText(status_panel, pinch_status, (width // 2 - 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, pinch_color, 2)

    # Instructions
    cv2.putText(status_panel, "PINCH to click buttons", (width - 300, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

    # Add status panel to bottom of frame
    frame[height - 100:height, 0:width] = status_panel

    # Add border to frame
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (100, 100, 100), 2)

    # Show frame
    cv2.imshow('Virtual Button Control - Press Q to quit', frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        handle_button_action(2)  # Red
    elif key == ord('2'):
        handle_button_action(3)  # Green
    elif key == ord('3'):
        handle_button_action(4)  # Blue
    elif key == ord(' '):
        # Space toggles drawing
        drawing_mode = not drawing_mode
        print(f"Drawing: {'ON' if drawing_mode else 'OFF'}")

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("DAY 4 COMPLETE! Excellent work!")
print("You now have a fully interactive hand-controlled program!")
print("Next: Record a demo video and prepare submission")
print("=" * 50)
