Gesture Tracking Interactive Program
====================================

Author: Abdinor
Date: December 2024

DESCRIPTION:
A real-time hand gesture tracking system built with Python, OpenCV, and MediaPipe.
This program allows users to interact with virtual objects using hand gestures,
recognizes 7 different gestures, and includes a drawing canvas.

FEATURES:
1. Real-time hand detection with 21 landmarks
2. Finger tracking with visual blue trail
3. Virtual button interaction using pinch gestures
4. Recognition of 7 hand gestures:
   - Open Hand: Drawing mode
   - Fist: Eraser mode
   - Peace Sign: Green color
   - OK Sign: Red color
   - Rock Sign: Blue color
   - Thumbs Up: Clear canvas
   - Thumbs Down: Undo last action
5. Virtual drawing canvas with multiple colors
6. Scoring/achievement system

HOW TO RUN:
1. Install Python 3.10 or higher
2. Open terminal/command prompt in project folder
3. Run: pip install -r requirements.txt
4. Run: python gesture_tracker.py
5. Show your hand to the webcam

REQUIRED HARDWARE:
- Webcam
- Good lighting conditions

FILES INCLUDED:
1. hand_tracker.py - Basic hand detection (Day 2)
2. finger_tracker.py - Finger tracking with trail (Day 3)
3. button_tracker.py - Virtual button interaction (Day 4)
4. gesture_tracker.py - Main program with gesture recognition (Day 5)
5. requirements.txt - Python dependencies
6. README.txt - This file
7. demo_video.mp4 - 1-minute demonstration video

KEYBOARD CONTROLS:
- Press 'Q': Quit program
- Press 'S': Save current drawing
- Press 'D': Toggle drawing mode (in finger_tracker.py)
- Press 'C': Clear canvas

GESTURE CONTROLS:
- Show OPEN HAND (all 5 fingers) to start drawing
- Make a FIST to switch to eraser mode
- Show PEACE SIGN (✌️) for green color
- Show OK SIGN (👌) for red color
- Show ROCK SIGN (🤘) for blue color
- THUMBS UP (👍) to clear canvas
- THUMBS DOWN (👎) to undo last drawing

TROUBLESHOOTING:
1. If webcam doesn't open: Change cv2.VideoCapture(0) to cv2.VideoCapture(1)
2. If hand not detected: Improve lighting, ensure hand is visible
3. If program is slow: Make sure no other programs are using the webcam

NOTE:
This project was developed step-by-step over 7 days as part of a computer vision learning project.

Contact: Abdinor155@gmail.com