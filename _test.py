import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
drawing = np.zeros((720,1280,3), dtype=np.uint8) + 255
prev_x, prev_y = 0, 0
drawing_mode = False
points = [deque(maxlen=1024)]
index = 0

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally for a later selfie-view display
        # Convert the BGR frame to RGB before processing.
        # frame = cv2.cvtColor(, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        # Process the frame and find hand landmarks.
        results = hands.process(frame)

        # Draw hand landmarks of each hand.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                                
                # Draw all landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of the tip of the index finger.
                x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                
                # Get the coordinates of the tip of the thumb.
                thumb_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1])
                thumb_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])
                
                # Calculate the distance between the tip of the index finger and the tip of the thumb.
                distance = np.sqrt((thumb_x - x)**2 + (thumb_y - y)**2)
                
                # If the distance is less than a threshold, set drawing mode to True. Else, set it to False.
                if distance < 50:
                    drawing_mode = False
                else:
                    drawing_mode = True

                if y <= 65 and 40 <= x <= 140: # Clear Button
                    points = [deque(maxlen=512)]
                    index = 0
                    paintWindow[67:,:,:] = 255

                # Draw a line from the previous position to the current position, if drawing_mode is True.
                if drawing_mode:
                    cv2.line(drawing, (prev_x, prev_y), (x, y), (0, 0, 255), 10)

                prev_x, prev_y = x, y

        # Convert the frame from RGB to BGR for display with OpenCV.
        # drawing = cv2.cvtColor(drawing, cv2.COLOR_RGB2BGR)

        # Merge the drawing onto the frame.
        alpha = 0.5
        merged = cv2.addWeighted(frame, alpha, drawing, 1 - alpha, 0)

        # Display the resulting frame.
        cv2.imshow('MediaPipe Hands', merged)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
