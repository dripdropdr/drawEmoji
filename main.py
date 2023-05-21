# All the imports go here
import cv2
import numpy as np
import mediapipe as mp
from collections import deque


# Giving different arrays to handle colour points of different colour
points = [deque(maxlen=1024)]


# These indexes will be used to mark the points in particular arrays of specific colour
index = 0

#The kernel to be used for dilation purpose 
kernel = np.ones((5,5),np.uint8)

colors = (0, 0, 0)

# Here is code for Canvas setup
paintWindow = np.zeros((720,1280,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    y, x, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # CLEAR button 
    frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        center = (landmarks[8][0],landmarks[8][1])
        middle = (landmarks[12][0],landmarks[12][1])
        cv2.circle(frame, center, 3, (0,255,0),-1)
        print(center[1]-middle[1])
        if (middle[1]-center[1]<30):
            points.append(deque(maxlen=512))
            index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 140: # Clear Button
                points = [deque(maxlen=512)]
                index = 0
                paintWindow[67:,:,:] = 255
        else :
            points[index].appendleft(center)

    # Append the next deques when nothing is detected to avoids messing up
    else:
        points.append(deque(maxlen=512))
        index += 1

    # Draw lines of all the colors on the canvas and frame
    for j in range(len(points)):
        for k in range(1, len(points[j])):
            if points[j][k - 1] is None or points[j][k] is None:
                continue
            cv2.line(frame, points[j][k - 1], points[j][k], colors, 2)
            cv2.line(paintWindow, points[j][k - 1], points[j][k], colors, 2)

    cv2.imshow("Output", frame) 
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()