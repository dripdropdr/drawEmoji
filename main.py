# All the imports go here
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import torch
from torchvision import datasets, models, transforms
from copy import deepcopy
from PIL import Image
import os
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# initialize classifier
id2label = {0: 'angel', 1: 'apple', 2: 'arm', 3: 'banana', 4: 'baseball', 5: 'basketball', 6: 'bear', 7: 'beard', 8: 'bird', 9: 'book', 10: 'bowtie', 11: 'bread', 12: 'butterfly', 13: 'cake', 14: 'campfire', 15: 'carrot', 16: 'cat', 17: 'cloud', 18: 'coffee_cup', 19: 'crown', 20: 'diamond', 21: 'dog', 22: 'donut', 23: 'eye', 24: 'face', 25: 'flower', 26: 'garden', 27: 'hand', 28: 'headphones', 29: 'house_plant', 30: 'ice_cream', 31: 'leaf', 32: 'light_bulb', 33: 'lightning', 34: 'ocean', 35: 'palm_tree', 36: 'pizza', 37: 'rabbit', 38: 'smiley_face', 39: 'snowflake', 40: 'snowman', 41: 'star', 42: 'strawberry', 43: 'sun', 44: 'teddy-bear'}

print('model loading ... ')
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3)
model.fc = torch.nn.Linear(model.fc.in_features, 45)
model.to(device)
checkpoint = torch.load('output/emoji/chekcpoint0022.pth')
model.load_state_dict(checkpoint)
model.eval()
print('finish model load !!')

input_size = 64
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# points, emoji & emoji location array
points = [deque(maxlen=1024)]
emoji = []
emojiflg=False

# index indicating deque in points, colors for lines
index = 0
colors = (0, 0, 0)

# Here is code for Canvas setup
paintWindow = np.zeros((720,1280,3)) + 255
# paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
# paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), (125, 125, 125), -1)
# cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# cv2.putText(paintWindow, "EMOJI", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
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

    h, w, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # CLEAR button 
    frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    # EMOJI button 
    frame = cv2.rectangle(frame, (160,1), (255,65), (125, 125, 125), -1)
    cv2.putText(frame, "EMOJI", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    frame = cv2.rectangle(frame, (275,1), (370,65), (0,0,0), 2)
    cv2.putText(frame, "EMO CLEAR", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * w)
                lmy = int(lm.y * h)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        center = (landmarks[8][0],landmarks[8][1])
        middle = (landmarks[12][0],landmarks[12][1])
        cv2.circle(frame, center, 3, (0,255,0),-1)

        # Stop drawing
        if (middle[1]-center[1]<30):
            points.append(deque(maxlen=512))
            index += 1
        # In button
        elif center[1] <= 65:
            # clear button
            if 40 <= center[0] <= 140: 
                points = [deque(maxlen=512)]
                index = 0
                paintWindow[67:,:,:] = 255
            # emoji button
            elif 160 <= center[0] <= 255:
                emojiflg = True
                image = deepcopy(paintWindow)
                points = [deque(maxlen=512)]
                index = 0
                paintWindow[67:,:,:] = 255
            # emoji clear button
            elif 275 <= center[0] <= 370:
                emoji = []
        else :
            points[index].appendleft(center)

    # Append the next deques when nothing is detected to avoids messing up
    else:
        points.append(deque(maxlen=512))
        index += 1

    # draw emoji
    if len(emoji) > 0:
        print(len(emoji))
        for emo, y, x in emoji:
            frame[y:y+emo.shape[0], x:x+emo.shape[1], :] = emo

    # Draw lines of all the colors on the canvas and frame
    for j in range(len(points)):
        for k in range(1, len(points[j])):
            if points[j][k - 1] is None or points[j][k] is None:
                continue
            cv2.line(frame, points[j][k - 1], points[j][k], colors, 2)
            cv2.line(paintWindow, points[j][k - 1], points[j][k], colors, 2)

    cv2.imshow("Output", frame) 
    cv2.imshow("Paint", paintWindow)


    if emojiflg:
        # Classificate drawing
        '''
            paintedWindow : (h * w) numpy array.
        '''
        # Crop paintedWindow to square.
        image = image[:, int(w/2 - h/2):int(w/2 + h/2)]
        image = Image.fromarray(np.uint8(image)).convert('L')
        # image = np.expand_dims(image, axis=0)

        # Put image to Classifier
        
        # image = Image.open("bus.jpg").convert("RGB")
        image = data_transforms['val'](image)

        # predict images
        output = model(image[None])
        pred = torch.argmax(output, axis=1)
        emo_img = cv2.imread(f'emoji/{id2label[pred.item()]}.png')
        emoji.append([emo_img, (random.randint(0, h - 100)), random.randint(0, w - 100)])
        print(id2label[pred.item()], emoji)
        emojiflg = False

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()