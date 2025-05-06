import pyautogui
import cv2
import mediapipe as mp 
import numpy as np 
import time

print("Everything imported successfully")

mpHands = mp.solutions.hands 
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

s = pyautogui.size()

cap = cv2.VideoCapture(0)
last_click = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)

#--new code
    x,y,c = frame.shape
    result = hands.process(frame)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            #--new code pt2
            pyautogui.moveTo(int(handslms.landmark[8].x*s[0]), int(handslms.landmark[8].y*s[1]), _pause=False)
            #--end of new code pt2
            #--new code pt3
            index_x = int(handslms.landmark[8].x*s[0])
            index_y = int(handslms.landmark[8].y*s[1])
            thumb_x = int(handslms.landmark[4].x*s[0])
            thumb_y = int(handslms.landmark[4].y*s[1])
            distance = ((index_x - thumb_x)**2 + (index_y - thumb_y)**2) ** 0.5

            if distance < 40 and time.time() - last_click > 0.5:
                pyautogui.click()
                last_click = time.time()

            #--end of new code pt3
            for lm in handslms.landmark:
                lmx = int(lm.x*x)
                lmy = int(lm.y*y)
                landmarks.append([lmx,lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
#--end of new code    

    cv2.imshow('Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)