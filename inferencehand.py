import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
import threading

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

num_of_timesteps = 9
model = load_model(f'model/best_model_9.h5')

mppose = mp.solutions.pose
pose = mppose.Pose()
mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

NUM_HAND_LANDMARKS = 21

def make_dat(hand_landmarks):
    lm_list = []
    
    if hand_landmarks:
        hand_lm = hand_landmarks.landmark
        base_x = hand_lm[0].x
        base_y = hand_lm[0].y
        base_z = hand_lm[0].z
        center_x = np.mean([lm.x for lm in hand_lm])
        center_y = np.mean([lm.y for lm in hand_lm])
        center_z = np.mean([lm.z for lm in hand_lm])

        distances = [np.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2 + (lm.z - center_z)**2) for lm in hand_lm[1:]]
        scale_factors = [1.0 / dist for dist in distances]

        lm_list.append(0.0)
        lm_list.append(0.0)
        lm_list.append(0.0)
        lm_list.append(hand_lm[0].visibility)

        for lm, scale_factor in zip(hand_lm[1:], scale_factors):
            lm_list.append((lm.x - base_x) * scale_factor)
            lm_list.append((lm.y - base_y) * scale_factor)
            lm_list.append((lm.z - base_z) * scale_factor)
            lm_list.append(lm.visibility)
    else:
        lm_list.extend([0.0] * (NUM_HAND_LANDMARKS * 4))
    
    
    
    print(f"Length of lm_list: {len(lm_list)}")  
    return lm_list

def draw_land(mpDraw, results_hand, img):
    # Vẽ hand landmarks
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                img,
                hand_landmarks,
                mphands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
    
    return img

def draw_class_on_image(confidence, label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 50)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    cv2.putText(img, str(confidence),
                (400, 600),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

label = "Unknown"
confidence = 0
def detect(model, lm_list):
    global label
    global confidence
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    predicted_label_index = np.argmax(results, axis=1)[0]
    classes = ['a', 'b', 'c', 'o', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
         'l', 'm', 'n', 'p', 'q', 'r', 's', 'space', 't', 'u',
         'v', 'w', 'x', 'y', 'z', 'yes', 'no', 'me', 'you', 'hello',
         'i_love_you', 'thank_you', 'sorry', 'do', 'eat', 'what', 'why', 
         'who', 'where', 'when', 'how', 'how_much', 'go', 'happy', 
         'sad', 'good', 'bad'] 
    #no, thankyou, how, good, when
    confidence = np.max(results, axis=1)[0]
    if confidence > 0.85:
        label = classes[predicted_label_index]
    else:
        label = "cant detect"


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
lm_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hand = hands.process(frameRGB)
    
    if results_hand.multi_hand_landmarks :
        hand_landmarks = results_hand.multi_hand_landmarks[0] if results_hand.multi_hand_landmarks else None
        

        if hand_landmarks:
            lm = make_dat(hand_landmarks)   
            lm_list.append(lm)
            if len(lm_list) == num_of_timesteps:
                detect_thread = threading.Thread(target=detect, args=(model, lm_list,))
                detect_thread.start()
                lm_list = []
        frame = draw_land(mpDraw, results_hand, frame)
    frame = draw_class_on_image(confidence, label, frame)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
