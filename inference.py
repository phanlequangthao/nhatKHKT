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
#hello, hihia
num_of_timesteps = 9
model = load_model(f'model/model_{num_of_timesteps}.keras')

mppose = mp.solutions.pose
pose = mppose.Pose()
mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

# Số lượng landmarks của tay và pose
NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = 33

def make_dat(hand_landmarks, pose_landmarks):
    lm_list = []
    
    # Process hand landmarks
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
        # Fill with zeros if hand landmarks are not detected
        lm_list.extend([0.0] * (NUM_HAND_LANDMARKS * 4))
    
    # Process pose landmarks
    if pose_landmarks:
        pose_lm = pose_landmarks.landmark
        base_x = pose_lm[0].x
        base_y = pose_lm[0].y
        base_z = pose_lm[0].z
        
        center_x = np.mean([lm.x for lm in pose_lm])
        center_y = np.mean([lm.y for lm in pose_lm])
        center_z = np.mean([lm.z for lm in pose_lm])

        distances = [np.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2 + (lm.z - center_z)**2) for lm in pose_lm[1:]]
        scale_factors = [1.0 / dist for dist in distances]

        lm_list.append(0.0)
        lm_list.append(0.0)
        lm_list.append(0.0)
        lm_list.append(pose_lm[0].visibility)

        for lm, scale_factor in zip(pose_lm[1:], scale_factors):
            lm_list.append((lm.x - base_x) * scale_factor)
            lm_list.append((lm.y - base_y) * scale_factor)
            lm_list.append((lm.z - base_z) * scale_factor)
            lm_list.append(lm.visibility)
    else:
        # Fill with zeros if pose landmarks are not detected
        lm_list.extend([0.0] * (NUM_POSE_LANDMARKS * 4))
    
    print(f"Length of lm_list: {len(lm_list)}")  
    return lm_list

def draw_land(mpDraw, results_hand, results_pose, img):
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
    
    # Vẽ pose landmarks
    if results_pose.pose_landmarks:
        mpDraw.draw_landmarks(
            img,
            results_pose.pose_landmarks,
            mppose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
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
                (50,50),
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
         'i_love_you', 'thank_you', 'sorry', 'more', 'hurt', 'all_done']
    confidence = np.max(results, axis=1)[0]
    if confidence > 0.95:
        label = classes[predicted_label_index]
    else:
        label = "neutral"


cap = cv2.VideoCapture(0)
cap.set(640,640)

lm_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hand = hands.process(frameRGB)
    results_pose = pose.process(frameRGB)   
    if results_hand.multi_hand_landmarks or results_pose.pose_landmarks:
        hand_landmarks = results_hand.multi_hand_landmarks[0] if results_hand.multi_hand_landmarks else None
        pose_landmarks = results_pose.pose_landmarks if results_pose.pose_landmarks else None

        if hand_landmarks:
            lm = make_dat(hand_landmarks, pose_landmarks)   
            lm_list.append(lm)
            if len(lm_list) == num_of_timesteps:
                detect_thread = threading.Thread(target=detect, args=(model, lm_list,))
                detect_thread.start()
                lm_list = []
        frame = draw_land(mpDraw, results_hand, results_pose, frame)
    # frame = cv2.flip(frame, 1)
    frame = draw_class_on_image(confidence, label, frame)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
