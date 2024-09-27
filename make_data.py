import os
import shutil
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

mppose = mp.solutions.pose
pose = mppose.Pose()
mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

# Số lượng landmarks của tay và pose
NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = 33

label = ['a', 'b', 'c', 'o', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
         'l', 'm', 'n', 'p', 'q', 'r', 's', 'space', 't', 'u',
         'v', 'w', 'x', 'y', 'z', 'yes', 'no', 'me', 'you', 'hello',
         'i_love_you', 'thank_you', 'sorry']

if os.path.exists('./dataset'):
    shutil.rmtree('./dataset')

for cl in label:
    os.makedirs(f'./dataset/{cl}')

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

def get_hand_bbox(landmarks, image_width, image_height):
        x_min, x_max, y_min, y_max = float('inf'), 0, float('inf'), 0

        for lm in landmarks.landmark:
            x, y = int(lm.x * image_width), int(lm.y * image_height)
            x_min = min(x_min, x) - 3
            x_max = max(x_max, x) + 1
            y_min = min(y_min, y) - 3
            y_max = max(y_max, y) + 1

        bbox = ((x_min, y_min), (x_max, y_max))
        return bbox

def convert_coordinates(x1, y1, x2, y2, image_width, image_height):
    x = (x1 + x2) / (2 * image_width)
    y = (y1 + y2) / (2 * image_height)
    w = (x2 - x1) / image_width
    h = (y2 - y1) / image_height
    return x, y, w, h

for cl in label:
    cnt = 0
    cnt_img = 0
    for file in os.listdir(f'./videoposeahand/{cl}'):
        lm_list = []
        print(f'cnt: {cnt}, lm_list: {lm_list}')
        print(f'processing: ./video/{cl}/{file}')
        cap = cv2.VideoCapture(f'./videoposeahand/{cl}/{file}')
        cap.set(640,640)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hand = hands.process(frameRGB)
            results_pose = pose.process(frameRGB)
            
            if results_hand.multi_hand_landmarks or results_pose.pose_landmarks:
                hand_landmarks = results_hand.multi_hand_landmarks[0] if results_hand.multi_hand_landmarks else None
                pose_landmarks = results_pose.pose_landmarks if results_pose.pose_landmarks else None
                
                lm = make_dat(hand_landmarks, pose_landmarks)
                lm_list.append(lm)
                frame = draw_land(mpDraw, results_hand, results_pose, frame)
            
            cnt_img += 1
            # cv2.imshow(f'processing: ./video/{cl}/{file}', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        df = pd.DataFrame(lm_list)
        df.to_csv(f'./dataset/{cl}/{cl}_{cnt}.txt', index=False)
        cnt += 1
        cap.release()
        cv2.destroyAllWindows()
