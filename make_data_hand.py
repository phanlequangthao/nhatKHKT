import os
import shutil
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import json 
mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils
label = []
with open("labels.json", "r", encoding="utf-8") as f:
    label = json.load(f)
# label = ['a', 'b', 'c', 'o', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
#          'l', 'm', 'n', 'p', 'q', 'r', 's', 'space', 't', 'u',
#          'v', 'w', 'x', 'y', 'z', 'yes', 'no', 'me', 'you', 'hello',
#          'i_love_you', 'thank_you', 'sorry', 'do', 'eat', 'what', 'why', 
#          'who', 'where', 'when', 'how', 'how_much', 'go', 'happy', 
#          'sad', 'angry', 'good', 'bad']

# if os.path.exists('./dataset'):
#     shutil.rmtree('./dataset')

# for cl in label:
#     os.makedirs(f'./dataset/{cl}')

def make_dat(hand_landmarks):
    lm_list = []
    landmarks = hand_landmarks.landmark
    
    base_x = landmarks[0].x 
    base_y = landmarks[0].y
    base_z = landmarks[0].z
    #tọa độ x y z trước khi chuẩn hóa 
    center_x = np.mean([lm.x for lm in landmarks])
    center_y = np.mean([lm.y for lm in landmarks])
    center_z = np.mean([lm.z for lm in landmarks])
    #tính tọa độ trung tâm của x y z
    distances = [np.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2 + (lm.z - center_z)**2) for lm in landmarks[1:]]
    #tính khoảng cách giữa các điểm x_i y_i z_i với tọa độ trung tâm bằng khoảng cách euclid
    scale_factors = [1.0 / dist for dist in distances]
    #hệ số tỉ lệ hóa
    lm_list.append(0.0)
    lm_list.append(0.0)
    lm_list.append(0.0)
    lm_list.append(landmarks[0].visibility)
    #đặt gốc tọa độ Oxyz 
    for lm, scale_factor in zip(landmarks[1:], scale_factors):
        lm_list.append((lm.x - base_x) * scale_factor)
        lm_list.append((lm.y - base_y) * scale_factor)
        lm_list.append((lm.z - base_z) * scale_factor)
        lm_list.append(lm.visibility)
    #các tọa độ mới
    print(f"len lmlist: {len(lm_list)}")  
    return lm_list

def draw_land(mpDraw, results, img):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                img,
                hand_landmarks,
                mphands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
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
    video_folder = f'./videohandpersonality/{cl}'
    output_folder = f'./datasetpersonality/{cl}'

    if not os.path.exists(video_folder):
        print(f"[SKIP] Không có thư mục video: {video_folder}")
        continue

    os.makedirs(output_folder, exist_ok=True)
    cnt = 0
    cnt_img = 0
    for file in os.listdir(f'./videohandpersonality/{cl}'):
        lm_list = []
        print(f'cnt: {cnt}, lm_list: {lm_list}')
        print(f'processing: ./videohandpersonality/{cl}/{file}')   
        cap = cv2.VideoCapture(f'./videohandpersonality/{cl}/{file}')
        cap.set(3, 640)
        cap.set(4, 480)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frameRGB)
            if results.multi_hand_landmarks:
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # bbox = get_hand_bbox(hand_landmarks, width, height)
                    # cv2.rectangle(frameRGB, bbox[0], bbox[1], (255, 255, 255), 1)
                    # x, y, w, h = convert_coordinates(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1],width, height)
                    # filename = f"data/image/{cl}_{cnt_img}.jpg"
                    # cv2.imwrite(filename, frame)
                    # label_filename = os.path.join ("data/label/", f"{cl}_{cnt_img}.txt")
                    # with open(label_filename, 'a') as label_file:
                    #         label_file.write(f'{label.index(cl)} {x} {y} {w} {h} \n')
                    lm = make_dat(hand_landmarks)
                    lm_list.append(lm)
                    frame = draw_land(mpDraw, results, frame)
            cnt_img += 1
            # cv2.imshow(f'processing: ./videohand/{cl}/{file}', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        df = pd.DataFrame(lm_list)
        df.to_csv(f'./datasetpersonality/{cl}/{cl}_{cnt}.txt', index=False)
        cnt += 1
        cap.release()
        cv2.destroyAllWindows()