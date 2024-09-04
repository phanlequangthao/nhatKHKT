import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
import csv
import os
import numpy as np
num_coords = 21
landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]
with open('data.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

def get_hand_bbox(landmarks, image_width, image_height):
        x_min, x_max, y_min, y_max = float('inf'), 0, float('inf'), 0

        for landmark in landmarks.landmark:
            x, y = int(landmark.x * image_width), int(landmark.y * image_height)
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
label = ['a', 'b', 'c', 'o', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
         'l', 'm', 'n', 'p', 'q', 'r', 's', 'space', 't', 'u',
         'v', 'w', 'x', 'y', 'z', 'yes', 'no', 'me', 'you', 'hello',
         'i_love_you', 'eat', 'thank_you', 'little', 'sorry', 'drink',
         '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

class_name = "b"
cnt = 1
cap = cv2.VideoCapture(0)
cap.set(640,640)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        H, W, _ = frame.shape
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2)
                                 )
        try:
            rh = results.right_hand_landmarks.landmark
            rh_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in rh]).flatten())
            row = rh_row
            row.insert(0, class_name)
            with open(r'data.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            if results.right_hand_landmarks:
                    bbox = get_hand_bbox(results.right_hand_landmarks, W, H)
                    cv2.rectangle(image, bbox[0], bbox[1], (255, 255, 255), 1)
                    print(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
                    x, y, w, h = convert_coordinates(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1],width, height)
                    print(cnt)
                    filename = f"data/image/{class_name}_{cnt}.jpg"
                    cv2.imwrite(filename, frame)
                    #f"data/label/{class_name}_{cnt}.jpg"
                    label_filename = os.path.join ("data/label/", f"{class_name}_{cnt}.txt")
                    with open(label_filename, 'a') as label_file:
                        label_file.write(f'{label.index(class_name)} {x} {y} {w} {h} \n')
                    cnt += 1
                    if(cnt == 1200):
                        exit()
        except:
            pass
        cv2.imshow('dt', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()