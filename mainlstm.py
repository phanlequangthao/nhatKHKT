from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.uic import loadUi
import cv2
import sys
import imagiz
import speech_recognition as sr
from moviepy.editor import  ImageSequenceClip
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
from win32com.client import Dispatch
import os
import socket
import threading
import subprocess
import tensorflow as tf
from keras.models import load_model
import base64
import time

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
mphands = mp.solutions.hands
hands = mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
mppose = mp.solutions.pose
pose = mppose.Pose()
speak = Dispatch("SAPI.SpVoice").Speak
server=imagiz.Server()
host = '26.64.220.173'
port = 12345

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host, port))
class SpeechToVideoThread(QThread):
    video = pyqtSignal(QImage)
    audioTextChanged = pyqtSignal(str)
    def __init__(self, img_dir, video_output_path):
        super(SpeechToVideoThread, self).__init__()
        self.img_dir = img_dir
        self.video_output_path = video_output_path
        self.audio_text = ""
        self.is_recording = False
    def run(self):
        r = sr.Recognizer()
        m = sr.Microphone()
        print("A moment of silence, please...")
        with m as source: r.adjust_for_ambient_noise(source)
        print("Set minimum energy threshold to {}".format(r.energy_threshold))
        while self.is_recording:
            print("Say something!")
            with m as source: audio = r.listen(source)
            print("Got it! Now to recognize it...")
            try:
                self.audio_text = r.recognize_google(audio)
                print("You said {}".format(self.audio_text))
                self.create_video_from_text()
                self.audioTextChanged.emit(self.audio_text)
            except sr.UnknownValueError:
                print("Oops! Didn't catch that")
            except sr.RequestError as e:
                print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
    def start_recording(self):
        self.is_recording = True #Em đánh dấu cho biến is_recording là đang hoạt động
        self.start() 

    def stop_recording(self):
        self.is_recording = False
        self.wait()
        
    # hàm thêm đường dẫn ảnh từ văn bản nhận diện được
    def create_video_from_text(self):
        print("in create_video_from_text")
        print(self.audio_text)
        img_list = []
        for char in self.audio_text.lower():
            if char != ' ':
                img_path = os.path.join(self.img_dir, f"{char}.jpg").replace('\\', '/')
                if os.path.exists(img_path):
                    img_list.append(img_path)
            elif char == ' ':
                img_path = os.path.join(self.img_dir, 'space.jpg').replace('\\', '/')
                if os.path.exists(img_path):
                    img_list.append(img_path)
            else:
                continue
        print("Image List:", img_list)
        if img_list:
            self.show_video(img_list)
            self.audioTextChanged.emit("Video created!")
            print("Done")
    #tạo video bằng ảnh ngôn ngữ ký hiệu
    def show_video(self, img_list):
        frame_list = []
        for img_path in img_list:
            frame = cv2.imread(img_path)
            if frame is not None:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(rgb_image)

        if frame_list:
            fps = 0.25
            clip = ImageSequenceClip(frame_list, fps=fps)
            clip.write_videofile(self.video_output_path, codec='libx264', fps=fps)
class Video(QThread):
    vid = pyqtSignal(QImage)
    def run(self):
        self.hilo_corriendo = True
        video_path = r"C:\Users\chojl\Pictures\Camera Roll\a2.mp4"
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate
        delay = int(1000 / fps)  # Calculate delay between frames
        while self.hilo_corriendo:
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(890, 440, Qt.KeepAspectRatio)
                self.vid.emit(p)
                
                self.msleep(delay)  # Introduce delay to match the frame rate
        cap.release()
    def stop(self):
        self.hilo_corriendo = False
        self.quit()


class Video2(QThread):
    vid2 = pyqtSignal(QImage)

    def run(self):
        self.check = True
        video_path = r"a2.mp4"
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS) 
        delay = int(1000 / fps)
        
        while self.check:
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(890, 440, Qt.KeepAspectRatio)
                self.vid2.emit(p)
                
                self.msleep(delay)  # Introduce delay to match the frame rate
        cap.release()

    def stop(self):
        self.check = False
        self.quit()



class Ham_Camera(QThread):
    luongPixMap1 = pyqtSignal(QImage)
    luongPixMap2 = pyqtSignal(QImage)
    luongString1 = pyqtSignal(str)
    luongString2 = pyqtSignal(str)
    luongClearSignal = pyqtSignal()
    checkTrungChanged = pyqtSignal(str)

    def __init__(self):
        super(Ham_Camera, self).__init__()
        self.checkTrung = ""
        self.trangThai = True
        self.string = ""
        self.string2 = ""
        self.f_cnt_threshold = 20
        self.current_f_cnt = 0
        self.luongString1.connect(self.update_string1)
        self.luongString2.connect(self.update_string2)
        self.luongClearSignal.connect(self.clear_string)

        self.latest_frame_from_server = None 

    def update_string1(self, new_string):
        self.string = new_string

    def update_string2(self, new_string):
        self.string2 = new_string

    def clear_string(self):
        self.string = ""

    @staticmethod
    

    def make_dat(hand_landmarks):
        lm_list = []
        NUM_HAND_LANDMARKS = 21
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

    @staticmethod
    def detect(model, lm_list):
        lm_list = np.array(lm_list)
        lm_list = np.expand_dims(lm_list, axis=0)
        results = model.predict(lm_list)
        predicted_label_index = np.argmax(results, axis=1)[0]
        classes = ['a', 'b', 'c', 'o', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
         'l', 'm', 'n', 'p', 'q', 'r', 's', 'space', 't', 'u',
         'v', 'w', 'x', 'y', 'z', 'yes', 'no', 'me', 'you', 'hello',
         'i_love_you', 'thank_you', 'sorry', 'do', 'eat', 'what', 'why', 
         'who', 'where', 'how_much', 'go', 'happy', 'sad', 'bad']
        confidence = np.max(results, axis=1)[0]
        if confidence > 0.95:
            temp = classes[predicted_label_index]
            if temp == "space":
                label = " "
            else:
                label = temp.replace("_", " ")
        else:
            label = "cant detect"

    def run(self):
        model = load_model('./model/best_model_12.h5')

        cap = cv2.VideoCapture(camera_index)
        cap.set(3, 640)
        cap.set(4, 480)
        lm_list = []

        threading.Thread(target=self.receive_frame, daemon=True).start()

        f_cnt = 0 
        while self.trangThai:
            ret, frame1 = cap.read()
            frame2 = self.latest_frame_from_server
            if ret:
                image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                cv2.imwrite('shared_frame.jpg', image1)
                results_hand = hands.process(image1)
                image2 = frame2
                if results_hand.multi_hand_landmarks:
                    hand_landmarks = results_hand.multi_hand_landmarks[0] if results_hand.multi_hand_landmarks else None
                    if hand_landmarks:
                        lm = self.make_landmark_timestep(hand_landmarks)
                        lm_list.append(lm)
                        if len(lm_list) == 9:
                            label = self.detect(model, lm_list)
                            lm_list = []

                            if label != "neutral":
                                if label == self.checkTrung:
                                    f_cnt += 1  
                                else:
                                    f_cnt = 1  
                                    self.checkTrung = label  
                                if f_cnt >= 10:  
                                    if label == "space":
                                        self.string += " "
                                    else:
                                        self.string += label
                                    self.luongString1.emit(self.string)
                                    self.checkTrungChanged.emit(self.checkTrung)

                image1.flags.writeable = True
                try:
                    for hand_landmarks in results_hand.multi_hand_landmarks:
                        mpDraw.draw_landmarks(
                            image1, hand_landmarks, mphands.HAND_CONNECTIONS,
                            mpDraw.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                            mpDraw.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                        )
                except:
                    print("ko co landmark")

                h, w, ch = image1.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(image1.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                self.luongPixMap1.emit(p)

                if frame2 is not None:
                    h2, w2, ch2 = image2.shape
                    bytes_per_line2 = ch2 * w2
                    convert_to_Qt_format2 = QImage(image2.data, w2, h2, bytes_per_line2, QImage.Format_RGB888)
                    p2 = convert_to_Qt_format2.scaled(640, 480, Qt.KeepAspectRatio)
                    self.luongPixMap2.emit(p2)
            else:
                break
        cap.release()

    def receive_frame(self):
        while self.trangThai:
            message_cam = server.receive()
            # frame2 = cv2.imdecode(message_cam.image, 1)
            frame2 = cv2.imdecode(np.frombuffer(message_cam.image, np.uint8), 1)
            self.latest_frame_from_server = frame2

    def stop(self):
        self.trangThai = False

    def get_hand_bbox(self, landmarks, image_width, image_height):
        x_min, x_max, y_min, y_max = float('inf'), 0, float('inf'), 0

        for landmark in landmarks.landmark:
            x, y = int(landmark.x * image_width), int(landmark.y * image_height)
            x_min = min(x_min, x) - 3
            x_max = max(x_max, x) + 1
            y_min = min(y_min, y) - 3
            y_max = max(y_max, y) + 1

        bbox = ((x_min, y_min), (x_max, y_max))
        return bbox


"""
Ham_Camera được sử dụng để khởi tạo webcam và chạy mô hình dự đoán của dự án, hình ảnh được ghi nhận từ webcam sẽ được
chuyển thành hình ảnh sau đó cập nhật lên label_cam, tốc dộ cập nhật gần như bằng với thời gian thực
"""
class Ham_Chinh(QMainWindow):
    messageSent = pyqtSignal(str)
    # Lớp Ham_Chinh là lớp chính của chương trình, chịu trách nhiệm khởi tạo các thành phần giao diện và kết nối các tín hiệu giữa các lớp.
    def __init__(self):
        # Gọi hàm khởi tạo của lớp QMainWindow
        super(Ham_Chinh, self).__init__()
        # Tải giao diện từ file ui.ui
        loadUi('main.ui', self)
        
        # Khởi tạo luồng camera
        self.Work = Video()
        self.Work2 = Video2()
        self.thread_camera = Ham_Camera()
        self.thread_camera.luongClearSignal.connect(self.process_string)
        self.thread_camera.checkTrungChanged.connect(self.handle_check_trung_changed)
        # Khởi tạo luồng video
        self.img_dir = r'D:\a\img'
        self.video_output_path = r'output_video.mp4'
        self.thread_vid = SpeechToVideoThread(self.img_dir, self.video_output_path)
        # Kết nối tín hiệu luongPixMap của luồng camera với hàm setCamera
        self.thread_camera.luongPixMap1.connect(self.setCamera1)
        self.thread_camera.luongPixMap2.connect(self.setCamera2)
        # Kết nối tín hiệu startcam của nút startcam với hàm khoiDongCamera
        self.startcam.clicked.connect(self.khoiDongCamera)
        # Kết nối tín hiệu pausecam của nút pausecam với hàm tamDungCamera
        self.pausecam.clicked.connect(self.tamDungCamera)
        # Kết nối tín hiệu clear của nút clear với hàm xoaToanBo
        self.clear.clicked.connect(self.xoaToanBo)
        # Kết nối tín hiệu delete_2 của nút delete_2 với hàm xoaChu
        self.delete_2.clicked.connect(self.xoaChu)
        self.space.clicked.connect(self.spacee)
        self.check.clicked.connect(self.checkk)
        self.send.clicked.connect(self.sendMess)
        self.messageSent.connect(self.send_message)
        #Kết nối tín hiệu speak với hàm nói ra văn bản
        # message = client.recv(1024).decode()
        # self.text2.setText(message)
        # Kết nối tín hiệu luongString1 của luồng camera với hàm setText của label text
        self.thread_camera.luongString1.connect(self.text1.setText)
        self.thread_camera.luongString2.connect(self.text2.setText)
        #voice to text/video
        self.record_button.clicked.connect(self.start_recording)
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.stop_record_button.setEnabled(False)
        self.play_video.clicked.connect(self.start_video)
        self.send_video.clicked.connect(self.sendVideo)
        # self.stop_video.clicked.connect(self.stop_vide)
        self.thread_vid.audioTextChanged.connect(self.text_2.setText)


        self.listen_thread = threading.Thread(target=self.listen_for_messages)
        self.listen_thread.start()
        
        
    def sendVideo(self):
        client.send("START_VIDEO".encode())
        file_name = r'i.mp4'
        file_size = os.path.getsize(file_name)
        time.sleep(5)
        client.send(f"{file_name}|{file_size}".encode())

        # Opening file and sending data.
        with open(file_name, "rb") as file:
            c = 0
            i = 0
            while c <= file_size:
                data = file.read(1024)
                if not (data):
                    break
                # print(i)
                i += 1
                client.sendall(data)
                c += len(data)
                print(c)
            print("done")
        

    def start_video(self):
        self.Work.start()
        # self.Work2.start()
        self.Work.vid.connect(self.Imageupd_slot)
        # self.Work2.vid2.connect(self.vidletter)
    def listen_for_messages(self):
        while True:
            try:
                message = client.recv(1024).decode()
                print(f"Recv: {message}")
                if "OTHER_USER_IP:" in message:
                    other_user_ip = message.split(":")[1]
                    subprocess.Popen([sys.executable, "client_camera.py", "--server_ip", other_user_ip])
                    subprocess.Popen([sys.executable, "test_client.py", "--host_ip", other_user_ip])
                    subprocess.Popen([sys.executable, "test_server.py"])
                    print("done")
                elif message == "START_VIDEO":
                    print("Video data incoming...")
                    self.receive_video_data()
                else:
                    self.text2.setText(message)
            except UnicodeDecodeError as e:
                print("Unicode decode error occurred: ", e)
            except Exception as e:
                print("An error occurred!", e)
                client.close()
                break


    def receive_video_data(self):
        try:
            # Nhận tên file và kích thước file
            file_info = client.recv(1024).decode()
            file_name, file_size = file_info.split("|")
            file_size = int(file_size)
        except Exception as e:
            print("Error receiving file info:", e)
            return
        
        with open(file_name, "wb") as file:
            c = 0

            while c <= int(file_size):
                data = client.recv(1024)
                if not (data):
                    break
                file.write(data)
                c += len(data)
        
        self.Work2.start()
        self.Work2.vid2.connect(self.vidletter)


    def sendMess(self):
        mess = self.text1.text()
        self.messageSent.emit(mess)

    def send_message(self, message):
        client.send(message.encode())
        print("Sent message successfully")
    def Imageupd_slot(self, Image):
        self.img_label.setPixmap(QPixmap.fromImage(Image))
    def vidletter(self, Image):
        self.img_label_2.setPixmap(QPixmap.fromImage(Image))
    # def stop_vide(self):
    #     self.Work.stop()
    #     self.Work2.stop()
    def setCamera1(self, image1):
        # Cập nhật hình ảnh lên label cam
        self.camera1.setPixmap(QPixmap.fromImage(image1))
    def setCamera2(self, image2):
        # Cập nhật hình ảnh lên label cam
        self.camera2.setPixmap(QPixmap.fromImage(image2))
    def khoiDongCamera(self):
        # Khởi động luồng camera để bắt đầu nhận diện vật thể
        self.thread_camera.start()
    def tamDungCamera(self):
        # Dừng luồng camera để tạm dừng nhận diện vật thể
        self.thread_camera.stop()
        # Chờ luồng camera hoàn toàn dừng trước khi tiếp tục
        self.thread_camera.wait()
    def xoaToanBo(self):
        # Xóa toàn bộ nội dung trong label text
        self.thread_camera.luongClearSignal.emit()
        self.text1.setText()
    def process_string(self):
        # Truy cập và xử lý giá trị từ Ham_Camera
        self.thread_camera.string = ""  
        # Cập nhật giá trị trong Ham_Camera
        self.thread_camera.luongString1.emit(self.thread_camera.string)
        self.text1.setText()
    def xoaChu(self):
        # Xóa ký tự cuối cùng trong textt
        textt = self.text1.text()  
        textt = textt[:-1]
        print(textt)
        # Cập nhật textt lên label text
        self.text1.setText(textt)
        self.thread_camera.luongString1.emit(textt)
    def spacee(self):
        # Xóa ký tự cuối cùng trong textt
        textt = self.text1.text()  
        textt = textt + " "
        print(textt)
        # Cập nhật textt lên label text
        self.text1.setText(textt)
        self.thread_camera.luongString1.emit(textt)
    def checkk(self):
        self.thread_camera.checkTrung = ""
        self.thread_camera.checkTrungChanged.emit(self.thread_camera.checkTrung)
    def handle_check_trung_changed(self, new_check_trung):
        if new_check_trung == "":
            print("checkTrung done")
    def start_recording(self):
        self.record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        self.thread_vid.start_recording()
    def stop_recording(self):
        self.record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        self.thread_vid.stop_recording()
    

if __name__ == '__main__':
    room_code = input("Enter room code or type 'NEW' for a new room: ")
    client.send(room_code.encode())
    server_message = client.recv(1024).decode()
    print(server_message)
    camera_index = int(input("Nhập số camera bạn muốn chọn, 0 là webcam mặc định: "))
    if "Connected to room" in server_message:
        app = QApplication(sys.argv)
        window = Ham_Chinh()
        window.setWindowTitle('MainApp')
        window.show()
        sys.exit(app.exec_())