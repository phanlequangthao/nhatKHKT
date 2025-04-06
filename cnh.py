import json
import os
import cv2
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QProgressBar
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# Thêm class Worker để chạy các tác vụ nặng
class Worker(QThread):
    finished = pyqtSignal()
    
    def run(self):
        os.system("python make_data_hand.py")
        os.system("python train.py")
        self.finished.emit()

class PersonalityDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Personalize Data")
        self.setFixedSize(800, 600)

        # UI
        self.layout = QVBoxLayout()
        self.camera_label = QLabel(self)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.camera_label)

        self.word_input = QLineEdit(self)
        self.word_input.setPlaceholderText("Enter word to collect")
        self.layout.addWidget(self.word_input)

        self.start_btn = QPushButton("Start", self)
        self.start_btn.clicked.connect(self.start_collection)
        self.layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.clicked.connect(self.stop_collection)
        self.stop_btn.setEnabled(False)
        self.layout.addWidget(self.stop_btn)

        self.done_btn = QPushButton("Done", self)
        self.done_btn.clicked.connect(self.finalize_collection)
        self.done_btn.setEnabled(False)
        self.layout.addWidget(self.done_btn)

        # Thêm progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

        self.setLayout(self.layout)

        # Vars
        self.cap = None
        self.timer = QTimer()
        self.collected_words = []
        self.current_word = None
        self.is_recording = False
        self.video_writer = None

        self.personality_folder = "videohandpersonality"
        if not os.path.exists(self.personality_folder):
            os.makedirs(self.personality_folder)

        # Worker thread
        self.worker = Worker()
        self.worker.finished.connect(self.on_worker_finished)

    def start_collection(self):
        word = self.word_input.text().strip()
        if not word:
            QMessageBox.warning(self, "Warning", "Please enter a word!")
            return

        self.current_word = word
        if word not in self.collected_words:
            self.collected_words.append(word)

        word_folder = os.path.join(self.personality_folder, word)
        os.makedirs(word_folder, exist_ok=True)

        # Unique file name
        index = 1
        while os.path.exists(os.path.join(word_folder, f"{word}_{index}.mp4")):
            index += 1
        self.video_path = os.path.join(word_folder, f"{word}_{index}.mp4")

        # Camera
        self.cap = cv2.VideoCapture(1)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.word_input.setEnabled(False)
        self.is_recording = True

    def stop_collection(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.word_input.setEnabled(True)
        self.done_btn.setEnabled(True)
        self.is_recording = False

    def finalize_collection(self):
        if not self.collected_words:
            QMessageBox.warning(self, "Warning", "No words collected!")
            return

        # Kiểm tra xem có tồn tại labels.json hay không
        if os.path.exists("labels.json"):
            # Hiển thị hộp thoại để hỏi có muốn dùng labels cũ hay tạo mới
            reply = QMessageBox.question(self, "Use Old Labels?", 
                                        "Do you want to use the existing labels? If not, the current labels will overwrite the old ones.",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                # Nếu chọn No, thì sẽ tạo labels mới từ collected_words
                with open("labels.json", "w") as f:
                    json.dump(self.collected_words, f)
            else:
                # Nếu chọn Yes, giữ nguyên labels.json cũ
                pass
        else:
            # Nếu labels.json không tồn tại, tạo mới
            with open("labels.json", "w") as f:
                json.dump(self.collected_words, f)

        # Hiển thị progress bar và vô hiệu hóa nút Done
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        self.done_btn.setEnabled(False)

        # Bắt đầu worker thread
        self.worker.start()


    def on_worker_finished(self):
        # Ẩn progress bar và hiển thị thông báo
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "Success", "Personalization completed!")
        self.close()

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                return

            self.display_frame(frame)
            if self.is_recording:
                self.save_frame(frame)

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
        self.camera_label.setPixmap(QPixmap.fromImage(p))

    def save_frame(self, frame):
        if not self.video_writer:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 20.0, (640, 480))
        self.video_writer.write(frame)

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        event.accept()

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    dialog = PersonalityDialog()
    dialog.show()
    sys.exit(app.exec_())
