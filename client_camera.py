import imagiz
import cv2

server_ip = "26.213.15.26"

client = imagiz.Client("cc1", server_ip=server_ip)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
while True:
    try:
        frame = cv2.imread('shared_frame.jpg', 1)
        r, image = cv2.imencode('.jpg', frame, encode_param)
        client.send(image)
    except cv2.error as e:
        print(f"Error: {e}")
        continue