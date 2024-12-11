import imagiz
import cv2
import argparse

parser = argparse.ArgumentParser(description='c')
parser.add_argument('--server_ip', type=str, required=True, help='i')
args = parser.parse_args()

server_ip = args.server_ip

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