import socket, cv2, pickle,struct,time
import pyshine as ps
import subprocess
import re
output = subprocess.check_output("ipconfig", shell=True, text=True)
match = re.search(r"Ethernet adapter Radmin VPN:.*?IPv4 Address.*?: (\d+\.\d+\.\d+\.\d+)", output, re.DOTALL)
if match:
    ipv4_address = match.group(1)
else:
    print("Không tìm thấy địa chỉ IPv4 cho Radmin VPN.")

mode =  'send'
name = 'SERVER TRANSMITTING AUDIO'
audio,context= ps.audioCapture(mode=mode)

server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = ipv4_address
port = 4982
backlog = 5
socket_address = (host_ip,port)
print('STARTING SERVER AT',socket_address,'...')
server_socket.bind(socket_address)
server_socket.listen(backlog)

while True:
	client_socket,addr = server_socket.accept()
	print('GOT CONNECTION FROM:',addr)
	if client_socket:

		while(True):
			frame = audio.get()
			
			a = pickle.dumps(frame)
			message = struct.pack("Q",len(a))+a
			client_socket.sendall(message)
			
	else:
		break

client_socket.close()