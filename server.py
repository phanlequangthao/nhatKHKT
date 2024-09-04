import socket
import threading
import random

host = '26.64.220.173'
port = 12345

clients = {}
room_codes = {}
client_addresses = {}
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((host, port))
server.listen()
print(f"ok {host}:{port}")

def handle_client(conn, addr):
    print(f"{addr}")
    client_ip = addr[0]
    room_code = ""
    
    try:
        room_code = conn.recv(1024).decode()
        if room_code == 'NEW':
            room_code = str(random.randint(1000, 9999))
            while room_code in room_codes:
                room_code = str(random.randint(1000, 9999))
            room_codes[room_code] = []
        if room_code not in room_codes:
            conn.send("wrong code".encode())
            conn.close()
            return
        conn.send(f"Connected to room {room_code}".encode())
    except:
        conn.close()
        return
    
    room_codes[room_code].append(conn)
    clients[conn] = room_code
    client_addresses[conn] = client_ip  
    
    if len(room_codes[room_code]) == 2:
        ip_1, ip_2 = (client_addresses[client] for client in room_codes[room_code])
        room_codes[room_code][0].send(f"OTHER_USER_IP:{ip_2}".encode())
        room_codes[room_code][1].send(f"OTHER_USER_IP:{ip_1}".encode())

    while True:
        try:
            msg = conn.recv(1024)
            broadcast(msg, room_code, conn)
        except ConnectionResetError:
            break
        except Exception as e:
            print(f"Error: {e}")
            break
    
    try:
        conn.close()
        room_codes[room_code].remove(conn)
        del clients[conn]
        del client_addresses[conn]
    except (KeyError, ValueError):
        pass

def broadcast(msg, room_code, sender):
    for client in room_codes[room_code]:
        if client != sender:
            try:
                client.send(msg)
            except:
                client.close()
                try:
                    room_codes[room_code].remove(client)
                    del clients[client]
                    del client_addresses[client]
                except (KeyError, ValueError):
                    pass

def receive_video_data(conn, room_code):
    buffer_size = 4096
    while True:
        try:
            data = conn.recv(buffer_size)
            if data.endswith("END_VIDEO_DATA"):
                data = data[:-len("END_VIDEO_DATA")]
                broadcast(data, room_code, conn)
                break
            broadcast(data, room_code, conn)
        except Exception as e:
            print(f"Error receiving video data: {e}")
            break

def start():
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"cur connect {threading.active_count() - 1}")

start()
