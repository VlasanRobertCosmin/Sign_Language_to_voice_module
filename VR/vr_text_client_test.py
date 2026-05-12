import socket
import json

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Python VR/Text client")
print("Type text and press ENTER.")
print("Type q to quit.")

while True:
    text = input("Text to sign: ")

    if text.lower() == "q":
        break

    payload = {
        "type": "text",
        "text": text
    }

    message = json.dumps(payload).encode("utf-8")
    sock.sendto(message, (UDP_IP, UDP_PORT))

    print(f"Sent: {payload}")