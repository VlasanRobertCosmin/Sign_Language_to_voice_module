import socket

QUEST_IP = "192.168.0.215"
QUEST_PORT = 5051

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Sending UDP to {QUEST_IP}:{QUEST_PORT}")
print("Type messages and press Enter. Ctrl+C to quit.")

while True:
    try:
        msg = input("Send> ").strip()
        if not msg:
            continue

        sock.sendto(msg.encode("utf-8"), (QUEST_IP, QUEST_PORT))
        print("Sent:", msg)
    except KeyboardInterrupt:
        break

sock.close()