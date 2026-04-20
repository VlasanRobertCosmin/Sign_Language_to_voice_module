"""
Python UDP Receiver for ASL Recognition
=======================================
Receives JSON packets from the ASL recognition sender.

Expected packet format:
{
  "label": "hello",
  "confidence": 0.92,
  "top_3": [
    {"label": "hello", "confidence": 0.92},
    {"label": "thanks", "confidence": 0.05},
    {"label": "yes", "confidence": 0.02}
  ],
  "timestamp": 1712345678.123
}
"""

import socket
import json
import time

# Match these with the sender
UDP_IP = "0.0.0.0"   # Listen on all interfaces
UDP_PORT = 5005

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print("=" * 50)
    print("ASL UDP RECEIVER STARTED")
    print("=" * 50)
    print(f"Listening on {UDP_IP}:{UDP_PORT}")
    print("Waiting for packets...\n")

    while True:
        try:
            data, addr = sock.recvfrom(8192)
            message = data.decode("utf-8")

            try:
                payload = json.loads(message)

                label = payload.get("label", "unknown")
                confidence = payload.get("confidence", 0.0)
                top_3 = payload.get("top_3", [])
                timestamp = payload.get("timestamp", None)

                print("-" * 50)
                print(f"From: {addr[0]}:{addr[1]}")
                print(f"Label: {label}")
                print(f"Confidence: {confidence * 100:.1f}%")

                if timestamp is not None:
                    readable_time = time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(timestamp)
                    )
                    print(f"Timestamp: {readable_time}")

                if top_3:
                    print("Top 3 predictions:")
                    for i, item in enumerate(top_3, start=1):
                        sign = item.get("label", "unknown")
                        conf = item.get("confidence", 0.0)
                        print(f"  {i}. {sign} - {conf * 100:.1f}%")

            except json.JSONDecodeError:
                print("-" * 50)
                print(f"From: {addr[0]}:{addr[1]}")
                print("Raw message:")
                print(message)

        except KeyboardInterrupt:
            print("\nReceiver stopped.")
            break
        except Exception as e:
            print(f"Receiver error: {e}")

if __name__ == "__main__":
    main()