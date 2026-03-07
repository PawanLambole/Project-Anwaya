import cv2
import time

print("Testing camera 0 without DSHOW")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera 0 without DSHOW")
else:
    ret, frame = cap.read()
    print("Read frame:", ret)
    cap.release()

print("Testing camera 0 with DSHOW")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera 0 with DSHOW")
else:
    ret, frame = cap.read()
    print("Read frame:", ret)
    cap.release()
