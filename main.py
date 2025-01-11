import cv2
import mediapipe as mp

cap = cv2.VideoCapture('Video1.mp4')

while True:
    success, img = cap.read()
    cv2.imshow("img", img)
