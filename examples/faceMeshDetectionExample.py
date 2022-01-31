import cv2
from cv2 import trace
import mediapipe
import meshTracker as mt

cap = cv2.VideoCapture(0)
tracker = mt.MeshDetection()

while True:
    ret, frame = cap.read()
    if ret:
        _, img = tracker.findFaceMesh(frame)
        cv2.imshow("framing", img)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()