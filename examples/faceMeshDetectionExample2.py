import cv2
import mediapipe
import meshTracker as mt

frame = cv2.imread("demoface.jpeg")
tracker = mt.MeshDetection()
_, img = tracker.findFaceMesh(frame)
cv2.imshow("Framing", img)
cv2.waitKey(0)
cv2.destroyAllWindows()