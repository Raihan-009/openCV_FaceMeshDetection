# openCV_FaceMeshDetection
openCV mediapipe Based Project

## Necessarry dependencies
<p> You can simply pip to install necessarry module. </p>

<code>pip install opencv-python</code>

<code>pip install mediapipe</code>

-----------------------------------
MediaPipe Face Mesh
-----------------------------------

<p>MediaPipe Face Mesh is a face geometry solution that estimates 468 3D face landmarks in real-time even on mobile devices. It employs machine learning (ML) to infer the 3D surface geometry, requiring only a single camera input without the need for a dedicated depth sensor. Utilizing lightweight model architectures together with GPU acceleration throughout the pipeline, the solution delivers real-time performance critical for live experiences.</p>

<p align = "center">
    <img src = "https://github.com/Raihan-009/openCV_FaceMeshDetection/blob/main/facialLandmarks.jpeg">
</p>

---------------------------------------------------
Project FaceMesh Detection with MediaPipe
---------------------------------------------------

> For Static FaceMesh Detection 

```python
import cv2
import mediapipe
import meshTracker as mt

frame = cv2.imread("demoface.jpeg")
tracker = mt.MeshDetection()
_, img = tracker.findFaceMesh(frame)
cv2.imshow("Framing", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<p align = "center">
    <img src = "https://github.com/Raihan-009/openCV_FaceMeshDetection/blob/main/results/staticFaceMeshDetection.png">
</p>


> For Dynamic or Real Time FaceMesh Detection

```python
import cv2
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
```

-----------------------------------
DLIB C++ Library
-----------------------------------

<p>Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems. It is used in both industry and academia in a wide range of domains including robotics, embedded devices, mobile phones, and large high performance computing environments. </p>
To get know more about dlib visit [](http://dlib.net/)


