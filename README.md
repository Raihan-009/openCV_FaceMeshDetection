# openCV_FaceMeshDetection
openCV mediapipe Based Project


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

## Necessarry dependencies
<p> You can simply pip to install necessarry module. </p>

<code>pip install opencv-python</code>

<code>pip install mediapipe</code>


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

<p>Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems. It is used in both industry and academia in a wide range of domains including robotics, embedded devices, mobile phones, and large high performance computing environments. To get know more about dlib visit <a href = "http://dlib.net/"> dlib c++ library.</a> </p>


<p align = "center">
    <img src = "https://github.com/Raihan-009/openCV_FaceMeshDetection/blob/main/68_points_face_landmark.png">
</p>

---------------------------------------------------
Project FaceMesh Detection with dlib
---------------------------------------------------


## Necessarry dependencies
<p> You can simply pip to install necessarry module. </p>

<code>pip install cmake</code>

<code>pip install dlib</code>


> For Static FaceMesh Detection using dlib


```python
import cv2
import dlib

img = cv2.imread("demoface.jpeg")

hog_face_detector = dlib.get_frontal_face_detector()
dlib_faceial_landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = hog_face_detector(grayImg)

for face in faces:
    facial_landmarks = dlib_faceial_landmarks(grayImg, face)

    for n in range(0,68):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y

        cv2.circle(img,(x,y),2,(0,0,255),2)
    cv2.imshow("Framing", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

<p align = "center">
    <img src = "https://github.com/Raihan-009/openCV_FaceMeshDetection/blob/main/results/staticFaceMeshDetectionDLIB.png">
</p>



> For Dynamic or Real Time FaceMesh Detection using dlib

```python
import cv2
import dlib

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()
dlib_faceLandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = hog_face_detector(gray)
        allfaces = []
        for face in faces:
            face_landmarks = dlib_faceLandmark(gray,face)
            allface = []
            for n in range(0,68):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                allface.append([n+1,x,y])
                cv2.circle(frame, (x,y), 1, (255,0,0), 2)
                cv2.putText(frame, str(n+1), (x,y), cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),1)
            allfaces.append(allface)
        cv2.imshow("Hola!", frame)
        print(allface)
        print(len(allfaces))
    else:
        break
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
