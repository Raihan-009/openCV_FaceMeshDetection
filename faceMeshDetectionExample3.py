import cv2
import dlib

img = cv2.imread("demoface.jpeg")

hog_face_detector = dlib.get_frontal_face_detector()
dlib_faceial_landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = hog_face_detector(grayImg)
# print(len(faces))

for face in faces:
    facial_landmarks = dlib_faceial_landmarks(grayImg, face)

    for n in range(0,68):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y

        cv2.circle(img,(x,y),2,(0,0,255),2)
    cv2.imshow("Framing", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()