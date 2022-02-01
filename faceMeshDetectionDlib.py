import cv2
import dlib

cap = cv2.VideoCapture("Video.mp4")

hog_face_detector = dlib.get_frontal_face_detector()
dlib_faceLandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = hog_face_detector(gray)
        # print(faces)
        # print(type(faces))
        allfaces = []
        for face in faces:
            face_landmarks = dlib_faceLandmark(gray,face)
            # print(face_landmarks)
            # print(type(face_landmarks))
            allface = []
            for n in range(0,68):
                x = face_landmarks.part(n).x
                # print(x)
                #print(type(x))
                y = face_landmarks.part(n).y
                # print(y)
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