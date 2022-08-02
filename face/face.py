import cv2 
import sys, os
import matplotlib.pyplot as plt

'''
    захват лица из видео
'''
# dependency: python -m pip install opencv-python 
# Признаки https://github.com/opencv/opencv/tree/4.x/data/haarcascades
# https://github.com/opencv/opencv/tree/master/data/haarcascades
print(cv2.__version__)

def blur_face(frame_slice):  
    (h,w) = frame_slice.shape[:2]
    dW = int(w/3.0)
    dH = int(h/3.0)
    if dW % 2 == 0:
        dW -= 1
    if dH % 2 == 0:
        dH -= 1    
    return cv2.GaussianBlur(frame_slice,(dW,dH),0)

def face_detected(params):
    title = "Face detected"
    cv2.namedWindow(title) 
    cap = cv2.VideoCapture("/container_data/source/video/gigant.mp4") # 0 - запущенная камера
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,500)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,500)
    face_cascade = cv2.CascadeClassifier('/container_data/source/tools/haarcascades_4.5.5/haarcascade_frontalface_default.xml')

    if cap.isOpened(): # try to get the first frame 
        rval, frame = cap.read() 
    else: 
        rval = False 

    while rval: 
        # Обнаружение по признаку
        # scaleFactor - фактор размера искомых обьектов относительно применяемой модели признаков
        # minNeighbors - как много соседних обьектов рядом
        faces = face_cascade.detectMultiScale(frame,scaleFactor=2,minNeighbors=5,minSize=(20,20))
        # выделение признака
        for (x, y, w, h) in faces:
            # показать область контуром
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=2)
            # показать область розмытием
            frame[y:y+h,x:x+w] = blur_face(frame[y:y+h,x:x+w]) # размыть признак
    
        # показать кадр
        cv2.imshow(title, frame) 
        rval, frame = cap.read() 

        # Exit
        key = cv2.waitKey(20) & 0xFF
        if key == 27: # exit on ESC break 
            cap.release()
            cv2.destroyWindow(title)

# ----------------------------------------------------------------------------------------------------------------
'''
RetinaFace

https://github.com/serengil/retinaface/blob/master/tests/unit-tests.py

Другие алгоритмы:
https://habr.com/ru/post/661671/
DBFace https://github.com/dlunion/DBFace
Face Recognition https://github.com/ageitgey/face_recognition
Ultra-Light-Fast-Generic-Face-Detector-1MB https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

Error:
    The TensorFlow library was compiled to use AVX instructions, but these aren't available on your machine.
    Aborted (core dumped)

'''    

# from retinaface import RetinaFace

def int_tuple(t):
    return tuple(int(x) for x in t)

def face_detect_retinaface(params):
    img_path = "/container_data/source/faces.jpeg"
    img = cv2.imread(img_path)
    resp = RetinaFace.detect_faces(img_path, threshold = 0.1)

    for key in resp:
        identity = resp[key]

        #---------------------
        confidence = identity["score"]

        rectangle_color = (255, 255, 255)

        landmarks = identity["landmarks"]
        diameter = 1
        cv2.circle(img, int_tuple(landmarks["left_eye"]), diameter, (0, 0, 255), -1)
        cv2.circle(img, int_tuple(landmarks["right_eye"]), diameter, (0, 0, 255), -1)
        cv2.circle(img, int_tuple(landmarks["nose"]), diameter, (0, 0, 255), -1)
        cv2.circle(img, int_tuple(landmarks["mouth_left"]), diameter, (0, 0, 255), -1)
        cv2.circle(img, int_tuple(landmarks["mouth_right"]), diameter, (0, 0, 255), -1)

        facial_area = identity["facial_area"]

        cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), rectangle_color, 1)
        #facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
        #plt.imshow(facial_img[:, :, ::-1])

    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    plt.show()


def main(params):
    if True:
        face_detected(params)

 
if __name__ == "__main__":
    main(sys.argv[1:])

# Run:    
# python  /container_data/face.py 