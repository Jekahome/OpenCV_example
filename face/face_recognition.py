
import numpy as np
import cv2 
import sys, os

'''
face recognition - распознает конкретное лицо

Обученнение модели определенных лиц в файле face/faces_train.py
'''
def face_recognition(params):
    try:
        window = "Detected Face"
        cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(window, 500, 500);
        
        haar_cascade = cv2.CascadeClassifier('/container_data/source/tools/haarcascades_4.5.5/haarcascade_frontalface_default.xml')

        # list имен в там же порядке как при обучении, так как алгоритм распознавания возвращает индекс label'a,а не имя label'a 
        people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Alexey Arestovich', 'Ilya Varlamov','Mark Feigin']
        # features = np.load('/container_data/face/tools/features.npy', allow_pickle=True)
        # labels = np.load('/container_data/face/tools/labels.npy')

        # Загрузка обученной модели
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read('/container_data/face/tools/face_trained.yml')


        for file in os.listdir(r'/container_data/face/tools/src'):
            # Обьект для распознавания
            img = cv2.imread(os.path.join(r'/container_data/face/tools/src',file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('Person', gray)

            # Найти лицо т.е. ROI и скормить его алгоритму распознавания
            faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
            name_save = ''
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h,x:x+w]
                label, confidence = face_recognizer.predict(faces_roi)
                print(f'Label = {people[label]} with a confidence of {confidence}')
                cv2.putText(img, str(people[label]), (x-w//2,y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), thickness=1)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=1)
                name_save+=people[label];name_save+="\n"
            if  name_save != '':   
                cv2.imwrite(f"/container_data/face/tools/dst/{name_save}.png", img,[cv2.IMWRITE_PNG_COMPRESSION,1]) 
            #cv2.imshow(window, img)
            #cv2.waitKey(0)
    except Exception as e:
        print(e)
        cv2.destroyAllWindows() 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno) 
         
# ----------------------------------------------------------------------------------------------------------------
  
def face_recognition_video(params):  
    window = "face_recognition_video"
    cv2.namedWindow(window)
    cv2.resizeWindow(window, 640, 480);
    people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Alexey Arestovich', 'Ilya Varlamov','Mark Feigin']# ,'Paula','Hose'
    haar_cascade = cv2.CascadeClassifier('/container_data/source/tools/haarcascades_4.5.5/haarcascade_frontalface_default.xml')
    # Загрузка обученной модели
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('/container_data/face/tools/face_trained.yml')
    
    pipeline_YUY2 = 'v4l2src device=/dev/video0 ! video/x-raw, width=(int)640, height=(int)480, format=YUY2 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1'
    cap = cv2.VideoCapture( pipeline_YUY2,cv2.CAP_GSTREAMER)
    if cap.isOpened(): 
        rval = True 
    else: 
        rval = False 
        
    while rval: 
        rval, frame = cap.read() 
        frame_flip = cv2.flip(frame,1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = cv2.bilateralFilter(frame, 9, 75, 75)
       
        faces_rect = haar_cascade.detectMultiScale(frame_gray, 1.1, 4)
        for (x,y,w,h) in faces_rect:
                faces_roi = frame_gray[y:y+h,x:x+w]
                label, confidence = face_recognizer.predict(faces_roi)
                print(f'Label = {people[label]} with a confidence of {confidence}')
                cv2.putText(frame_flip, str(people[label]), (x-w//2,y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), thickness=1)
                cv2.rectangle(frame_flip, (x,y), (x+w,y+h), (0,255,0), thickness=1)
       
        cv2.imshow(window, frame_flip)          
        key = cv2.waitKey(30) & 0xFF
        if key == 27: # exit on ESC break 
            cap.release()
            cv2.destroyWindow(window)
            return 0
# ----------------------------------------------------------------------------------------------------------------
 
        
def main(params):  
    if False:
        face_recognition(params)    
    if True:
        face_recognition_video(params)    
if __name__ == "__main__":
    main(sys.argv[1:])
    
    
# Run:    
# python  /container_data/face/face_recognition.py