#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Обучение на маленьких данных не даст хороших результатов, для построения модели необходимо минимум 1000 картинок
'''
import os
import cv2  
import numpy  
import sys, os

features = []
labels = []

'''
Цель - заполнить features и labels

features - признаки, массивы изображений лиц
labels - метка, какому лицу принадлежит метка

Возможно заменить заполнение признаков вручную,вырезая нужную область но для ускорения можно положиться на работу детектора лиц
и доверить ему вырезания нужных областей(ROI) из картинки 
'''
def prepare_raw_data_for_train():
    global features,labels
    
    DIR = r'/container_data/source/face_train/face_train_data_set'  
    DIR_PRAPARE_RAW = r'/container_data/source/face_train/prepare_raw_face'
    haar_cascade = cv2.CascadeClassifier('/container_data/source/tools/haarcascades_4.5.5/haarcascade_frontalface_default.xml')
    people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Alexey Arestovich', 'Ilya Varlamov','Mark Feigin']# ,'Paula','Hose' название папки из DIR оно же и будет labels

    if os.path.exists(DIR_PRAPARE_RAW) == False:
        os.mkdir(DIR_PRAPARE_RAW)
                
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
      
        if os.path.exists(path) == False:
            continue
        
        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv2.imread(img_path)
            if img_array is None:
                continue 
                
            # Слишком большой размер уменьшить до приемлемых размеров    
            if img_array.shape[1] > 700:   
                final_wide = 700
                r = float(final_wide) / img_array.shape[1]
                dim = (final_wide, int(img_array.shape[0] * r)) 
                img_array = cv2.resize(img_array, dim, interpolation = cv2.INTER_CUBIC)
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

            # save raw face ----------------------------------------------------------------
            new_dir = os.path.join(DIR_PRAPARE_RAW,person)    
            if os.path.exists(new_dir) == False:
                os.mkdir(new_dir)
            # ------------------------------------------------------------------------------
            count = 0    
            for (x,y,w,h) in faces_rect:
                if len(faces_rect) == 1:
                    faces_roi = gray[y:y+h, x:x+w]
                    features.append(faces_roi)
                    labels.append(label) 
                    count+=1 
                    # Сохранить то что пойдет в обучение
                    cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,255,0),3, cv2.LINE_8) 
            if len(faces_rect) == 1:
                cv2.imwrite(os.path.join(new_dir,str(count)+'_'+img), img_array,[cv2.IMWRITE_PNG_COMPRESSION,1]) 
             
                
# ----------------------------------------------------------------------------------------------
 
def main(params): 
    global features,labels 
    if True:
        prepare_raw_data_for_train()
        print('Training done ---------------')
        if len(features) == 0:
            print('Empty data set')
            return
        features = numpy.array(features, dtype='object') # признаки, массивы изображений лиц
        labels = numpy.array(labels) # метка, какому лицу принадлежит метка

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Обучите Распознаватель по списку features и списку labels
        face_recognizer.train(features,labels)

        face_recognizer.save('/container_data/face/tools/face_trained.yml')
        os.remove("/container_data/face/tools/features.npy")
        os.remove("/container_data/face/tools/labels.npy")
        numpy.save('/container_data/face/tools/features.npy', features)
        numpy.save('/container_data/face/tools/labels.npy', labels)  
        
if __name__ == "__main__":
    main(sys.argv[1:])
    
    
# Run:    
# python  /container_data/face/face_recognition_opencv/faces_train.py