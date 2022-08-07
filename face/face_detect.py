#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2  
import sys, os

'''
face detect - просто обнаруживает наличие лица с помощью предварительно обученных моделей по методу классификации изображений

Два вида классификаторов:
- Haar classifier каскадный классификатор
- Local Binary Patterns Histograms (LBPH) локальная двоичная гистограмма (менее подвержены шуму на изображении)
'''

'''
Обнаружение лица на основе каскадного классификатора
каскадные классификаторы https://github.com/opencv/opencv/tree/4.x/data/haarcascades
'''
def face_detect(params):
    img = cv2.imread('/container_data/source/smile.jpeg')
    cv2.imshow('Group of 5 people', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray People', gray)

    haar_cascade = cv2.CascadeClassifier('/container_data/source/tools/haarcascades_4.5.5/haarcascade_frontalface_default.xml')
    # scaleFactor - фактор размера искомых обьектов относительно применяемой модели признаков
    # minNeighbors - как много соседних обьектов рядом

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
    print(f'Number of faces found = {len(faces_rect)}')

    for (x,y,w,h) in faces_rect:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    cv2.imshow('Detected Faces', img)
    cv2.waitKey(0)


def main(params):  
    if True:
        face_detect(params)    
        
if __name__ == "__main__":
    main(sys.argv[1:])
    

# Run:    
# python  /container_data/face/face_detect.py 