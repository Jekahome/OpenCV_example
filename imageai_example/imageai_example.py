# https://github.com/OlafenwaMoses/ImageAI

# https://www.youtube.com/watch?v=f7IZBmjbpqU&list=PLOjc9X-vV0SEHQDXm30Ts-3GGqn_PKaNu&index=5&ab_channel=BorisBochkarev%7CBeTry

'''
    Распознавание контуров (плохо справляется)

'''
'''
pip install tensorflow==2.4.0
pip install keras==2.4.3 numpy==1.19.3 pillow==7.0.0 scipy==1.4.1 h5py==2.10.0 matplotlib==3.3.2 opencv-python keras-resnet==0.2.0
'''
import numpy
import cv2
import imutils

# $ /home/jeka/.local/bin/python imageai_example.py 

image = cv2.imread("/container_data/source/canny/books.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
gray = cv2.GaussianBlur(gray,(5,5),0)# заблурим
cv2.imwrite("/container_data/source/canny/books_2.jpg",gray)
edges = cv2.Canny(gray,10,200)
cv2.imwrite("/container_data/source/canny/books_3.jpg",edges)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,50))# размер искомых обьектов
closed = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel)
cv2.imwrite("/container_data/source/canny/books_4.jpg",closed)
cnts = cv2.findContours(closed.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # https://github.com/opencv/opencv/blob/17234f82d025e3bbfbf611089637e5aa2038e7b8/doc/js_tutorials/js_imgproc/js_contours/js_contours_hierarchy/js_contours_hierarchy.markdown
cnts = imutils.grab_contours(cnts) # подсчитывает количество контуров

total_books = 0
for c in cnts:
    p = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.02*200,True) # сгладить контур
    if len(approx) >= 3:
        cv2.drawContours(image,[approx],-1,(0,255,0),4)
        total_books += 1
        
print(total_books)
cv2.imwrite("/container_data/source/canny/books_5.jpg",image)        