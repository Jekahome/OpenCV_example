#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy
import imutils
from imutils import contours
from PIL import Image
import sys, os
from matplotlib import pyplot as plt 
from platform import python_version
import easyocr
 

print('OpenCV',cv2.__version__)  
print('Python',python_version())

  
'''
    OpenCV – библиотека по работе с фото и видео;
    MatplotLib – библиотека для более комфортного и информативного вывода фото;
    EaseOCR – библиотека для чтения текста с фото;
    Numpy – библиотека для работы с числами и массивами данных;
    ImUtils – библиотека, что предоставляет функции по работе с фото. В частности удобный способ получения всех контуров.
'''
def use_easyocr(params):
    window = "use_easyocr"
    cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) # Create window handler

    # Распознавание номера у машины из картинки -------------------------------------------------------
    img = cv2.imread("/container_data/source/car_numbers/car2.jpeg")
    if img.size == 0:
        print('Image not found')
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_filter = cv2.bilateralFilter(gray, 11, 15, 15)
    edges = cv2.Canny(img_filter,150,200, apertureSize = 3)
    cv2.imshow(window, edges)
    cnts,h = cv2.findContours(edges.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
    print('Count contours:',len(cnts))
    
    #img_contours = numpy.uint8(numpy.zeros((img.shape[0],img.shape[1])))
    img_contours = numpy.zeros(gray.shape,numpy.uint8)
    car_numbers = dict()
    count = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt,oriented=False)# Вычисляет площадь контура. Функция наверняка даст неверный результат для контуров с самопересечениями.
        arclen = cv2.arcLength(cnt,True) # возвращает длину дуги контура
        eps = 0.03 # точность аппроксимации
        epsilon = eps*arclen
        is_closed_circuit = True # замкнутый контур
        num_top = cv2.approxPolyDP(cnt,epsilon,is_closed_circuit)# Аппроксимирование (упрощение) области контура
        if len(num_top) == 4 and area < 10000 and area > 1300:           
            print(f"Area={area} cnts={len(cnt)}")
            cv2.drawContours(img_contours ,cnt,-1,255,thickness=cv2.FILLED) # только для контура
            # нарисовать аппроксимированный контур (меньше деталей, видно вершины)
            cv2.drawContours(img_contours,[num_top],0,255,thickness=cv2.FILLED)
            
            x_c,y_c,w_c,h_c = cv2.boundingRect(num_top)# получить квадрат области
            img_crop = img[y_c:y_c+h_c,x_c:x_c+w_c,...] 
            text = easyocr.Reader(['en'])
            text = text.readtext(img_crop)
           
            count+=1
            car_numbers[count] = dict()
            car_numbers[count]["img"] = img_crop
            car_numbers[count]["text"] = text[0][-2]
     
    for key, value in car_numbers.items():
        plt.subplot(3,3,key),plt.imshow(value["img"],'gray')
        plt.title(value["text"])
        plt.xticks([]),plt.yticks([])
    plt.savefig('/container_data/source/car_numbers/all_search_numbers.png')     
    plt.show()   
    
    # показать все номера на исходном изображении
    bitwise_img = cv2.bitwise_and(img,img,mask=img_contours)
    plt.imshow(bitwise_img)
    plt.savefig('/container_data/source/car_numbers/car_numbers_mask.png') 
    plt.show()          
    
        
    print('Enter ESC')
    key = cv2.waitKey(0) & 0xFF
    if key == 27: # exit on ESC break 
        cv2.destroyAllWindows()



def main(params):
    if False:
        use_easyocr(params)
  
    
if __name__ == "__main__":
    main(sys.argv[1:])
    

# Run:    
# python  /container_data/car_number.py
 