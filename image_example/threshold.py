#!/usr/bin/python
# -*- coding: utf-8 -*-
from statistics import variance
import cv2
import numpy
import sys, os
import copy

from matplotlib import pyplot as plt # apt-get install -y python3-pip python3-tk&&  python -mpip install -U matplotlib && pip install pyqt5
from platform import python_version

print('OpenCV',cv2.__version__)  
print('Python',python_version())

def func_pass(v):
    pass 

'''
Пороги
Простые пороги на черно-белом изображении
Для усиления/ослабления цветовых областей

THRESH_BINARY - все что темнее определенной граници становится черным если меньше то белым
THRESH_BINARY_INV - как BINARY наоборот (к примеру исключить красный, выделим красный черным и он удалится)
THRESH_TRUNC - все что было до серого делает его белым 
THRESH_TOZERO - все что меньше определенного порога становится черным , а что больше порога останется как есть
THRESH_TOZERO_INV - как TOZERO наоборот
https://docs.opencv.org/4.5.5/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576a147222a96556ebc1d948b372bcd7ac59

cv2.threshold
https://docs.opencv.org/4.5.5/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57

https://russianblogs.com/article/44371526060/

'''
def threshold_simply():
    window = "threshold_simply"
    cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    thresh = 127 # значение порога
    maxval = 255 # значение для пикселя если сработал фильтр (для фильтра THRESH_BINARY при value_pixel > thresh  значение 0 иначе 255), максимальное значение для uint8
    cv2.createTrackbar("thresh black->white",window,0,255,func_pass)
    cv2.setTrackbarPos("thresh black->white",window,thresh)
    
    img = cv2.imread("/container_data/source/threshold/threshold.png",cv2.IMREAD_ANYCOLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     
    while True: 
        # Типы порогов ------------------------------------------------------------------------------------------- 
        cv2.imshow(window, img) 
        thresh = cv2.getTrackbarPos("thresh black->white",window) 
        # 255 - white, чем больше значение pixel тем он светлее
        # 0 - black, чем меньше значение pixel тем он темнее
        '''
        THRESH_BINARY - все пиксели светлее `thresh` т.е. меньше, станет белым (255 - белый), все пиксели темнее `thresh` т.е. больше станет черным (0 - черный)
        Если темнее `pixel < thresh`  тогда `pixel = 255 т.е. белый`
        Если светлее `pixel > thresh`  тогда `pixel = 0 т.е. черный`
        '''
        ret,frame_tresh_binary = cv2.threshold(img,thresh,maxval,cv2.THRESH_BINARY)
        #cv2.imshow("THRESH_BINARY", frame_tresh_binary) 
        
        '''
        THRESH_BINARY_INV - все что светлее `thresh` станет черным, все что темнее `thresh` станет белым
        
        '''
        ret,frame_tresh_binary_inv = cv2.threshold(img,thresh,maxval,cv2.THRESH_BINARY_INV)
        #cv2.imshow("THRESH_BINARY_INV", frame_tresh_binary_inv) 
        
        '''
        THRESH_TRUNC - все пиксели меньше `thresh` станут насыщенными т.е. черный усиливается,а пиксели больше чем `thresh` станут белыми
        Если темнее `pixel < thresh` то pixel более насышен
        Если светлее `pixel > thresh` то pixel=255 
        '''
        ret,frame_tresh_trunc = cv2.threshold(img,thresh,maxval,cv2.THRESH_TRUNC)
        #cv2.imshow("THRESH_TRUNC", frame_tresh_trunc) 
        
        '''
        THRESH_TOZERO - все пиксели больше `thresh` т.е. светлее останутся как есть, остальные станут черными
        Если светлее `pixel > thresh` то pixel=pixel 
        Если темнее `pixel < thresh` то pixel=0
        '''
        ret,frame_tresh_tozero = cv2.threshold(img,thresh,maxval,cv2.THRESH_TOZERO)
        cv2.imshow("THRESH_TOZERO", frame_tresh_tozero) 
        
        '''
        THRESH_TOZERO_INV - как THRESH_TOZERO наоборот
        '''
        ret,frame_tresh_tozero_inv = cv2.threshold(img,thresh,maxval,cv2.THRESH_TOZERO_INV)
        cv2.imshow("THRESH_TOZERO_INV", frame_tresh_tozero_inv)  
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # exit on ESC break 
            cv2.destroyAllWindows()
            break 
 
    #cv2.imwrite("/container_data/source/threshold/THRESH_BINARY.png", frame_tresh_binary)
    #cv2.imwrite("/container_data/source/threshold/THRESH_BINARY_INV.png", frame_tresh_binary_inv)
    #cv2.imwrite("/container_data/source/threshold/THRESH_TRUNC.png", frame_tresh_trunc)
    
    #cv2.imwrite("/container_data/source/threshold/ADAPTIVE_THRESH_MEAN_C.png", frame_tresh_adaptive_mean)
    #cv2.imwrite("/container_data/source/threshold/ADAPTIVE_THRESH_GAUSSIAN_C.png", frame_tresh_adaptive_gauss)
    #cv2.imwrite("/container_data/source/threshold/NO-ADAPTIVE-threshold.png", frame_tresh_for_compare)
    
    titles = ['Original Image','BINARY',           'BINARY_INV',          'TRUNC',           'TOZERO',           'TOZERO_INV']
    images = [img,             frame_tresh_binary, frame_tresh_binary_inv, frame_tresh_trunc, frame_tresh_tozero, frame_tresh_tozero_inv]
    
    for i in range(int(len(titles))):
        # subplot(nrows, ncols, index, **kwargs)
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
# ==========================================================================================================  
'''
Пороги Для усиления/ослабления цветовых областей
Когда мы не можем применить простые threshold пороги из-за неравномерности освещенности картинки,
когда нет однозначно темно и светло,а есть градиент тогда адаптивный порог сможет подобрать необходимый порог автоматически 

Адаптивный порог рисует контур вместо заливки как у простого порога threshold !

Адаптивный порог
    Когда изображение имеет разные условия освещения в разных областях
    алгоритм вычисляет порог для небольших областей изображения, поэтому мы даем разные пороговые значения 
    разным областям одного и того же изображения, что дает нам лучшие изображения при разном освещении в результате.

    cv2.ADAPTIVE_THRESH_MEAN_C: Порог - это среднее значение окрестности.
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C: Порог - это взвешенная сумма значений соседства, где вес - это окно Гаусса.

    cv2.adaptiveThreshold https://docs.opencv.org/4.5.5/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3

'''
def threshold_adaptive():
    window = "threshold_adaptive"
    cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
   
    maxval = 255 # значение для пикселя если сработал фильтр (для фильтра THRESH_BINARY при value_pixel > thresh  значение 0 иначе 255), максимальное значение для uint8
    thresh = 127 # значение порога
    C = 8 # Это просто константа, вычитаемая из рассчитанного среднего или средневзвешенного значения
    cv2.createTrackbar("thresh black->white",window,0,255,func_pass)
    cv2.setTrackbarPos("thresh black->white",window,thresh)
    cv2.createTrackbar("C",window,0,25,func_pass)
    cv2.setTrackbarPos("C",window,C)
    
    img = cv2.imread("/container_data/source/threshold/threshold_adaptive.png",cv2.IMREAD_ANYCOLOR)
    img  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    cv2.imshow(window, img) 
    
    # Адаптивный порог --------------------------------------------------------------------------------------------
    blockSize = 9 # Размер микрорайона, определяющий размер пороговой площади
   
    while True:
        thresh = cv2.getTrackbarPos("thresh black->white",window) 
        _ret,frame_tresh_for_compare = cv2.threshold(img,thresh,maxval,cv2.THRESH_BINARY)
        cv2.imshow("THRESH_BINARY", frame_tresh_for_compare)
        
        C = cv2.getTrackbarPos("C",window) 
        frame_tresh_adaptive_mean = cv2.adaptiveThreshold(img,maxval,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize,C)
        cv2.imshow("A_THRESH_MEAN_C", frame_tresh_adaptive_mean)
    
        frame_tresh_adaptive_gauss = cv2.adaptiveThreshold(img,maxval,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize,C)  
        cv2.imshow("A_THRESH_GAUSSIAN_C", frame_tresh_adaptive_gauss)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # exit on ESC break 
            cv2.destroyAllWindows()
            break 
 
    titles = ['Original Image', f"Global Thresholding (v = {thresh})",
            f"Adaptive Mean Thresholding C={C}", f"Adaptive Gaussian Thresholding C={C}"]
    images = [img, frame_tresh_for_compare, frame_tresh_adaptive_mean, frame_tresh_adaptive_gauss]
    
    for i in range(int(len(images))):
        # subplot(nrows, ncols, index, **kwargs)
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
  
# ==========================================================================================================  
'''
Порог Бинаризации Оцу
    Для бимодальных изображений - это изображение с двумя пиками на гистограмме
    (Для небимодальных изображений бинаризация не точна.)
    Автоматически рассчитает порог на основе гистограммы бимодального изображения
   
https://russianblogs.com/article/44371526060/
'''
def threshold_otsu():
    window = "threshold_otsu"
    cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    thresh = 106 # значение порога
     
    cv2.createTrackbar("thresh black->white",window,0,255,func_pass)
    cv2.setTrackbarPos("thresh black->white",window,thresh)
    img = cv2.imread("/container_data/source/threshold/threshold_OTSU.png",cv2.IMREAD_ANYCOLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    while True: 
        # Типы порогов ------------------------------------------------------------------------------------------- 
        cv2.imshow(window, img) 
        thresh = cv2.getTrackbarPos("thresh black->white",window) 
        
        # global пороговое значение
        ret1,frame_threshold = cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)
        cv2.imshow("T", frame_threshold) 
        print("threshold ret1=",ret1) # 106
        
        # Otsu's пороговое значение
        ret2,frame_thresh_otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow("T+OTSU", frame_thresh_otsu) 
        print("THRESH_OTSU ret1=",ret2) # 110
        
        # Пороговая обработка OTSU после фильтрации по Гауссу
        blur_gauss = cv2.GaussianBlur(img,(5,5),0)
        ret3,frame_thresh_otsu_gauss = cv2.threshold(blur_gauss,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow("T+OTSU+GaussianBlur", frame_thresh_otsu_gauss) 
        print("THRESH_OTSU_GaussianBlur ret3=",ret3)# 110
        
        blur_bilateral = cv2.bilateralFilter(img,d=9,sigmaColor=75,sigmaSpace=75)
        ret3,frame_thresh_otsu_bilateralFilter = cv2.threshold(blur_bilateral,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        frame_thresh_otsu_bilateralFilter = cv2.dilate(frame_thresh_otsu_bilateralFilter,kernel,iterations=3)
        frame_thresh_otsu_bilateralFilter = cv2.erode(frame_thresh_otsu_bilateralFilter,kernel,iterations=3)
        cv2.imshow("T+OTSU+bilateral+MORPH_CLOSE", frame_thresh_otsu_bilateralFilter) 
        
        print("\n\n")
        key = cv2.waitKey(100) & 0xFF
        if key == 27: # exit on ESC break 
            cv2.destroyAllWindows()
            break 
        
    # plot all the images and their histograms
    images = [img, 0, frame_threshold,
              img, 0, frame_thresh_otsu,
              blur_gauss, 0, frame_thresh_otsu_gauss,
              blur_bilateral,0,frame_thresh_otsu_bilateralFilter]
    titles = ['Original Noisy Image','Histogram',f"Global Thresholding (v={thresh})",
            'Original Noisy Image','Histogram',f"Otsu's Thresholding {ret1}",
            'Gaussian Image','Histogram',f"Otsu's Thresholding {ret2}",
            'bilateral MORPH_CLOSE Image','Histogram',f"Otsu's Thresholding{ret3}"]
    for i in range(int(len(images)/3)):
        # subplot(nrows, ncols, index, **kwargs)
        # картинка до
        plt.subplot(4,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        # гисторграмма
        plt.subplot(4,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        # картинка после
        plt.subplot(4,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
   
    plt.show()
    
    cv2.imwrite("/container_data/source/threshold/otsu_base_threshold.png", frame_threshold)
    cv2.imwrite("/container_data/source/threshold/otsu_THRESH_OTSU.png", frame_thresh_otsu)
    cv2.imwrite("/container_data/source/threshold/otsu_THRESH_OTSU_GaussianBlur.png", frame_thresh_otsu_gauss)    

# Пороги  -------------------------------------------------------------------------------------------------------  
if True:
    threshold_simply()
    
if False:
    threshold_adaptive()  
      
if False:    
    threshold_otsu()
    
# Run:    
# python  /container_data/image_example/threshold.py      