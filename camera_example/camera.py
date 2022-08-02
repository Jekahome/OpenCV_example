#!/usr/bin/python
# -*- coding: utf-8 -*-
from binascii import a2b_hex
import sys, os
import cv2 
import numpy
from platform import python_version
import colorsys
import urllib.request

'''
    захват видео
    
   cv.read() https://docs.opencv.org/4.5.5/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1
'''
# dependency: python -m pip install opencv-python 
print('OpenCV',cv2.__version__)  
print('Python',python_version())  

def func_pass(v):
    pass 

# ==========================================================================================================  
def test_url_video():
    url = "http://192.168.1.132:8090/" 
    title = "Title window"
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=numpy.array(bytearray(img_resp.read()),dtype=numpy.uint8)
        frame=cv2.imdecode(imgnp,-1)
        cv2.imshow(title, frame) 
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # exit on ESC break 
            cv2.destroyWindow(title)
            return 0

# ==========================================================================================================  
'''
Простые пороги на черно-белом изображении
THRESH_BINARY - все что темнее определенной граници становится черным если меньше то белым
THRESH_BINARY_INV - как BINARY наоборот (к примеру исключить красный, выделим красный черным и он удалится)
THRESH_TRUNC - все что было до серого делает его белым 
THRESH_TOZERO - все что меньше определенного порога становится черным , а что больше порога останется как есть
THRESH_TOZERO_INV - как TOZERO наоборот
https://docs.opencv.org/4.5.5/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576a147222a96556ebc1d948b372bcd7ac59

cv2.threshold
https://docs.opencv.org/4.5.5/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57


1.Найти порог https://russianblogs.com/article/44371526060/
2.применить порог
'''
def threshold_example():
    try:
        window = "threshold_example"
        delay_skip = 1
        cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
        vc = cv2.VideoCapture(0 ,cv2.CAP_VFW) # 0 - запущенная камера
        
        if vc.isOpened(): # try to get the first frame 
            rval = True 
        else: 
            rval = False 
        
        thresh = 73 # значение порога
        maxval = 255 # максимальное значение для uint8
        C = 3 # Это просто константа, вычитаемая из рассчитанного среднего или средневзвешенного значения
        blockSize = 7 # Размер микрорайона, определяющий размер пороговой площади
      
        cv2.createTrackbar("thresh",window,0,255,func_pass)
        cv2.setTrackbarPos("thresh",window,thresh)
        cv2.createTrackbar("C",window,0,25,func_pass)
        cv2.setTrackbarPos("C",window,C)
       
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        while rval: 
            rval, frame = vc.read() 
            frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cv2.imshow(window, frame_gray) 
            thresh = cv2.getTrackbarPos("thresh",window) 
            
            '''
            THRESH_BINARY - все что светлее `thresh` т.е. меньше станет белым (255 - белый), все что темнее `thresh` т.е. больше станет черным (0 - черный)
            `thresh = 127` если `value = 126` получается `value < thresh`  тогда `value = 255 т.е. белый`
                           если `value = 128` получается `value > thresh`  тогда `value = 0 т.е. черный`
            '''
            #ret,frame_tresh_1 = cv2.threshold(frame_gray,thresh,maxval,cv2.THRESH_BINARY)
            #cv2.imshow("THRESH_BINARY", frame_tresh_1) 
            '''
            THRESH_BINARY_INV - все что светлее `thresh` станет черным, все что темнее `thresh` станет белым
            
            '''
            #ret,frame_tresh_2 = cv2.threshold(frame_gray,thresh,maxval,cv2.THRESH_BINARY_INV)
            #cv2.imshow("THRESH_BINARY_INV", frame_tresh_2) 
            
            ret,frame_tresh_3 = cv2.threshold(frame_gray,thresh,maxval,cv2.THRESH_TRUNC)
            cv2.imshow("THRESH_TRUNC", frame_tresh_3) 
            
            ret,frame_tresh_4 = cv2.threshold(frame_gray,thresh,maxval,cv2.THRESH_TOZERO)
            cv2.imshow("THRESH_TOZERO", frame_tresh_4) 
            
            ret,frame_tresh_5 = cv2.threshold(frame_gray,thresh,maxval,cv2.THRESH_TOZERO_INV)
            cv2.imshow("THRESH_TOZERO_INV", frame_tresh_5) 
            
            # Адаптивный порог --------------------------------------------------------------------------------------------
            C = cv2.getTrackbarPos("C",window) 
            
            frame_tresh_adaptive_mean = cv2.adaptiveThreshold(frame_gray,maxval,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize,C)
            cv2.imshow("ADAPTIVE_THRESH_MEAN_C", frame_tresh_adaptive_mean)
        
            frame_tresh_adaptive_gauss = cv2.adaptiveThreshold(frame_gray,maxval,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize,C)
            cv2.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", frame_tresh_adaptive_gauss)
           
            # + морфология (усилить черный контур)
            frame_tresh_adaptive_gauss = cv2.erode(frame_tresh_adaptive_gauss,kernel,iterations=2)
            frame_tresh_adaptive_gauss = cv2.dilate(frame_tresh_adaptive_gauss,kernel,iterations=1)
            cv2.imshow("ADAPTIVE_THRESH_GAUSSIAN_C + cv2.MORPH_OPEN", frame_tresh_adaptive_gauss)
        
            # OTSU --------------------------------------------------------------------------------------------------------
            blur_gauss = cv2.GaussianBlur(frame_gray,(3,3),0)
            ret3,frame_thresh_otsu_gauss = cv2.threshold(blur_gauss,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imshow("OTSU", frame_thresh_otsu_gauss)
        
            key = cv2.waitKey(delay_skip) & 0xFF
            if key == 27: # exit on ESC break 
                vc.release()
                cv2.destroyAllWindows()
                return 0 
    except Exception as e: 
        print(e)
        cv2.destroyAllWindows()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
# ==========================================================================================================  
'''
Построение маски inRange
'''
def in_range_example():
    window = "in_range_example"
    cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
    # верхний порог цвета
    cv2.createTrackbar("min",window,0,255,func_pass)
    cv2.createTrackbar("max",window,0,255,func_pass)
    cv2.setTrackbarPos("min",window,14)
    cv2.setTrackbarPos("max",window,90)
    
    thresh = 95 # значение порога
    maxval = 255 # значение для пикселя если сработал фильтр (для фильтра THRESH_BINARY при value_pixel > thresh  значение 0 иначе 255), максимальное значение для uint8
    cv2.createTrackbar("thresh black->white",window,0,255,func_pass)
    cv2.setTrackbarPos("thresh black->white",window,thresh)
    
    vc = cv2.VideoCapture(0 ,cv2.CAP_VFW)
    if vc.isOpened():  
        rval = True 
    else: 
        rval = False 

    while rval: 
        rval, frame = vc.read() 
        
        # get trackbar
        max = cv2.getTrackbarPos("max",window)
        min = cv2.getTrackbarPos("min",window)
    
        frame = cv2.bilateralFilter(frame,d=9,sigmaColor=75,sigmaSpace=75)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
          
        thresh = cv2.getTrackbarPos("thresh black->white",window)  
        ret,frame_tr = cv2.threshold(frame,thresh,maxval,cv2.THRESH_TOZERO_INV)
        cv2.imshow("THRESH_TOZERO_INV",frame_tr)
          
        ret,frame_tr_z = cv2.threshold(frame,thresh,maxval,cv2.THRESH_TOZERO)
        cv2.imshow("THRESH_TOZERO",frame_tr_z)
          
        # Маска для цветного кадра (HSV,BGR,...)
        #high = numpy.array([max,255,255])
        #low = numpy.array([min,0,0])           
        #frame_mask = cv2.inRange(frame,low,high,cv2.CV_8U)
        
        # Маска для черно-белого кадра COLOR_BGR2GRAY
        frame_mask = cv2.inRange(frame_tr,numpy.array([min]),numpy.array([max]),cv2.CV_8U)
        cv2.imshow(window,frame_mask)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # exit on ESC break 
            vc.release()
            cv2.destroyAllWindows()
            return 0 

# ==========================================================================================================  

'''
Выделение обьекта по форме
Распознавание и отслеживание объектов по форме. 

Canny находит края на основе контраста

Лучшее распознавание обьектов по форме когда обьекты контрасные
'''
def selecting_object_by_shape():
    try:
        window = "5 selecting_object_by_shape"
        cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
        cv2.moveWindow(window, 1000, 0)
        cv2.resizeWindow(window, 400, 400);
 
        window_tresh_adaptive = "1 Tresh adaptive"
        cv2.namedWindow(window_tresh_adaptive,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
        cv2.moveWindow(window_tresh_adaptive, 0, 0)
        cv2.resizeWindow(window_tresh_adaptive, 400, 400);
        
        window_tresh = "2 Tresh"
        cv2.namedWindow(window_tresh,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
        cv2.moveWindow(window_tresh, 500, 0)
        cv2.resizeWindow(window_tresh, 400, 400);
        
        window_morph =  "3 window_morph"
        cv2.namedWindow(window_morph,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.moveWindow(window_morph, 0, 500)
        cv2.resizeWindow(window_morph, 400, 400);
         
        iterations = 3 # 1 сильно выделит край
        cv2.createTrackbar("iterations",window_morph,0,10,func_pass)
        cv2.setTrackbarPos("iterations",window_morph,iterations)
        
        window_edge = "4 Edge"
        cv2.namedWindow(window_edge,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
        cv2.moveWindow(window_edge, 500, 500)
        cv2.resizeWindow(window_edge, 400, 400);        
        
        C = 4 # 1 сильно выделит край
        cv2.createTrackbar("C",window_tresh_adaptive,0,10,func_pass)
        cv2.setTrackbarPos("C",window_tresh_adaptive,C)
        
        thresh = 255
        cv2.createTrackbar("thresh black->white",window_tresh,0,255,func_pass)
        cv2.setTrackbarPos("thresh black->white",window_tresh,thresh)
        
        # верхний порог цвета
        cv2.createTrackbar("min",window_edge,0,255,func_pass)
        cv2.createTrackbar("max",window_edge,0,255,func_pass)
        thresh_min = 10  
        thresh_max = 40
        cv2.setTrackbarPos("min",window_edge,thresh_min)
        cv2.setTrackbarPos("max",window_edge,thresh_max)
        
        epsilon = 100 # 0.0-1.0 Параметр, определяющий точность аппроксимации. Это максимальное расстояние между исходной кривой и ее аппроксимацией.
        cv2.createTrackbar("epsilon",window,1,100,func_pass)
        cv2.setTrackbarPos("epsilon",window,epsilon)
        
        #vc = cv2.VideoCapture(0 ,cv2.CAP_VFW)
        #vc = cv2.VideoCapture("/container_data/source/video/figura.MOV")
        vc = cv2.VideoCapture("/container_data/source/video/figura_2.MOV")
        if vc.isOpened():  
            rval = True 
        else: 
            rval = False 
    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))# MORPH_RECT MORPH_CROSS MORPH_ELLIPSE
        while rval: 
            rval, frame = vc.read() 
            if rval == False:
                vc = cv2.VideoCapture("/container_data/source/video/figura_2.MOV")
                rval, frame = vc.read()
                #cv2.destroyAllWindows()
                #break
                
            # Сгладить края очень мягко, не потеряв в контрастности иначе Canny не сможет построить края !!!
            #frame = cv2.bilateralFilter(frame,d=3,sigmaColor=75,sigmaSpace=75)
            
            frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
             
            # Пороги адаптивный
            maxval = 255 # значение для пикселя если сработал фильтр (для фильтра THRESH_BINARY при value_pixel > thresh  значение 0 иначе 255), максимальное значение для uint8
            C = cv2.getTrackbarPos("C",window_tresh_adaptive) # Это просто константа, вычитаемая из рассчитанного среднего или средневзвешенного значения
            blockSize = 23 # Размер микрорайона, определяющий размер пороговой площади
            img_tresh_adaptive = cv2.adaptiveThreshold(frame_gray,maxval,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize,C)  
            cv2.imshow(window_tresh_adaptive, img_tresh_adaptive)
             
            # Пороги Усилить черный 
            thresh = cv2.getTrackbarPos("thresh black->white",window_tresh) 
            ret,img_tresh = cv2.threshold(img_tresh_adaptive,thresh,maxval,cv2.THRESH_TRUNC)
            cv2.imshow(window_tresh, img_tresh)
             
            # Морфология
            iterations = cv2.getTrackbarPos("iterations",window_morph)
            img_morph = cv2.erode(img_tresh,kernel,iterations=iterations)# дорисует дыры в контуре  
            cv2.imshow(window_morph, img_morph)
   
            # Края 
            thresh_min = cv2.getTrackbarPos("min",window_edge)  
            thresh_max = cv2.getTrackbarPos("max",window_edge) 
            frame_edge = cv2.Canny(img_morph,thresh_min,thresh_max, apertureSize = 3)
            cv2.imshow(window_edge, frame_edge)
            
            
            contours,h = cv2.findContours(frame_edge, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours,key=cv2.contourArea,reverse=True)
            epsilon = cv2.getTrackbarPos("epsilon",window) 
            for cnts_roi in contours:
                area = cv2.contourArea(cnts_roi,oriented=False)
                if area > 1000:
                    # нарисовать точный контур
                    cv2.drawContours(frame,cnts_roi,-1,(128,0,128),3,cv2.LINE_AA)
                    p = cv2.arcLength(cnts_roi,True)
                    # Аппроксимирование Дорисовка контура
                    # https://docs.opencv.org/4.5.5/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
                    num_top = cv2.approxPolyDP(cnts_roi,(epsilon/100)*p,True)# Аппроксимирование (упрощение) области контура
                    x_c,y_c,w_c,h_c = cv2.boundingRect(num_top)
                    cv2.putText(frame,f"Area={area}",(x_c+10,y_c+(h_c//2)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),1)
                    cv2.putText(frame,f"Top={len(num_top)}",(x_c+10,y_c+(h_c//2)+20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),1)
                    # обвести фигуру прямоугольником
                    # cv2.rectangle(frame,(x_c,y_c),(x_c+w_c,y_c+h_c),(0,0,255),4)
                    # нарисовать аппроксимированный контур (меньше деталей, видно вершины)
                    cv2.drawContours(frame,[num_top],0,(128,0,128),3,cv2.LINE_AA)
            cv2.imshow(window, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # exit on ESC break 
                vc.release()
                cv2.destroyAllWindows()
                return 0 
            
    except Exception as e: 
        print(e)
        cv2.destroyAllWindows()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
# ==========================================================================================================  

'''
Выделение обьекта по цвету

1.Размытие с сохранением краев bilateralFilter
2.Преобразование в HSV cv2.cvtColor
3.Построить маску вхождения диапазона цвета cv2.inRange
4.Почистить шум морфологие
5.Выделить пиксели на оригинале по маске cv2.bitwise_and


Цветовое пространсво cv2.COLOR_BGR2HSV
https://russianblogs.com/article/41781040604/

cv2.CAP_VFW - https://docs.opencv.org/4.5.5/d4/d15/group__videoio__flags__base.html
cv2.CAP_GSTREAMER - https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c#install-gstreamer-on-ubuntu-or-debian
'''

'''
HSV - Hue, Saturation, Value — тон, насыщенность, яркость
H -  цветовой тон, (например, красный, зелёный или сине-голубой). Варьируется в пределах 0—360°, однако иногда приводится к диапазону 0—100 или 0—1
S - насыщенность. Варьируется в пределах 0—100 или 0—1. Чем больше этот параметр, тем «чище» цвет, поэтому этот параметр иногда называют чистотой цвета
V - яркость. Также задаётся в пределах 0—100 или 0—1
'''
def selecting_object_by_color():
    try:
        window = "hsv"
        cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
        # верхний порог цвета
        cv2.createTrackbar("H max",window,0,360,func_pass)
        cv2.createTrackbar("S max",window,0,255,func_pass)
        cv2.createTrackbar("V max",window,0,255,func_pass)    
        # нижний порог цвета
        cv2.createTrackbar("H min",window,0,360,func_pass)
        cv2.createTrackbar("S min",window,0,255,func_pass)    
        cv2.createTrackbar("V min",window,0,255,func_pass)

        # blue settings
        cv2.setTrackbarPos("H max",window,175)
        cv2.setTrackbarPos("S max",window,200)
        cv2.setTrackbarPos("V max",window,120)
        cv2.setTrackbarPos("H min",window,150)
        cv2.setTrackbarPos("S min",window,98)
        cv2.setTrackbarPos("V min",window,22)
        
        # green settings
        #cv2.setTrackbarPos("H max",window,90)
        #cv2.setTrackbarPos("S max",window,255)
        #cv2.setTrackbarPos("V max",window,255)
        #cv2.setTrackbarPos("H min",window,42)
        #cv2.setTrackbarPos("S min",window,74)
        #cv2.setTrackbarPos("V min",window,42)
        
        delay_skip = 1
        #gst_str = ("gst-launch-1.0 v4l2src device=\"/dev/video0\" ! decodebin ! videoconvert ! video/x-raw, width=640, height=480 ! appsink")
        #vc = cv2.VideoCapture(gst_str ,cv2.CAP_GSTREAMER)  
        vc = cv2.VideoCapture(0 ,cv2.CAP_VFW)
        if vc.isOpened():  
            rval = True 
        else: 
            rval = False 
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        while rval: 
            rval, frame_1 = vc.read() 
            
            # Сглаживание --------------------------------------------------------
            '''
                # различные формы сглаживания
                frame_gauss = cv2.GaussianBlur(frame, (15, 15), 0)
                
                frame_gauss = cv2.medianBlur(frame, 7)
            
                frame_gauss = cv2.bilateralFilter(frame, 15 ,75, 75)
                
                kernal = numpy.ones((9, 9), numpy.float32)/255
                frame_gauss = cv2.filter2D(frame, -1, kernal_blur)
            '''        
            #frame = cv2.GaussianBlur(frame, (3, 3), 0)
            frame_1 = cv2.bilateralFilter(frame_1,d=9,sigmaColor=75,sigmaSpace=75)
            
            # Преобразование в HSV ---------------------------------------------- 
            frame_2_hsv = cv2.cvtColor(frame_1,cv2.COLOR_BGR2HSV)
        
            # Построить маску, frame вхождения диапазона -------------------------
            h = cv2.getTrackbarPos("H max",window)
            s = cv2.getTrackbarPos("S max",window)
            v = cv2.getTrackbarPos("V max",window)
            hl = cv2.getTrackbarPos("H min",window)
            sl = cv2.getTrackbarPos("S min",window)
            vl = cv2.getTrackbarPos("V min",window)    
                
            colorHigh = numpy.array([h,s,v])
            colorLow = numpy.array([hl,sl,vl]) 
            frame_3_mask = cv2.inRange(frame_2_hsv,colorLow,colorHigh,cv2.CV_8U)  
            # показать не обработанную mask   
            if True:
              cv2.imshow("1 Mask", frame_3_mask)         

            # --------------------------------------------------------------------
            # Морфология
            frame_4_mask_morph = cv2.dilate(frame_3_mask,kernel,iterations=2)
            frame_4_mask_morph = cv2.erode(frame_4_mask_morph,kernel,iterations=2)
            #frame_4_mask_morph = cv2.morphologyEx(frame_3_mask, cv2.MORPH_CLOSE, kernel)
             
            # Show morphological transformation mask
            if True:
              cv2.imshow('2 Morphological', frame_4_mask_morph)
            # -------------------------------------------------------------------   
            # наложение mask поверх исходного изображения
            # взять пиксель из frame_4_mask_morph значением 255 т.е выделенный обьект и по его координатам взять из frame остальная область т.е 0 будет черным фоном
            frame_result = cv2.bitwise_and(frame_1, frame_1, mask = frame_4_mask_morph)
            if True:
                cv2.imshow('Result bitwise_and', frame_result)# Show final output image
            # --------------------------------------------------------------------
            # Контур искомого цвета
            # попробовать cv2.RETR_EXTERNAL вместо cv2.RETR_TREE !!!!!!!!!!!
            contours, _ = cv2.findContours(frame_4_mask_morph  ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours,key=cv2.contourArea,reverse=True)# сортируем по площади
            line_color = (0,0,255)
             
            for cnts_roi in contours:
                area = cv2.contourArea(cnts_roi)
                if area > 2000:
                    # варианты отобразить контур
                    if True:
                        # Через approxPolyDP упрощенный контур
                        # рисует прямойгольник вокруг найденной области
                        p = cv2.arcLength(cnts_roi,True)
                        num_top = cv2.approxPolyDP(cnts_roi,0.03*p,True)
                        x_c,y_c,w_c,h_c = cv2.boundingRect(num_top)# получить квадрат области
                        frame_1 = cv2.rectangle(frame_1,(x_c,y_c),(x_c+w_c,y_c+h_c),line_color,2)
                        frame_1 = cv2.rectangle(frame_1,(x_c,y_c),(x_c+60,y_c-25),(0,0,0),-1)
                        cv2.putText(frame_1,"Red",(x_c, y_c),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),1)
                        # Отслеживание обьекта
                        # x __ left=min right=max
                        # y |  top=min down=max
                        frame_1 = cv2.circle(frame_1,(x_c+(w_c//2),y_c+(h_c//2)),20,line_color,-1 )
                        print(F"X:{x_c+(w_c//2)} Y:{y_c+(h_c//2)}")
                    else:
                        # На основе всех имеющихся данных от findContours
                        cv2.drawContours(frame_1,[cnts_roi],-1,line_color,1)# обводит контуром найденную область
                        M = cv2.moments(cnts_roi)
                        cx = int(M["m10"]/M["m00"])
                        cy = int(M["m01"]/M["m00"])
                        cv2.circle(frame_1,(cx,cy),7,line_color,-1)
                        cv2.putText(frame_1,"Red",(cx-20, cy-20),cv2.FONT_HERSHEY_SIMPLEX, 1, line_color,1)
                else:break
                
            # --------------------------------------------------------------------
            # Поворот видео потока
            if False:
                # ROTATE_90_CLOCKWISE,ROTATE_180,ROTATE_90_COUNTERCLOCKWISE
                frame_rotate = cv2.rotate(frame,cv2.ROTATE_180)
                cv2.imshow("rorate", frame_rotate)  
            # --------------------------------------------------------------------
            # Изменение размера видео потока
            if False:
                final_wide = frame_1.shape[1]-400 # уменьшить 
                dim = (final_wide, int(frame_1.shape[0] *  (float(final_wide) / frame_1.shape[1])))
                frame_resized = cv2.resize(frame_1, dim, interpolation = cv2.INTER_CUBIC)
                cv2.imshow('resize INTER_CUBIC',frame_resized)
            # --------------------------------------------------------------------
            # показать оригинал
            cv2.imshow(window, frame_1)  
            
            
            key = cv2.waitKey(delay_skip) & 0xFF
            if key == 27: # exit on ESC break 
                vc.release()
                cv2.destroyAllWindows()
                return 0
    except Exception as e: 
        print(e)
        cv2.destroyAllWindows()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
# ==========================================================================================================  
  
#mouse callback function
def run_stop_toggle(event,x,y,flags,param):
    global is_l_button_down_activ
    # Mouse Events
    # https://docs.opencv.org/4.5.5/d0/d90/group__highgui__window__flags.html#gga927593befdddc7e7013602bca9b079b0ad3b2124722127f85f6b335aee8ae5fcc
    if event==cv2.EVENT_LBUTTONDOWN:
         if is_l_button_down_activ == True:
             is_l_button_down_activ = False
         else:is_l_button_down_activ = True       
                
'''
Градиенты, края и контур
Градиент - это направленное изменение интенсивности цвета в изображении. 
Определение краев по технологии Canny edge основано на градиентах

Canny находит края на основе контраста
cv2.Canny - находит края на изображении с помощью алгоритма Канни.
https://docs.opencv.org/4.5.5/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de

cv2.findContours - находит контуры
https://docs.opencv.org/4.5.5/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0

cv2.drawContours - рисует контур
https://docs.opencv.org/4.5.5/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
'''        
def gradients_and_edges_and_contours():
    try:  
        global is_l_button_down_activ
        window = "edge canny"
        window_contours = "contours"
        window_threshold = "edge threshold"
        
        cv2.namedWindow(window,flags=cv2.WINDOW_AUTOSIZE|cv2.WINDOW_KEEPRATIO|cv2.WINDOW_GUI_EXPANDED ) 
        #cv2.resizeWindow(window, 720, 520);
        cv2.moveWindow(window, 800, 0)
        cv2.setWindowProperty(window, cv2.WND_PROP_ASPECT_RATIO,cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty(window, cv2.WND_PROP_OPENGL,cv2.WINDOW_OPENGL)
    
        cv2.namedWindow(window_contours,flags=cv2.WINDOW_AUTOSIZE|cv2.WINDOW_KEEPRATIO|cv2.WINDOW_GUI_EXPANDED) 
        cv2.setWindowProperty(window_contours, cv2.WND_PROP_OPENGL,cv2.WINDOW_OPENGL)
        #cv2.resizeWindow(window_contours, 520, 320); 
        cv2.moveWindow(window_contours, 0, 0)

        cv2.namedWindow(window_threshold,flags=cv2.WINDOW_AUTOSIZE|cv2.WINDOW_KEEPRATIO|cv2.WINDOW_GUI_EXPANDED)
        #cv2.resizeWindow(window_threshold, 520, 320); 
        cv2.moveWindow(window_threshold, 0, 600)
    
        cv2.createTrackbar("threshold min",window,0,500,func_pass)
        cv2.createTrackbar("threshold max",window,0,500,func_pass)
        cv2.setTrackbarPos("threshold min",window,100)
        cv2.setTrackbarPos("threshold max",window,200)
        cv2.setMouseCallback(window_contours,run_stop_toggle)
                
        vc = cv2.VideoCapture(0 ,cv2.CAP_VFW)
        #vc = cv2.VideoCapture('/container_data/source/video/murom_640x360.mp4')
        if vc.isOpened():  
            rval = True 
        else: 
            rval = False 

        is_init_frame_copy = False
        frame_copy = None
        rval, frame = vc.read()
        while rval: 
            if is_l_button_down_activ == False:
                rval, frame = vc.read()
                # предварительная подготовка
                frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # черное-белый фрейм
                frame_gray = cv2.bilateralFilter(frame_gray,d=9,sigmaColor=75,sigmaSpace=75) # сглаживание с сохранением краев
               
                
                is_init_frame_copy = False
            else:
                if is_init_frame_copy == False: 
                    frame_copy = frame_gray.copy()
                    is_init_frame_copy = True
                frame_gray = frame_copy.copy()
                    
                if False:
                    '''
                        Как выглядят градиенты
                        cv2.CV_64F - формат данных
                        1,0 - вычисление по X , 0, - вычисление по Y 
                        ksize=5 радиус вычислений пикселей
                    '''
                    framme_sobel_x = cv2.Sobel(frame_gray,cv2.CV_64F,1,0,ksize=5)
                    framme_sobel_y = cv2.Sobel(frame_gray,cv2.CV_64F,0,1,ksize=5)
                    cv2.imshow("Sobel X", framme_sobel_x)  
                    cv2.imshow("Sobel Y", framme_sobel_y)  
            # края      
            threshold_min = cv2.getTrackbarPos("threshold min",window) # первый порог для процедуры гистерезиса.
            threshold_max = cv2.getTrackbarPos("threshold max",window) # второй порог для процедуры гистерезиса.
            
            # По правильному края надо строить на основе морфологии
            # а морфологию строить по биполярной маске как в `smoothing_and_morphological_transformations`
            
            # края Canny ------------------------------------------------------------------------------------------------------
            frame_edge = cv2.Canny(frame_gray,threshold_min,threshold_max)
            cv2.imshow(window, frame_edge)  
        
            # края threshold --------------------------------------------------------------------------------------------------
            if True:
                retval, frame_edge_threshold = cv2.threshold(frame_gray,threshold_min,threshold_max,cv2.THRESH_BINARY)
                cv2.imshow(window_threshold,frame_edge_threshold)
            
            # контур
            '''
                hierarchy - топология - это иерархия изображения, взаимосвязь пикселей
                cv2.RETR_LIST - не фильтрует контуры
                cv2.CHAIN_APPROX_NONE - не урезать контуры
            '''
            contours,_hierarchy = cv2.findContours(frame_edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            # сортировка контуров по убыванию, т.е. самый большой первый
            contours = sorted(contours,key=cv2.contourArea,reverse=True)
            if len(contours) > 0:
                index_contours = -1 # -1 - все контуры
                color_contours = (0,0,255)
                thickness = 3 # толщина контура, -1 - залить ???
                # hierarchy ???
                # maxLevel ???
                #cv2.drawContours(frame_gray,[contours[0]],index_contours,color_contours,thickness,cv2.LINE_AA) # -1 - все контуры
                cv2.drawContours(frame_gray,contours,index_contours,color_contours,thickness,cv2.LINE_AA) # -1 - все контуры
                
                cv2.imshow(window_contours, frame_gray)  
        
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # exit on ESC break 
                vc.release()
                cv2.destroyAllWindows()
                return 0
    except Exception as e: 
        print(e)
        cv2.destroyAllWindows()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
# ==========================================================================================================  
'''
Сглаживание и морфологические трансформации
    Сглаживание как подготовка кадра для распознания форм и цветов.Устранение шумов и помех.
    
    Морфология для изменения формы изображения (Улучшает распознавание по цвету и контуру)
        Erosion - Уменьшает белый шум.для gray уменьшает границы белого и увеличивает границы черного 
                  убирает белые вкрапления на черном (если все значения ядра совпали)
                  т.е. уменьшает белый усиливает черный
        Dilation - Уменьшает черный шум.для gray уменьшает границы черного и увеличивает границы белого 
                  убирает черные вкрапления на белом (если все значения ядра совпали)
                  т.е. уменьшает черный усиливает белый
        открытие - последовательно выполнить Erosion после Dilation
                   убрать белый шум на черном фоне
        закрытие - последовательно выполнить Dilation после Erosion 
                   убрать черный шум внутри белой фигуры
    
    Билатеральное размытие
    https://docs.opencv.org/4.5.5/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
    
    Последовательность действий для контура:
        rval, frame_1 = vc.read(0) 
        frame_2_blur = cv2.bilateralFilter(frame_1,d=9,sigmaColor=75,sigmaSpace=75)
        frame_3_hsv = cv2.cvtColor(frame_2_blur,cv2.COLOR_BGR2HSV)
        frame_4_mask = cv2.inRange(frame_3_hsv,colorLow,colorHigh,cv2.CV_8U)
        frame_5_mask_morph = cv2.dilate(frame_4_mask,kernel,iterations=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # выполнить нужно закрытие (а не открытие как в видео)
        frame_5_mask_morph = cv2.erode(frame_5_mask_morph,kernel,iterations=3)
        frame_6_edge = cv2.Canny(frame_5_mask_morph,threshold_min,threshold_max)
        contours,_h = cv2.findContours(frame_6_edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            cv2.drawContours(frame_1,contours,index_contours,color_contours,thickness,cv2.LINE_AA)
        cv2.imshow(window,frame_1)

        если надо:
        bitwise_result = cv2.bitwise_and(frame_1, frame_1, mask = frame_4_mask)
'''    
def smoothing_and_morphological_transformations():
    window = "smoothing_and_morphological_transformations"
    
    window_bitwise_and = "bitwise_and"
    window_mask = "#2 mask"
    window_morphology = "#3 morphology"
    window_edge = "#4 edge"
    
    
    cv2.namedWindow(window,flags=cv2.WINDOW_AUTOSIZE|cv2.WINDOW_KEEPRATIO|cv2.WINDOW_GUI_EXPANDED)
    cv2.moveWindow(window, 0, 0)

    cv2.namedWindow(window_mask,flags=cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO|cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(window_mask, 350, 260)
    cv2.moveWindow(window_mask, 1050, 0)
    
    cv2.namedWindow(window_bitwise_and,flags=cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO|cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(window_bitwise_and, 350, 260)
    cv2.moveWindow(window_bitwise_and, 650, 0)
    
    cv2.namedWindow(window_edge,flags=cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO|cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(window_edge, 350, 260)
    cv2.moveWindow(window_edge, 1050, 300)
    
    cv2.namedWindow(window_morphology,flags=cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO|cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(window_morphology, 350, 260)
    cv2.moveWindow(window_morphology, 650, 300)
    
    # верхний порог цвета
    cv2.createTrackbar("H max",window,0,360,func_pass)
    cv2.createTrackbar("S max",window,0,255,func_pass)
    cv2.createTrackbar("V max",window,0,255,func_pass)    
    # нижний порог цвета
    cv2.createTrackbar("H min",window,0,360,func_pass)
    cv2.createTrackbar("S min",window,0,255,func_pass)    
    cv2.createTrackbar("V min",window,0,255,func_pass)
    
    cv2.createTrackbar("threshold min",window,0,300,func_pass)
    cv2.createTrackbar("threshold max",window,0,300,func_pass)
    cv2.setTrackbarPos("threshold min",window,20)
    cv2.setTrackbarPos("threshold max",window,200)
        
    # red settings
    cv2.setTrackbarPos("H max",window,183)
    cv2.setTrackbarPos("S max",window,208)
    cv2.setTrackbarPos("V max",window,255)
    
    cv2.setTrackbarPos("H min",window,149)
    cv2.setTrackbarPos("S min",window,149)
    cv2.setTrackbarPos("V min",window,15)
    
    vc = cv2.VideoCapture(0 ,cv2.CAP_VFW)
    #vc = cv2.VideoCapture('/container_data/source/video/murom_640x360.mp4')
    
    # ядро размытия [3,3] или [5,5]
    core_a = 9
    core_b = 9
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    if vc.isOpened():  
        rval = True 
    else: 
        rval = False 
    while rval: 
        rval, frame_1 = vc.read(0)   
   
        # get trackbar
        h = cv2.getTrackbarPos("H max",window)
        s = cv2.getTrackbarPos("S max",window)
        v = cv2.getTrackbarPos("V max",window)
        hl = cv2.getTrackbarPos("H min",window)
        sl = cv2.getTrackbarPos("S min",window)
        vl = cv2.getTrackbarPos("V min",window)
        
        # Сглаживание -------------------------------------------------------------------------------------------
        '''
            # различные формы сглаживания
            frame_gauss = cv2.GaussianBlur(frame, (15, 15), 0)
            
            frame_gauss = cv2.medianBlur(frame, 7)
         
            frame_gauss = cv2.bilateralFilter(frame, 15 ,75, 75)
            
            kernal = numpy.ones((9, 9), numpy.float32)/255
            frame_gauss = cv2.filter2D(frame, -1, kernal_blur)
        '''
        # Сглаживание среднеарифметическое   
        if False:
            frame_2_blur = cv2.blur(frame_1,[core_a,core_b])
            cv2.imshow("blur",frame_2_blur)   
             
        # Сглаживание Gaussian более еффективен чем `blur`  
        if False:
            frame_2_blur = cv2.GaussianBlur(frame_1, (core_a,core_b), 0)
            cv2.imshow("blur",frame_2_blur)
        
        if False:
            kernal_blur = numpy.ones((9, 9), numpy.float32)/255
            frame_2_blur = cv2.filter2D(frame_1, -1, kernal_blur)
            
        # лучше справился чем filter2D и GaussianBlur и blur 
        if False:
            frame_2_blur = cv2.medianBlur(frame_1, 7)
            
        # Сглаживание Билатеральное еще лучше Gaussian за счет сохранения краев  
        # меньшая скорость работы
        if True:
            frame_2_blur = cv2.bilateralFilter(frame_1,d=9,sigmaColor=75,sigmaSpace=75)
            #cv2.imshow("blur",frame_2_blur)
         
        # Переобразование в цветовое пространство HSV --------------------------------------------------------
        # необходимо для создания бинарной маски 0 или 1 
        colorHigh = numpy.array([h,s,v])
        colorLow = numpy.array([hl,sl,vl])        
        frame_3_hsv = cv2.cvtColor(frame_2_blur,cv2.COLOR_BGR2HSV)
        
        # Построить frame mask на основе разности диапазонов вхождения ---------------------------------------
        frame_4_mask = cv2.inRange(frame_3_hsv,colorLow,colorHigh,cv2.CV_8U)
        cv2.imshow(window_mask,frame_4_mask)
        
        # Наложение маски на цветной поток (для просмотра)
        if True:
            bitwise_result = cv2.bitwise_and(frame_1, frame_1, mask = frame_4_mask)
            cv2.imshow(window_bitwise_and,bitwise_result)
        
        # Морфологическое открытие и закрытие ----------------------------------------------------------------
        #frame_5_mask_morph = cv2.morphologyEx(frame_4_mask, cv2.MORPH_CLOSE, kernel) # dilate => erode 
        frame_5_mask_morph = cv2.dilate(frame_4_mask,kernel,iterations=3)
        frame_5_mask_morph = cv2.erode(frame_5_mask_morph,kernel,iterations=3)
        frame_5_mask_morph = cv2.morphologyEx(frame_5_mask_morph, cv2.MORPH_OPEN, kernel) # erode => dilate
        cv2.imshow(window_morphology,frame_5_mask_morph)
         
        # Края -----------------------------------------------------------------------------------------------
        # сравнить края так как данные хорышие передаются, а края строют плохо и дальше морфология не помогает
        threshold_min = cv2.getTrackbarPos("threshold min",window) # первый порог для процедуры гистерезиса.
        threshold_max = cv2.getTrackbarPos("threshold max",window) # второй порог для процедуры гистерезиса.
        frame_6_edge = cv2.Canny(frame_5_mask_morph,threshold_min,threshold_max)
        cv2.imshow(window_edge,frame_6_edge)
        
        # TODO:Попытаться улучшить используя пороги 
        #retval, frame_6_edge = cv2.threshold(frame_5_mask_morph,threshold_min,threshold_max,cv2.THRESH_BINARY)
        #cv2.imshow(window_edge,frame_6_edge)
         
        # Контур ---------------------------------------------------------------------------------------------
        contours,_hierarchy = cv2.findContours(frame_6_edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        # сортировка контуров по убыванию, т.е. самый большой первый
        contours = sorted(contours,key=cv2.contourArea,reverse=True)
        cv2.putText(frame_1,"["+str(len(contours))+"]",(10, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
        index_contours = -1
        color_contours = (0,255,255)
        thickness = 1
        if len(contours) > 0:
            #cv2.drawContours(frame_1,[contours[0]],index_contours,color_contours,thickness,cv2.LINE_AA)
            cv2.drawContours(frame_1,contours,index_contours,color_contours,thickness,cv2.LINE_AA)
        cv2.imshow(window,frame_1)
         
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # exit on ESC break 
            vc.release()
            cv2.destroyAllWindows()
            return 0      

# ==========================================================================================================  
def show_max(h_min,s_min,v_min,h_max,s_max,v_max):
    global img,window
    frameBGR = cv2.GaussianBlur(img, (7, 7), 0)
    # Convert the frame to HSV colour model.
    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
    # HSV values to define a colour range.
    colorLow = numpy.array([h_min,s_min,v_min])
    colorHigh = numpy.array([h_max,s_max,v_max])
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
    result = cv2.bitwise_and(img, img, mask = mask)
    cv2.imshow(window, result)# Show final output image

def show_min(h_min,s_min,v_min,h_max,s_max,v_max):
    global img,window
    frameBGR = cv2.GaussianBlur(img, (7, 7), 0)
    # Convert the frame to HSV colour model.
    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
    # HSV values to define a colour range.
    colorLow = numpy.array([h_min,s_min,v_min])
    colorHigh = numpy.array([h_max,s_max,v_max])
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
    result = cv2.bitwise_and(img, img, mask = mask)
    cv2.imshow(window, result)# Show final output image
    
       
def h_max(h_max_value):
    global gh_max,gs_max,gv_max,gh_min,gs_min,gv_min
    gh_max = h_max_value
    show_max(gh_min,gs_min,gv_min,gh_max,gs_max,gv_max)
def s_max(s_max_value):
    global gh_max,gs_max,gv_max,gh_min,gs_min,gv_min
    gs_max = s_max_value
    show_max(gh_min,gs_min,gv_min,gh_max,gs_max,gv_max)
def v_max(v_max_value):
    global gh_max,gs_max,gv_max,gh_min,gs_min,gv_min
    gv_max = v_max_value
    show_max(gh_min,gs_min,gv_min,gh_max,gs_max,gv_max)          
def h_min(h_min_value):
    global gh_max,gs_max,gv_max,gh_min,gs_min,gv_min
    gh_min = h_min_value
    show_min(gh_min,gs_min,gv_min,gh_max,gs_max,gv_max) 
def s_min(s_min_value):
    global gh_max,gs_max,gv_max,gh_min,gs_min,gv_min
    gs_min = s_min_value
    show_min(gh_min,gs_min,gv_min,gh_max,gs_max,gv_max)
def v_min(v_min_value):
    global gh_max,gs_max,gv_max,gh_min,gs_min,gv_min
    gv_min = v_min_value
    show_min(gh_min,gs_min,gv_min,gh_max,gs_max,gv_max)    
 
# ==========================================================================================================  
try:   
    if False: 
        test_url_video() 
      
    if False:  
        default_cap()
   
    if False:
        threshold_example()
       
    if False:
        in_range_example()   
        
    if False:
        selecting_object_by_color()
        
    if True:
        selecting_object_by_shape()
        
    if False:
       is_l_button_down_activ = False
       gradients_and_edges_and_contours()
    
    if False:
        smoothing_and_morphological_transformations()
        
        
    # Debug HSV ----------------------------------------------------------------------------------
    if False:
        window = "hsv"
        window2 = "rgb"
        cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
        img = cv2.imread(cv2.samples.findFile("/container_data/source/sRGB.svg.png"),cv2.IMREAD_COLOR)
        #img = numpy.zeros((700,700,3),numpy.uint8) 
        cv2.imshow(window,img)
        
        gh_max = 360
        gs_max = 255
        gv_max = 255
        gh_min = 0
        gs_min = 0
        gv_min = 0
        
        # верхний порог цвета
        cv2.createTrackbar("H max",window,0,360,h_max)
        cv2.createTrackbar("S max",window,0,255,s_max)
        cv2.createTrackbar("V max",window,0,255,v_max)    
        # нижний порог цвета
        cv2.createTrackbar("H min",window,0,360,h_min)
        cv2.createTrackbar("S min",window,0,255,s_min)    
        cv2.createTrackbar("V min",window,0,255,v_min) 
        
        cv2.setTrackbarPos("H max",window,gh_max)
        cv2.setTrackbarPos("S max",window,gs_max)
        cv2.setTrackbarPos("V max",window,gv_max)
        cv2.setTrackbarPos("S min",window,gs_min)
        cv2.setTrackbarPos("V min",window,gv_min)
        
        key = cv2.waitKey(0) & 0xFF
        if key == 27: # exit on ESC break 
            cv2.destroyAllWindows()
    # -------------------------------------------------------------------------------------------
      
except Exception as e: 
    print(e)
    cv2.destroyAllWindows()
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
    
# Use videowriter
# https://www.programcreek.com/python/example/72134/cv2.VideoWriter
# https://stackoverflow.com/questions/29317262/opencv-video-saving-in-python

# example
# /home/jeka/Projects/OpenCV/opencv/samples/python

# Run:    
# python  /container_data/camera_example/camera.py 
