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
Морфология — это простые преобразования, применяемые к бинарным изображениям 
    или изображениям в градациях серого(cv2.MORPH_TOPHAT,cv2.MORPH_BLACKHAT).

Морфологические операции для увеличения размеров объектов на изображениях, а также для их уменьшения . 
Мы также можем использовать морфологические операции, чтобы закрыть промежутки между объектами, а также открыть их.

Морфология для изменения формы изображения (Улучшает распознавание по цвету и контуру)
    Erosion(сужение) - Уменьшает белый шум.для gray уменьшает границы белого и увеличивает границы черного 
                убирает белые вкрапления на черном (если все значения ядра совпали)
                т.е. уменьшает белый усиливает черный
    Dilation(расширение) - Уменьшает черный шум.для gray уменьшает границы черного и увеличивает границы белого 
                убирает черные вкрапления на белом (если все значения ядра совпали)
                т.е. уменьшает черный усиливает белый
    Erosion и Dilation с нужным размером ядра возможно воздействовать на различные характеристики изображения
             
    Открытие - последовательно выполнить Erosion после Dilation
                убрать белый шум на черном фоне
    Закрытие - последовательно выполнить Dilation после Erosion 
                убрать черный шум внутри белой фигуры
                
    Градиент — это разница между erode и dilate изображения. 
               Это дает контур объекта переднего плана
               
    "white hat" - Операция cv2.MORPH_TOPHAT используется для выявления ярких областей изображения на темном фоне 
                  Разница между входным изображением и результатом Открытия
                     
    "black hat" - Операция cv2.MORPH_BLACKHAT используется для выявления темных областей на ярком фоне 
                  Разница между входным изображением и результатом Закрытия  
                              
    MORPH_HITMISS (попадание или промах) - для обнаружение формы или поиска определенных закономерностей в изображении

https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html                  
'''
def morphological_transformations():
    try:
        window = "white noise"
        window_result = "white noise result"
        window_2 = "black noise"
        window_2_result = "black noise result"
   
        img_1 = cv2.imread("/container_data/source/morphology/white_noise.png") 
        img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
        img_1 = cv2.bilateralFilter(img_1,d=9,sigmaColor=75,sigmaSpace=75)
        
        img_2 = cv2.imread("/container_data/source/morphology/black_noise.png") 
        img_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
        img_2 = cv2.bilateralFilter(img_2,d=9,sigmaColor=75,sigmaSpace=75)
        
        img_3 = cv2.imread("/container_data/source/morphology/black_A.png") 
        img_3 = cv2.cvtColor(img_3,cv2.COLOR_BGR2GRAY)
        img_3 = cv2.bilateralFilter(img_3,d=9,sigmaColor=75,sigmaSpace=75)
        
        img_4 = cv2.imread("/container_data/source/morphology/white_A.png") 
        img_4 = cv2.cvtColor(img_4,cv2.COLOR_BGR2GRAY)
        img_4 = cv2.bilateralFilter(img_4,d=9,sigmaColor=75,sigmaSpace=75)
        
        img_5  = cv2.imread("/container_data/source/morphology/moon.png") 
        img_5 = cv2.cvtColor(img_5,cv2.COLOR_BGR2GRAY)
         
        img_6 = cv2.imread("/container_data/source/morphology/car.webp") 
        img_6 = cv2.cvtColor(img_6,cv2.COLOR_BGR2GRAY)
   
        (h1,w1) = img_1.shape[:2]
        (h2,w2) = img_2.shape[:2]
        
        cv2.namedWindow(window,flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED ) 
        cv2.moveWindow(window, 0, 0)
        cv2.resizeWindow(window, int(w1*2), int(h1*2))
        
        cv2.namedWindow(window_result,flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED ) 
        cv2.moveWindow(window_result, int(w1*2), 0)
        cv2.resizeWindow(window_result, int(w1*2), int(h1*2))
        
        cv2.namedWindow(window_2,flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED ) 
        cv2.moveWindow(window_2, 0, int(h1*2))
        cv2.resizeWindow(window_2, int(w2*2), int(h2*2))
        
        cv2.namedWindow(window_2_result,flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED ) 
        cv2.moveWindow(window_2_result, int(w1*2), int(h1*2))
        cv2.resizeWindow(window_2_result, int(w1*2), int(h1*2))
        
 
        # 1 Формируем ядро
        '''
            cv2.MORPH_RECT - прямоугольный структурный элемент (numpy.ones((5,5),numpy.uint8))
            cv2.MORPH_CROSS - крестообразный структурный элемент
            cv2.MORPH_ELLIPSE - эллиптический структурирующий элемент
        '''
        #kernel = numpy.ones((5,5),numpy.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # 2 Применяем морфологические операции
        '''
        Расширение Dilation 
            frame_morph = cv2.dilate(img,kernel,iterations=1)
            
        Сужение Erosion
            frame_morph = cv2.erode(frame_morph,kernel,iterations=1) 
            
        Закрытие - это последовательно dilate => erode
            frame_morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                или
            frame_morph = cv2.dilate(img,kernel,iterations=1) 
            frame_morph = cv2.erode(frame_morph,kernel,iterations=1) 
        
        Открытие - это последовательно erode => dilate
            frame_morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
                или
            frame_morph = cv2.erode(img,kernel,iterations=1)  
            frame_morph = cv2.dilate(frame_morph,kernel,iterations=1) 
            
        Градиент — это разница между erode и dilate изображения. 
            Это дает контур объекта переднего плана.
            frame_morph = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        
        "white hat" - выявления ярких областей изображения на темном фоне 
                      frame_morph = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        
        "black hat" - выявления темных областей изображения на ярком фоне
                      frame_morph = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        
        '''
        # Для картинки с белым шумом используем `открытие` т.е. уменьшим сначала белый что бы черный поглотил все незначительное белое,
        # а потом увеличим обратно белое восстановив основную форму
        frame_morph_1 = cv2.erode(img_1,kernel,iterations=1)
        frame_morph_1 = cv2.dilate(frame_morph_1,kernel,iterations=1)
        cv2.imshow(window,img_1)
        cv2.imshow(window_result,frame_morph_1)
        
        # Для картинки с черным шумом используем `закрытие` т.е. уменьшим черный увелив при этом белый,а потом обратно увеличим черный
        frame_morph_2 = cv2.morphologyEx(img_2, cv2.MORPH_CLOSE, kernel)# dilate => erode
        frame_morph_2 = cv2.morphologyEx(frame_morph_2, cv2.MORPH_OPEN, kernel)# erode => dilate
        cv2.imshow(window_2_result,frame_morph_2)
        
        # MORPH_GRADIENT градиент покажет контур на области где будет разница от операции расширения и сужения
        frame_morph_3 = cv2.morphologyEx(img_3, cv2.MORPH_GRADIENT, kernel)
        
        # MORPH_GRADIENT градиент покажет контур на области где будет разница от операции расширения и сужения
        frame_morph_4 = cv2.morphologyEx(img_4, cv2.MORPH_GRADIENT, kernel)
        
        # MORPH_TOPHAT выделить светлые участки на темном фоне (можно работать с gray)
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))# ядро соответствует искомому обьекту т.е. номерному знаку
        frame_morph_6 = cv2.morphologyEx(img_6, cv2.MORPH_TOPHAT, rectKernel)
       
        # MORPH_BLACKHAT выделить темные участки на светлом фоне (можно работать с gray)
        frame_morph_7 = cv2.morphologyEx(frame_morph_6, cv2.MORPH_BLACKHAT, rectKernel)
          
        img_8 = cv2.imread("/container_data/source/morphology/spece_line.png",cv2.IMREAD_ANYCOLOR) 
        img_8 = cv2.cvtColor(img_8,cv2.COLOR_BGR2GRAY)
        
        cv2.imshow(window_2,img_2)
        
        if True:
            # plot
            titles = [
                    'Original Image','MORPH_OPEN(erode=>dilate)', 
                    'Original Image','MORPH_CLOSE(dilate=>erode)+MORPH_OPEN(erode=>dilate)', 
                    'Original Image','MORPH_GRADIENT',  
                    'Original Image','MORPH_GRADIENT'
                    ]
            images = [
                    img_1, frame_morph_1, 
                    img_2, frame_morph_2,
                    img_3, frame_morph_3,
                    img_4, frame_morph_4,
                    ]
            for i in range(int(len(titles))): 
                # subplot(nrows, ncols, index, **kwargs)
                plt.subplot(int(len(images)/2),2,i+1),plt.imshow(images[i],'gray')
                plt.title(titles[i])
                plt.xticks([]),plt.yticks([])
            #plt.tight_layout()
            plt.show() 

        if True:
            # plot
            titles = [
                    'Original Image','MORPH_TOPHAT',
                    'Original Image','MORPH_BLACKHAT',
                    ]
            images = [
                    img_6, frame_morph_6,
                    frame_morph_6, frame_morph_7
                    ]
            for i in range(int(len(titles))): 
                # subplot(nrows, ncols, index, **kwargs)
                plt.subplot(int(len(images)/2),2,i+1),plt.imshow(images[i],'gray')
                plt.title(titles[i])
                plt.xticks([]),plt.yticks([])
            #plt.tight_layout()
            plt.show() 
             
             
             
        if True:
            kernel = numpy.array((
                [1],
                [1],
                [1]),numpy.uint8)
            # с erode наоборот сужение по вертикали
            frame1 = cv2.dilate(img_8,kernel,iterations=2)
            
            kernel = numpy.array((
               # [ 0,  0,  0],
                 [ 1,  1,  1],
               # [ 0,  0,  0],
                ),numpy.uint8)
            # с erode наоборот сужение по горизонтали
            frame2 = cv2.dilate(img_8,kernel,iterations=1)
            
            kernel = numpy.array((
                [0, 0,  1],
                [0,  1, 1],
                [ 1, 1, 1],
                ),numpy.uint8)
            frame3 = cv2.dilate(img_8,kernel,iterations=1)
           
        if True:
            # plot
            titles = [
                    'Original black:0 white:255',
                    'Увеличить длину', 
                    'Увеличить ширину',
                    'Увеличить диагональ',
                    ]
            images = [
                    img_8, 
                    frame1,
                    frame2,
                    frame3
                    ]
            for i in range(len(images)): 
                # subplot(nrows, ncols, index, **kwargs)
                plt.subplot(4,1,i+1)
                plt.imshow(images[i],'gray')
                plt.title(titles[i], loc='left',fontsize=12 )  
                plt.xticks([]),plt.yticks([])
            plt.tight_layout()
            plt.grid()
            plt.show()        
             
        
    except Exception as e: 
        print(e)
        cv2.destroyAllWindows()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    
# ==========================================================================================================  
'''
Морфологическая операция MORPH_HITMISS (попадание или промах)
(тема для ознакомления с использованием ядер getStructuringElement)
https://theailearner.com/tag/hit-or-miss-transformation-opencv/
 

https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_hitOrMiss.html
T - образная маска, выбрать пиксели которые соотвествуют полностью маске 1 и не соотвествуют маске 0
если в изображении размер области равной маске соотвествует ей т.е. 1 из маски совпадает с 255 изображения и 0 маски совпадает с 0 изображения
тогда вся область занимаемая маской на картинке примет центральное core значение а остальные пиксели станут 0

https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_hitOrMiss.html

Маска:
     0 - это любой пиксель, не участвует
    -1 - принадлежит фону
     1 - шаблон
  
c = 1
kernel_T_3x3 = numpy.array((
        [ 1,  1,  1],
        [-1,  c, -1],
        [-1,  1, -1]),numpy.uint8)  
pic = numpy.array((
        [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [  0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [  0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [  0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [  1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        [  1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [  1, 0, 0, 0, 1, 0, 1, 0, 0, 1]),numpy.uint8)  

frame_out = cv2.morphologyEx(pic, cv2.MORPH_HITMISS, kernel_T_3x3)

   [[0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 1 0 1 0 1 0 1 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0]
    [1 0 0 0 0 0 0 0 0 1]
    [0 0 0 0 0 0 0 0 0 0]]

https://docs.opencv.org/3.4/db/d06/tutorial_hitOrMiss.html
'''
def morphological_transformations_hitmiss():
    try:
        img_color_vertical = cv2.imread("/container_data/source/morphology/vertical.png",cv2.IMREAD_ANYCOLOR) 
        img_vertical   = cv2.cvtColor(img_color_vertical,cv2.COLOR_BGR2GRAY)
        
        img_color_diagonal = cv2.imread("/container_data/source/morphology/diagonal.png",cv2.IMREAD_ANYCOLOR) 
        img_diagonal = cv2.cvtColor(img_color_diagonal,cv2.COLOR_BGR2GRAY)
        
        img_color_figure = cv2.imread("/container_data/source/morphology/figure.png",cv2.IMREAD_ANYCOLOR) 
        img_figure = cv2.cvtColor(img_color_figure,cv2.COLOR_BGR2GRAY)
    
        #ret,img_7 = cv2.threshold(img_7,45,255,cv2.THRESH_BINARY) 
        # Ядро для верхнего правого угла
        kernel_u_r_angl = numpy.array((
                [ 0, -1, -1],
                [ 1,  1, -1],
                [ 0,  1,  0]),numpy.uint8)
        # Ядро для верхнего левого угла
        kernel_u_l_angl = numpy.array((
                [-1, -1,  0],
                [-1,  1,  1],
                [ 0,  1,  0]),numpy.uint8)
        # Ядро для нижнего правого угла
        kernel_d_r_angl = numpy.array((
                [ 0,  1,  0],
                [ 1,  1, -1],
                [ 0, -1, -1]),numpy.uint8)
        # Ядро для нижнего левого угла
        kernel_d_l_angl = numpy.array((
                [ 0,  1,  0],
                [-1,  1,  1],
                [-1, -1,  0]),numpy.uint8)
        # ---------------------------------
        # Ядро для горизонтальной (низ)
        kernel_horizontal_down = numpy.array((
                [-1, -1, -1],
                [ 1,  1,  1],
                [ 1,  1,  1]),numpy.uint8)
        # Ядро для горизонтальной (верх)
        kernel_horizontal_up = numpy.array((
                [ 1,  1,  1],
                [ 1,  1,  1],
                [-1, -1, -1],),numpy.uint8)
        # ---------------------------------
        # Ядро для вертикальной (левый край)
        kernel_vertical_left = numpy.array((
                [1,  1, -1],
                [1,  1, -1],
                [1,  1, -1]),numpy.uint8)
        # Ядро для вертикальной (правый край)
        kernel_vertical_right = numpy.array((
                [-1, 1, 1],
                [-1, 1, 1],
                [-1, 1, 1]),numpy.uint8)       
        # ---------------------------------
        # Ядро для диагонали слева низ (добавить верх)
        kernel_l_line_down = numpy.array((
                [ 1, -1, -1],
                [ 1,  1, -1],
                [ 1,  1,  1]),numpy.uint8)
        # Ядро для диагонали слева верх (добавить низ)
        kernel_l_line_up = numpy.array((
                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1]),numpy.uint8)
        # ---------------------------------
        # Ядро для диагонали cправа (добавить верх)
        kernel_r_line_out = numpy.array((
                [-1, -1,  1],
                [-1,  1,  1],
                [ 1,  1,  1]),numpy.uint8)
         # Ядро для диагонали cправа (добавить низ)
        kernel_r_line_in = numpy.array((
                [ 1,  1,  1],
                [ 1,  1, -1],
                [ 1, -1, -1]),numpy.uint8)
        
        
        # Ядро для поиска левых конечных точек
        kernel3 = numpy.array((
                [ -1, -1,  0],
                [ -1,  1,  0],
                [ -1, -1,  0]),numpy.uint8)
    
    
        if True:
            #frame_morph_8_horizontal_down = cv2.morphologyEx(img_7, cv2.MORPH_HITMISS, kernel_horizontal_down)
            #frame_morph_8_horizontal_up = cv2.morphologyEx(img_7, cv2.MORPH_HITMISS, kernel_horizontal_up)
            #frame_morph_8_horizontal_or = cv2.bitwise_or(frame_morph_8_horizontal_down, frame_morph_8_horizontal_up)
            #frame_morph_8_horizontal_and = cv2.bitwise_and(frame_morph_8_horizontal_down, frame_morph_8_horizontal_up)
        
            frame_morph_8_vertical_left = cv2.morphologyEx(img_vertical, cv2.MORPH_HITMISS, kernel_vertical_left)
            frame_morph_8_vertical_right = cv2.morphologyEx(img_vertical, cv2.MORPH_HITMISS, kernel_vertical_right)
            frame_morph_8_vertical_or = cv2.bitwise_or(frame_morph_8_vertical_left, frame_morph_8_vertical_right)
            frame_morph_8_vertical_and = cv2.bitwise_and(frame_morph_8_vertical_left, frame_morph_8_vertical_right)
            # plot
            titles = [
                    'Original black:0 white:255',
                    'Original black:0 white:255',
                    'Уменьшить слева', 
                    'OR - Усилить white', 
                    'Уменьшить справа',  
                    'AND - Уменьшить white' 
                    ]
            images = [
                    img_color_vertical, 
                    img_color_vertical, 
                    frame_morph_8_vertical_left, 
                    frame_morph_8_vertical_or,
                    frame_morph_8_vertical_right,
                    frame_morph_8_vertical_and
                    ]
            for i in range(len(images)): 
                # subplot(nrows, ncols, index, **kwargs)
                plt.subplot(3,2,i+1)
                plt.imshow(images[i],'gray')
                plt.title(titles[i], loc='left',fontsize=12 ) # ,x=-4.5,y=0.3
                plt.xticks([]),plt.yticks([])
            plt.tight_layout()
            plt.grid()
            plt.show()  
            
        if True:
            frame_morph_8_l_line_down = cv2.morphologyEx(img_diagonal, cv2.MORPH_HITMISS, kernel_l_line_down)
            frame_morph_8_l_line_up = cv2.morphologyEx(img_diagonal, cv2.MORPH_HITMISS, kernel_l_line_up)
            frame_morph_8_l_line_or = cv2.bitwise_or(frame_morph_8_l_line_down, frame_morph_8_l_line_up)
            frame_morph_8_l_line_and = cv2.bitwise_and(frame_morph_8_l_line_down, frame_morph_8_l_line_up)
 
            # plot
            titles = [
                    'Original black:0 white:255',
                    'Original black:0 white:255',
                    'Уменьшить низ',
                    'OR - Усилить',
                    'Уменьшить верх',
                    'AND - Уменьшить'
                    ]
            images = [
                    img_color_diagonal, 
                    img_color_diagonal, 
                    frame_morph_8_l_line_down,
                    frame_morph_8_l_line_or,
                    frame_morph_8_l_line_up, 
                    
                    frame_morph_8_l_line_and
                    ]
            for i in range(len(images)): 
                # subplot(nrows, ncols, index, **kwargs)
                plt.subplot(3,2,i+1 )
                plt.imshow(images[i],'gray')
                plt.title(titles[i],fontsize=12)
                plt.xticks([]),plt.yticks([])
            plt.tight_layout()
            plt.grid()
            plt.show()   
          
        if True:
            frame_morph_8_u_r_angl = cv2.morphologyEx(img_figure, cv2.MORPH_HITMISS, kernel_u_r_angl)
            frame_morph_8_u_l_angl = cv2.morphologyEx(img_figure, cv2.MORPH_HITMISS, kernel_u_l_angl)
            frame_morph_8_d_r_angl = cv2.morphologyEx(img_figure, cv2.MORPH_HITMISS, kernel_d_r_angl)
            frame_morph_8_d_l_angl = cv2.morphologyEx(img_figure, cv2.MORPH_HITMISS, kernel_d_l_angl)
        
            # plot
            titles = [
                    'Original black:0 white:255',
                    'Верхний правый угол', 
                    'Верхний левый угол', 
                    'Нижний правый угол',  
                    'Нижний левый угол',
                    ]
            images = [
                    img_figure, 
                    frame_morph_8_u_r_angl, 
                    frame_morph_8_u_l_angl,
                    frame_morph_8_d_r_angl,
                    frame_morph_8_d_l_angl,                
                    ]
            for i in range(len(images)): 
                # subplot(nrows, ncols, index, **kwargs)
                plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
                plt.title(titles[i])
                plt.xticks([]),plt.yticks([])
            #plt.tight_layout()
            plt.show() 
        
    except Exception as e: 
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)     

# Морфология -----------------------------------------------------------------------------------------------------
if True:
    morphological_transformations()
    
if False:
    morphological_transformations_hitmiss()
    
# Run:    
# python  /container_data/image_example/morphological.py     