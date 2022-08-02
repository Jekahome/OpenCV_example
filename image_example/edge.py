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
Края(счетчики)
Края вычисляются как точки, которые являются экстремумами градиента изображения в направлении градиента.
Дело в том, что краевые пиксели являются локальным понятием: они просто указывают на значительную разницу между соседними пикселями.
Canny(на основе градиента - направленное изменение интенсивности цвета в изображении) делает несколько шагов дальше, 
чем другие детекторы, выход детектора Canny является двоичным изображением с шириной линий 1 px вместо ребер.
Canny находит максимальный градиент, то есть пики в градиенте. Реализация OpenCV Canny() фактически использует Sobel() в своем интерфейсе.


Контуры(ребра)
Контуры часто получаются из ребер, но они нацелены на контуры объектов. 
Таким образом, они должны быть замкнутыми кривыми. Вы можете думать о них как о границах 
(некоторые алгоритмы обработки изображений и библиотеки называют их такими). 
Когда они получены из ребер, вам нужно соединить края, чтобы получить замкнутый контур.
Бонус обнаружения контура, он фактически возвращает множество точек! 
Это здорово, потому что вы можете использовать эти точки дальше для некоторой обработки.
'''

'''
Canny - края

Canny - https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html

https://docs.opencv.org/4.5.5/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0

Общие шаги обнаружения края:

    1.(Сглаживание) Фильтрация: алгоритм обнаружения краев в основном основан на первой и второй производных интенсивности изображения, 
        но производные обычно очень чувствительны к шуму, поэтому фильтрация необходима для повышения производительности детектора границ. 
        Обычно используется метод фильтрации по Гауссу.

    2.(Градиент) Улучшение: Основой улучшения краев является определение значения изменения интенсивности окрестности каждой точки изображения. 
        Алгоритм улучшения может выделять точки со значительными изменениями соседних значений интенсивности серых точек изображения. 
        Он определяется путем расчета величины градиента.
        frame_morph = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        
    3.(Пороги) Обнаружение: благодаря улучшенным изображениям в окрестностях часто бывает много точек с относительно большими значениями градиента. 
        В некоторых приложениях эти точки не являются краевыми точками, которые необходимо найти, 
        поэтому следует использовать какой-либо метод для выбора этих точек. 
        Обычно используется Метод заключается в обнаружении методом пороговой обработки.

    Canny находит края на основе контраста
    
    
Что делать с контуром после его нахождения? 
    Например, следующее:
    - Выявить различные геометрические примитивы (прямые, окружности).
    - Превратить в цепочки точек и уже их отдельно анализировать.
    - Описать как граф и применять к нему алгоритмы на графах.
      Контур можно превратить в граф или в геометрические примитивы, тем самым описав его инвариантно к смещению, повороту и даже масштабированию.
      https://habr.com/ru/post/676838/
      https://habr.com/ru/post/656489/
      
'''
def canny_edge():
    try:
        window = "4 approx_example"
        cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.moveWindow(window, 0, 0)
        #img = cv2.imread("/container_data/source/edge/figure.png",cv2.IMREAD_ANYCOLOR)
        img_1 = cv2.imread("/container_data/source/edge/figure_2.png",cv2.IMREAD_ANYCOLOR)
        img_copy = img_1.copy()
        cv2.resizeWindow(window, 500, 500);
    
        window_tresh = "1 window_tresh"
        cv2.namedWindow(window_tresh,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.moveWindow(window_tresh, 1000, 0)
        cv2.resizeWindow(window_tresh, 500, 500);
        
        window_morph = "2 window_morph"
        cv2.namedWindow(window_morph,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.moveWindow(window_morph, 500, 500)
        cv2.resizeWindow(window_morph, 500, 500);
        
        window_edge = "3 window_edge"
        cv2.namedWindow(window_edge,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.moveWindow(window_edge, 500, 500)
        cv2.resizeWindow(window_edge, 500, 500);
        
        thresh = 255
        cv2.createTrackbar("thresh black->white",window,0,255,func_pass)
        cv2.setTrackbarPos("thresh black->white",window,thresh)
        
        cv2.createTrackbar("min",window,0,255,func_pass)
        cv2.createTrackbar("max",window,0,255,func_pass)
        thresh_min = 20
        thresh_max = 35
        cv2.setTrackbarPos("min",window,thresh_min)
        cv2.setTrackbarPos("max",window,thresh_max)
        C = 1 # 1 сильно выделит край
        cv2.createTrackbar("C",window,0,10,func_pass)
        cv2.setTrackbarPos("C",window,C)
        
        # 1 Фильтрация ---------------------------------------------------------------------------------------------------------------
        # Сгладить края очень мягко, не потеряв в контрастности иначе Canny не сможет построить края !!!
        # Однако Canny не сможет построить край если есть шум, т.е. надо избавиться от шума сглаживанием или другими способами
        #img_3_gray = cv2.GaussianBlur(img_1,(3,3),0)
        #img_3_gray = cv2.bilateralFilter(img_1,d=3,sigmaColor=75,sigmaSpace=75) # сглаживание с сохранением краев      
        img_2_blur = cv2.medianBlur(img_1, 3)
        
        img_3_gray = cv2.cvtColor(img_2_blur,cv2.COLOR_BGR2GRAY)
        
        # 2,3 Градиент и Пороги ------------------------------------------------------------------------------------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_morph_gradient = cv2.morphologyEx(img_3_gray, cv2.MORPH_GRADIENT, kernel)
        ret,img_4_tresh_binary = cv2.threshold(img_morph_gradient,1,255,cv2.THRESH_BINARY_INV)
        cv2.imshow(window_tresh, img_4_tresh_binary)
        
        # Морфология расширение, дорисовать контур
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_5_morph = cv2.erode(img_4_tresh_binary,kernel,iterations=2)# дорисует дыры в контуре  
        img_5_morph = cv2.dilate(img_5_morph,kernel,iterations=1)
        cv2.imshow(window_morph, img_5_morph)
        
        # cv2.Canny() состоит из Шумоподавление,Градиент интенсивности,Не максимальное подавление,Порог гистерезиса
        kernel_size = 3
        thresh_min = cv2.getTrackbarPos("min",window)
        thresh_max = cv2.getTrackbarPos("max",window)
        img_6_edge = cv2.Canny(img_5_morph,thresh_min,thresh_max, apertureSize = kernel_size)
        cv2.imshow(window_edge, img_6_edge)
        
        # Контуры findContours работают только после нахождения крвев (Canny) !!!
        '''
         # cv2.CHAIN_APPROX_SIMPLE - хранить контуры в виде отрезков (нарисовать самому не получится по точкам, так как есть первая точка и последняя)
         # cv2.CHAIN_APPROX_NONE - хранить контуры в виде точек, точки по которым можно быдет строить самому контур
         
        last_point=None
        for point in sel_countour:
            curr_point=point[0]
            if not(last_point is None):
                x1=int(last_point[0])
                y1=int(last_point[1])
                x2=int(curr_point[0])
                y2=int(curr_point[1])
                cv2.line(img_contours, (x1, y1), (x2, y2), 255, thickness=1)
            last_point=curr_point
        '''
        # cv2.RETR_LIST - внутренние контуры включить в выборку
        # cv2.RETR_EXTERNAL - только внешние контуры
        contours,h = cv2.findContours(img_6_edge, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours,key=cv2.contourArea,reverse=True)
        print(f"Count contours={len(contours)}")
        # можно вывести только определенные контуры или все
        sel_countours=[]
        for item in range(len(contours)):
            sel_countours.append(contours[item])
        


        #создадим пустую картинку
        #img_contours = numpy.zeros(img_1.shape)
        img_contours = numpy.uint8(numpy.zeros((img_1.shape[0],img_1.shape[1])))
        
        for cnts_roi in sel_countours:
            area = cv2.contourArea(cnts_roi,oriented=False)# Вычисляет площадь контура. Функция наверняка даст неверный результат для контуров с самопересечениями.
            if area > 1000:
                print(f"Area={area} cnts_roi={len(cnts_roi)}") # -1349782.0  (при oriented=True - по часовой стрелке или + против часовой стрелки)
                cv2.drawContours(img_1,cnts_roi,-1,(128,0,128),3,cv2.LINE_AA)# или  fillPoly 
                cv2.drawContours(img_contours,cnts_roi,-1,(128,0,128),3,cv2.LINE_AA) # только для контура
                arclen = cv2.arcLength(cnts_roi,True) # возвращает длину дуги контура
                eps = 0.03 # точность аппроксимации
                epsilon = eps*arclen
                is_closed_circuit = True # замкнутый контур
<<<<<<< HEAD
                num_top = cv2.approxPolyDP(cnts_roi,epsilon,is_closed_circuit)# Аппроксимирование (упрощение) области контура
=======
                num_top = cv2.approxPolyDP(cnts,epsilon,is_closed_circuit)# Аппроксимирование (упрощение) области контура
>>>>>>> ac03be3047363d1ff75dbd8e9b8973212d3519dc
                x_c,y_c,w_c,h_c = cv2.boundingRect(num_top)# получить квадрат области
                cv2.putText(img_1,f"Area={area}",(x_c+10,y_c+(h_c//2)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),1)
                cv2.putText(img_1,f"Top={len(num_top)}",(x_c+10,y_c+(h_c//2)+20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),1)
                # нарисовать аппроксимированный контур (меньше деталей, видно вершины)
                # cv2.drawContours(img_1,[num_top],0,(128,0,128),3,cv2.LINE_AA)
        cv2.imwrite("/container_data/source/edge/figure_2_result.png", img_1,[cv2.IMWRITE_PNG_COMPRESSION,1])        
        cv2.imshow(window, img_1) 
 
        # plot
        titles = ['Original Image',
                  '1 MORPH_GRADIENT+threshold BIN',
                  '2 Расширение',
                  '3 Canny',
                  '4 drawContours',
                  'Only contours'
                  ]
        images = [img_copy, 
                  img_4_tresh_binary,
                  img_5_morph, 
                  img_6_edge, 
                  img_1,
                  img_contours
                  ]
        plt.figure(figsize=[18,8])
        for i in range(int(len(titles))): 
            # subplot(nrows, ncols, index, **kwargs)
            plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i]);plt.xticks([]),plt.yticks([])
        plt.savefig('/container_data/source/edge/canny_result.png') 
        
        plt.show()          
 
  
        key = cv2.waitKey(0) & 0xFF
        if key == 27: # exit on ESC break 
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
OpenCV имеет три типа градиентных фильтров или фильтров высоких частот: Собела, Шарра и Лапласиана.
Строят ребра в отличии от хитрых краев Canny который использует больше этап обработки

    Оператор Собеля объединяет гауссовское сглаживание и дифференцирование, поэтому в результате получается защита от шума.
   
    Sobel - https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
    
    https://russianblogs.com/article/9615742843/
'''
def sobel_scharr_laplacian_edge():
    try:
        window_sobel = "Sobel"
        cv2.namedWindow(window_sobel,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.moveWindow(window_sobel, 0, 0)
        cv2.resizeWindow(window_sobel, 500, 500);
     
        window_scharr = "Scharr"
        cv2.namedWindow(window_scharr,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.moveWindow(window_scharr, 1000, 0)
        cv2.resizeWindow(window_scharr, 500, 500);
        
        window_laplacian = "Laplacian"
        cv2.namedWindow(window_laplacian,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.moveWindow(window_laplacian, 1000, 0)
        cv2.resizeWindow(window_laplacian, 500, 500);
         
        window_laplacian_matrix = "Laplacian matrix"
        cv2.namedWindow(window_laplacian_matrix,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.moveWindow(window_laplacian_matrix, 1000, 0)
        cv2.resizeWindow(window_laplacian_matrix, 500, 500);
         
        window_canny = "Canny"
        cv2.namedWindow(window_canny,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.moveWindow(window_canny, 1000, 0)
        cv2.resizeWindow(window_canny, 500, 500); 
         
        img_1 = cv2.imread("/container_data/source/edge/cow.jpg",cv2.IMREAD_ANYCOLOR)
        
        
        ddepth = cv2.CV_16S # Глубина целевого изображения. Поскольку наш ввод CV_8U, мы определяем ddepth = CV_16S, чтобы избежать переполнения.
        
        '''
        TODO:
            Проблема с тип выходных данных - cv2.CV_8U или np.uint8
            Переход от черного к белому считается положительным наклоном (он имеет положительное значение), 
            а переход от белого к черному считается отрицательным наклоном (имеет отрицательное значение). 
            Когда вы конвертируете данные в np.uint8, все отрицательные наклоны равны 0. Проще говоря, одно ребро не очень четкое.
            Решение:
            Если вы хотите обнаруживать оба ребра одновременно, лучше сохранить тип выходных данных в более высокой форме, 
            например cv2.CV_16S cv2.CV_64F и т. д., в зависимости от того, какое из них является абсолютным значением. 
            А затем преобразовал обратно в cv2.CV_8U
        '''
        # Sobel -------------------------------------------------------------------------------------------------
        img_1_blur = cv2.GaussianBlur(img_1, (3, 3), 0)
        img_1_gray = cv2.cvtColor(img_1_blur, cv2.COLOR_BGR2GRAY)# Convert the image to grayscale
        sobelx = cv2.Sobel(img_1_gray, ddepth, 1, 0, ksize=3, scale=1, borderType=cv2.BORDER_DEFAULT)
        sobely = cv2.Sobel(img_1_gray, ddepth, 0, 1, ksize=3, scale=1, borderType=cv2.BORDER_DEFAULT)
        sobelx_abs = cv2.convertScaleAbs(sobelx) # Принимаем абсолютное значение
        sobely_abs = cv2.convertScaleAbs(sobely)
        img_sobel = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0) # Объединить x, y два градиентных изображения
        # если дальше нужен CV_8U то:
        # img_sobel = numpy.absolute(img_sobel)
        # img_sobel = np.uint8(img_sobel)
        cv2.imwrite("/container_data/source/edge/Sobel.png", img_sobel,[cv2.IMWRITE_PNG_COMPRESSION,1])        
        cv2.imshow(window_sobel, img_sobel) 
  
        '''
        Когда ядро ​​равно 3, ядро Sobel может выдавать более очевидные ошибки. 
        По этой причине OpenCV предоставляет Scharr, функция работает только с ядрами размера 3, 
        Так же быстро, как функция Sobel, но более высокая точность
        '''
        # Scharr -------------------------------------------------------------------------------------------------
        img_1_blur = cv2.GaussianBlur(img_1, (3, 3), 0)
        scharrx = cv2.Scharr(img_1_blur, ddepth, 1, 0, 3)
        scharry = cv2.Scharr(img_1_blur, ddepth, 0, 1, 3)
        scharrx_abs = cv2.convertScaleAbs(scharrx)
        scharry_abs = cv2.convertScaleAbs(scharry)
        img_scharr = cv2.addWeighted(scharrx_abs, 0.5, scharry_abs, 0.5, 0)
        cv2.imwrite("/container_data/source/edge/Scharr.png", img_scharr,[cv2.IMWRITE_PNG_COMPRESSION,1])        
        cv2.imshow(window_scharr, img_scharr) 
    
    
        # Laplacian ----------------------------------------------------------------------------------------------
        img_1_blur = cv2.GaussianBlur(img_1, (3, 3), 0)
        img_1_gray = cv2.cvtColor(img_1_blur, cv2.COLOR_BGR2GRAY)# Convert the image to grayscale
        img_laplacian = cv2.Laplacian(img_1_gray,ddepth, ksize=3)
        img_laplacian = cv2.convertScaleAbs(img_laplacian)# converting back to uint8 cv2.CV_8U
        cv2.imshow(window_laplacian, img_laplacian) 
        
        kernel = numpy.array([[0,1,0], [1,-4,1], [0,1,0]])
        img_laplacian_matrix = cv2.filter2D(img_1_gray, -1, kernel)
        cv2.imshow(window_laplacian_matrix, img_laplacian_matrix)
        
        '''
        Для сравнения без предварительной подготовки изображения для хитрых краев Canny
        '''
        # Canny --------------------------------------------------------------------------------------------------
        img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)# Convert the image to grayscale
        img_1_blur = cv2.blur(img_1_gray, (3,3))
        img_canny = cv2.Canny(img_1_blur,50,150, apertureSize = 3)
        cv2.imshow(window_canny, img_canny)
        
        # plot
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGRA2RGB)
        titles = ['Sobel x,y',
                  'Sobel x',
                  'Sobel y',
                  'Scharr',
                  'Laplacian',
                  'Laplacian matrix',
                  'Canny',
                  'Original Image']
        images = [img_sobel,
                  sobelx,
                  sobely,
                  img_scharr,
                  img_laplacian,
                  img_laplacian_matrix,
                  img_canny,
                  img_1]
        plt.figure(figsize=[15,8])
        for i in range(int(len(titles))): 
            # subplot(nrows, ncols, index, **kwargs)
            plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.savefig('/container_data/source/edge/sobel_scharr_laplacian_edge.png') 
        plt.show() 
        
        
        key = cv2.waitKey(0) & 0xFF
        if key == 27: # exit on ESC break 
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
Поиск окружностей cv2.HoughCircles
 
'''
def houghCircles_example():
    img = cv2.imread("/container_data/source/edge/figure.png")
    
    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
    filterd_image  = cv2.medianBlur(img_grey,3)

    rows = filterd_image.shape[0]
    threshold_1 = 100
    threshold_2 = 30
    circles = cv2.HoughCircles(filterd_image, cv2.HOUGH_GRADIENT, 1, rows / 8,
                            param1=threshold_1, param2=threshold_2,
                            minRadius=1, maxRadius=100)

    img_res = numpy.zeros(img.shape)
    
    if circles is not None:
        circles = numpy.uint16(numpy.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(img_res, center, 1, (0, 100, 100), 3)
            cv2.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(img_res, center, radius, (255, 0, 255), 3)
            cv2.circle(img, center, radius, (255, 0, 255), 3)

    cv2.imshow('origin', img) # выводим итоговое изображение в окно
    cv2.imshow('res', img_res) # выводим итоговое изображение в окно

    # plot
    titles = ['Image','Only contours']
    images = [img, img_res ]
    for i in range(int(len(titles))): 
        # subplot(nrows, ncols, index, **kwargs)
        plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.savefig('/container_data/source/edge/HoughCircles.png') 
    plt.show() 
        
    key = cv2.waitKey(0) & 0xFF
    if key == 27: # exit on ESC break 
        cv2.destroyAllWindows()
        return 0
  
# ==========================================================================================================  
 
# Край ----------------------------------------------------------------------------------------------------------
if False:
    canny_edge()

if True:
    sobel_scharr_laplacian_edge()
    
if False:
    houghCircles_example()  
      
# Run:    
# python  /container_data/image_example/edge.py    