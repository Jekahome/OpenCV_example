#!/usr/bin/python
# -*- coding: utf-8 -*-
from curses import window
from statistics import variance
import cv2
import numpy
import sys, os
import copy

from matplotlib import pyplot as plt # apt-get install -y python3-pip python3-tk&&  python -mpip install -U matplotlib && pip install pyqt5
from platform import python_version

print('OpenCV',cv2.__version__)  
print('Python',python_version())

'''
Как правило, анализ изображения алгоритмами компьютерного зрения проходит следующие этапы (но некоторых этапов может и не быть):

1. Предобработка изображения. 
    На этом этапе может происходить улучшения качества изображения, такое как увеличение контрастности, 
    повышение резкости или наоборот, размытие изображения, чтобы удалить из него шумы и мелкие незначительные детали. 
    Все это нужно для того, чтобы в дальнейшем было легче производить анализ изображения.
    - Для размытия различные cv2.GaussianBlur,cv2.medianBlur,...
    - Для повышение контрастности cv2.filter2D с ядром kernal = numpy.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], numpy.float32)
    - Для повышение резкости cv2.filter2D с ядром kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    
2. Промежуточная фильтрация. (границы, углы)
    На этом этапе к изображению применяют различные фильтры, для того, чтобы обозначить на изображения 
    области интереса или облегчить работу специальным алгоритмам анализа изображения.

3. Выявление специальных признаков (фич). 
    Это может быть выделение особых точек, выделение контуров или еще каких-либо признаков.
    Например, особые точки, а именно, углы, изгибы, а так же некоторые бросающиеся в глаза особенности определенных объектов. 
    Например, на человеческом лице особыми точками являются глаза, нос, уголки рта.
    
4. Высокоуровневый анализ. ( нахождение областей интереса [:,:] ROI (Region Of Interest — регион интересов — интересующая область изображения))
    На этом этапе по найденным признакам на изображения определяться конкретные объекты, и, как правило, их координаты. 
    Так же на этом этапе может происходить сегментация либо какая-то иная высокоуровневая обработка.
'''

'''
Предобработка изображения. (размытие)

Сглаживание (размытие или наз. фильтрация изображения)
Уменьшить шум или искажения

Большая часть энергии сигнала или изображения сосредоточена в полосах низких и средних частот амплитудного спектра. 
В диапазоне высоких частот полезная информация будет подавлена ​​шумом. 
Следовательно, фильтр, который может уменьшить амплитуду высокочастотных компонентов, может уменьшить влияние шума.

Гауссовский шум может возникнуть, например, от помех. 
Или, если у нас было плохое освещение, картинка получилась темная, и мы попытались как-то исправить это, например, увеличить контрастность. 
Шумы при этом тоже усилятся.

Линейная фильтрация: прямоугольная фильтрация, фильтрация среднего, гауссова фильтрация

    Различные формы сглаживания
    frame_gauss = cv2.GaussianBlur(frame, (15, 15), 0)
    
    frame_gauss = cv2.medianBlur(frame, 7)

    frame_gauss = cv2.bilateralFilter(frame, 15 ,75, 75)
    
    kernal = numpy.ones((9, 9), numpy.float32)/255
    frame_gauss = cv2.filter2D(frame, -1, kernal_blur)
''' 

def filter_example():
    
    img = cv2.imread("/container_data/source/filter/Gaussian_noise.png",cv2.IMREAD_ANYCOLOR)
   
    # boxFilter (алгоритм линейной фильтрации)
    ddepth = -1 # глубина выходного изображения CV_8U/CV_16S/CV_32F/CV_64F
    ksize = (3,3) # размытие размера ядра
    boxFilter = cv2.boxFilter(img, ddepth, ksize, normalize=False,borderType=cv2.BORDER_DEFAULT) 
    #boxFilter = cv2.sqrBoxFilter(img, ddepth, ksize, normalize=False,borderType = cv2.BORDER_REPLICATE)

    # blur (алгоритм линейной фильтрации, Средний фильтр, усредняет)
    ksize = (3,3) # размытие размера ядра
    blur = cv2.blur(img, ksize)
    
    # GaussianBlur (алгоритм линейной фильтрации,взвешенное среднее,для удаление гауссовского шума)
    GaussianBlur = cv2.GaussianBlur(img, (3, 3), 0)
    
    # filter2D
    kernel = numpy.ones((9, 9), numpy.float32)/255 # ядро для размытия
    #kernel = numpy.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], numpy.float32)# ядро для повышения контрастности изображения
    filter2D = cv2.filter2D(img, -1, kernel)
    
    # medianBlur (нелинейная фильтрация. Этот метод устраняет импульсный шум, шум соли и перца)
    # Она также очень эффективна при фильтрации импульсных помех и шума сканирования изображения.
    # не подходит для некоторых изображений с большим количеством деталей, особенно для линий и шпилей
    # В медианном размытии центральный пиксель изображения заменяется медианой всех пикселей в области ядра, 
    # в результате чего это размытие наиболее эффективно при удалении шума в стиле «соли»
    # медленне в 5 раз чем blur 
    medianBlur = cv2.medianBlur(img, 3)    
    
    # bilateralFilter нелинейная фильтрация,сохраняет края
    # не может чисто отфильтровать высокочастотный шум в цветном изображении 
    # и может хорошо фильтровать только низкочастотную информацию
    bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)
    
    cv2.imshow("original", img)
    cv2.imshow("boxFilter", boxFilter)
    cv2.imshow("blur", blur)
    cv2.imshow("GaussianBlur", GaussianBlur)
    cv2.imshow("medianBlur", medianBlur)
    cv2.imshow("filter2D", filter2D)
    cv2.imshow("bilateralFilter", bilateralFilter)
    
    cv2.imwrite("/container_data/source/filter/boxFilter.png", boxFilter)
    cv2.imwrite("/container_data/source/filter/blur.png", blur)
    cv2.imwrite("/container_data/source/filter/GaussianBlur.png", GaussianBlur)
    cv2.imwrite("/container_data/source/filter/medianBlur.png", medianBlur)
    cv2.imwrite("/container_data/source/filter/filter2D.png", filter2D)
    cv2.imwrite("/container_data/source/filter/bilateralFilter.png", bilateralFilter)
    
    key = cv2.waitKey(0) & 0xFF
    if key == 27 or key == ord( "q" ): # exit on ESC break 
        cv2.destroyAllWindows()
 
'''
 Фильтр на изображение можно наложить и виде линейной свертки с определенной матрицей

 Мы берем скользящее окно, попиксельно умножаем яркость каждого пикселя этого окна на коэффициент в матрице, 
 складываем и результат записываем в центральную точку окна. 
 Потом окно сдвигаем на один пиксель и делаем то же самое. И так пока не пройдем по всему изображению. 


Пример:
    kernel = [
        [ 0,  1,  0],
        [ 0,  0,  0],
        [ 0,  0,  0],
    ]
    img = [
        [ 47, 48,  35],
        [ 45, 50,  45],
        [ 47, 49,  11],        
    ]
    filter = cv2.filter2D(img, -1, kernel)
    center_pixel = 47*0 + 48*1 + 35*0 + 45*0 + 50*0 + 45*0 + 47*0 + 49*0 + 11*0 == 48
    Результат светки: элемент (в центре -1) принял значение операции свертки матрицы
    [
        [ 0, 0, 0],
        [ 0,48, 0],
        [ 0, 0, 0],        
    ]
    
    
''' 
def matrix_filter():
    
    '''
    Как должен вести себя алгоритм свёртки на краях изображения?
    Создание изображение большего размера, чем исходное, у которого на краях будут заданное значение пикселей. Начинать обработку картинки нужно тогда не с 0-го, а с 1-го пикселя (и до n-1-го пискселя)
    Функция свёртки cvFilter2D() внутри себя уже вызывает функцию cvCopyMakeBorder() с параметром IPL_BORDER_REPLICATE
    
    img = numpy.zeros((12,12,3),numpy.uint8)
    img[:,:]=255
    img[6,6]=0
    print('img',img.shape)
    add_top = 1
    add_down = 1
    add_left = 1
    add_right = 1
    img = cv2.copyMakeBorder(img,add_top,add_down,add_left,add_right,cv2.BORDER_CONSTANT,0)
    '''
     
    window = "original"
    cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    
    window2 = "Sobel"
    cv2.namedWindow(window2,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    
    window3 = "Резкость"
    cv2.namedWindow(window3,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    
    window4 = "Контрасность"
    cv2.namedWindow(window4,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    
    window5 = "Края Лапласиан"
    cv2.namedWindow(window5,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    
    window6 = "Яркость"
    cv2.namedWindow(window6,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    
    window7 = "Сглаживание"
    cv2.namedWindow(window7,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    
    window8 = "Затемнение"
    cv2.namedWindow(window8,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    
    
    img = cv2.imread("/container_data/source/filter/not_noise.png",cv2.IMREAD_ANYCOLOR)
    cv2.imshow(window, img)
    
    # Края Sobel
    kernel = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], numpy.float32)  
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    filter = cv2.filter2D(img_gray, -1, kernel)# -1 якорь светки центр, целевое значение 
    cv2.imshow(window2, filter)
    
    # Края Лапласиан
    kernel = numpy.array([[0,1,0], [1,-4,1], [0,1,0]])
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    filter = cv2.filter2D(img_gray, -1, kernel)# -1 якорь светки центр, целевое значение 
    cv2.imshow(window5, filter)
    
    # Повышение резкости
    sharpness = 2
    kernel = numpy.array([[-0.1,-0.1,-0.1], [-0.1,sharpness,-0.1], [-0.1,-0.1,-0.1]])
    filter = cv2.filter2D(img, -1, kernel)# -1 якорь светки центр, целевое значение 
    cv2.imshow(window3, filter)
    
    # Повышение контрастности
    contrast = 5
    kernel = numpy.array([[0, -1, 0], [-1, contrast, -1], [0, -1, 0]], numpy.float32)
    filter = cv2.filter2D(img, -1, kernel)# -1 якорь светки центр, целевое значение 
    cv2.imshow(window4, filter)
    
    # Повышение яркости
    brightness = 1.5
    kernel = numpy.array([[-0.1,0.2,-0.1], [0.2,brightness,0.2], [-0.1,0.2,-0.1]])
    filter = cv2.filter2D(img, -1, kernel)# -1 якорь светки центр, целевое значение 
    cv2.imshow(window6, filter)
    
    # Сглаживание
    blur = 0.1
    kernel = numpy.array([[0.1,0.1,0.1], [0.1,blur,0.1], [0.1,0.1,0.1]])
    filter = cv2.filter2D(img, -1, kernel)# -1 якорь светки центр, целевое значение 
    cv2.imshow(window7, filter)
    
    # Затемнение
    blackout = 0.5
    kernel = numpy.array([[-0.1,0.1,-0.1], [0.1,blackout,0.1], [-0.1,0.1,-0.1]])
    filter = cv2.filter2D(img, -1, kernel)# -1 якорь светки центр, целевое значение 
    cv2.imshow(window8, filter)
    
    key = cv2.waitKey(0) & 0xFF
    if key == 27 or key == ord( "q" ): # exit on ESC break 
        cv2.destroyAllWindows()
        
def contrast():
    window = "original"
    cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    window_result = "contrast"
    cv2.namedWindow(window_result,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    
    img = cv2.imread("/container_data/source/filter/not_noise.png",cv2.IMREAD_ANYCOLOR)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8)) #  clipLimit - определят, насколько контрастнее станет фото

    # Далее, для наложения выровненных гистограмм на изображение преобразуется формат LAB, 
    # который представляет собой один канал светлоты L и два канала цвета A и B
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    # Наконец, мы можем применить выровненные гистограммы (только к каналу L – светлота)
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    # Затем объединяем эти каналы и производим обратное преобразование
    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
     
    cv2.imshow(window, img)         
    cv2.imshow(window_result, img2)  
        
    key = cv2.waitKey(0) & 0xFF
    if key == 27 or key == ord( "q" ): # exit on ESC break 
        cv2.destroyAllWindows()
        
'''
Выявление фич

Выявить углы

После выявления углов:
    - Составить из точек различеные геометрические фигуры, например, треугольники.
    - Превратить в цепочки точек и уже их отдельно анализировать.
    - Описать как граф и применять к нему алгоритмы на графах.
https://habr.com/ru/post/656489/

'''        
def corner_Harris():
    try: 
        window = "original"
        cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        window_result = "corner"
        cv2.namedWindow(window_result,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    
        img = cv2.imread("/container_data/source/filter/car.jpg",cv2.IMREAD_ANYCOLOR)
        img_copy = img.copy()
        
        # конвертировать входное изображение в Цветовое пространство в оттенках серого
        img_operated = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # изменить тип данных
        # установка 32-битной плавающей запятой
        img_operated = numpy.float32(img_operated)

        # применить метод cv2.cornerHarris
        # для определения углов с соответствующими
        # значения в качестве входных параметров
        img_dest = cv2.cornerHarris(img_operated, 2, 5, 0.07)

        # Результаты отмечены через расширенные углы
        img_dest = cv2.dilate(img_dest, None)

        # Возвращаясь к исходному изображению,
        # с оптимальным пороговым значением
        threshold = 0.01 # усилить порог 0.05
        img[img_dest > threshold * img_dest.max()] = [0, 0, 255]
        cv2.imwrite("/container_data/source/filter/cornerHarris.jpg",img,[cv2.IMWRITE_EXR_COMPRESSION_NO,1])
        
        cv2.imshow(window, img)         
        cv2.imshow(window_result, img_copy)  
            
        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord( "q" ): # exit on ESC break 
            cv2.destroyAllWindows()
        
    except Exception as e: 
        print(e)
        cv2.destroyAllWindows()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
          
        
          
                    
# Фильтрация ------------------------------------------------------------------------------------------------------    
     
if True:        
    filter_example()

if False:
    matrix_filter()

if False:
    contrast()
    
if False:
    corner_Harris()  
    
          
# Run:    
# python  /container_data/image_example/filter.py   