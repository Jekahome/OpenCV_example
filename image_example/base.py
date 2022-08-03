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
Работа с окнами window
img.shape - размерность изображения (450, 500, 3)  450 строк, 500 столбцов и 3 канала цвета
cv2.resize -  изменение размера
cv2.imwrite - Сохранение изображение на диск

cv2.cvtColor - конвертация цветовых пространств
cv2.getRotationMatrix2D - вычисляет аффинную матрицу двумерного вращения
cv2.warpAffine - Применяет аффинное преобразование к изображению
cv2.flip - Зеркальное отображение (по осям)


Заметьте что высота идет первая при работе со срезами img[0:100, 150:,]
'''
def base_example():    
    # Окна,Открытие,размерность ---------------------------------------------------------------------
    window = "One channel"
    cv2.namedWindow(window,flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
    cv2.setWindowProperty(window, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(window, cv2.WND_PROP_OPENGL, cv2.WINDOW_GUI_EXPANDED)
    '''
        cv2.IMREAD_UNCHANGED - вернуть загруженное изображение как есть
        cv2.IMREAD_GRAYSCALE - преобразовывать изображение в одноканальное изображение в градациях серого
        cv2.IMREAD_COLOR - преобразовывайте изображение в 3-канальное цветное изображение BGR
        cv2.IMREAD_ANYDEPTH - вернуть 16-битное/32-битное изображение, когда входные данные имеют соответствующую глубину, в противном случае преобразовать его в 8-битное
        cv2.IMREAD_ANYCOLOR - изображение читается в любом возможном цветовом формате
    '''
    # img is type cv2::Mat
    img = cv2.imread("/container_data/source/pazl/matrix-sunglasses.jpg",cv2.IMREAD_ANYCOLOR)
    # Проверка загрузки картинки
    if img is None:
        sys.exit("Could not read the image.")
    
    (w,h,color_channel) = img.shape #  (667, 1600, 3)  667 строк, 1600 столбцов и 3 канала цвета
    cv2.resizeWindow(window, int(h/2), int(w/2));
    cv2.imshow(window,img[:,:,0]) # отобразить только канал 0-BLUE,1-GREEN,2-RED
    _blue = img[w,h,0]
    _green = img[w,h,1]
    _red = img[w,h,2]
    
    cv2.imwrite("/container_data/source/pazl/matrix-sunglasses.jpg",img,[cv2.IMWRITE_EXR_COMPRESSION_NO,1])
    
    if cv2.waitKey(0) & 0xFF == ord( "q" ): # exit on `q` break 
        cv2.destroyWindow(window)
    
'''
Цветовые пространства

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
        или
    img = img[:, :, [2, 1, 0]]
        или
    # (::-1) — взять каждый элемент, но в обратном порядке.
    img = img[:, :, ::-1]
        
'''        
def cvtColor_example():        
   
    # конвкертация из стандартного opencv BGR в RGB
    img = cv2.imread("/container_data/source/pazl/pazl2.webp",cv2.IMREAD_ANYCOLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    cv2.namedWindow("color",cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
    cv2.imshow("color", rgb_img)

    # Цветовые пространства
    # https://docs.opencv.org/4.5.5/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0ad3db9ff253b87d02efe4887b2f5d77ee
    color_spaces = ('RGB','GRAY','HSV','LAB','XYZ','YUV')
    color_images = {color : cv2.cvtColor(img, getattr(cv2,'COLOR_BGR2' + color)) for color in color_spaces}
    for color in color_images:
        cv2.namedWindow(color,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
        cv2.imshow(color, color_images[color])
           
'''
Разделение на цвета

cv2.split()
cv2.merge()
'''    
def split_merge_example():    
    # Усилить цвет
    img_NZ_bgr = cv2.imread("/container_data/source/pazl/matrix-sunglasses.jpg",cv2.IMREAD_COLOR)
    b,g,r = cv2.split(img_NZ_bgr)
    b = b+50  # усилить синий
    imgMerged = cv2.merge((b,g,r)) 
    #imgMerged = cv2.merge((r,g,b))  # соберём сразу RGB 

    key = cv2.waitKey(0) & 0xFF
    if key == 27 or key == ord( "q" ): # exit on ESC break 
        cv2.destroyAllWindows() 
 
'''
Заливка cv2.floodFill

'''     
def floodFill_example():     
        # Заливка 
        # Связность определяется близостью цвета/яркости соседних пикселей.
        img_1 = cv2.imread("/container_data/source/base/500x500.png")
        mask = numpy.zeros((img_1.shape[0]+2, img_1.shape[1]+2,1), dtype=numpy.uint8)
        
        newVal = (0, 255, 255)
        starting_point = (150, 150)
        ret, dst, _mask, rect = cv2.floodFill(img_1.copy(), mask, starting_point, newVal)
       
        newVal = (0, 255, 255)
        starting_point = (250, 250)
        ret, dst2, _mask, rect = cv2.floodFill(img_1.copy(), mask, starting_point, newVal)
         
        # Когда это CV_FLOODFILL_FIXED_RANGE, обрабатываемый пиксель сравнивается с начальной точкой, и пиксель заполняется, если он находится в пределах диапазона.
        # Таким образом, в исходном изображении есть только трехканальные значения пикселей [b-250, g-250, r-250] <= [B, G, R] <= [b + 255, g + 255, r + 255] в этом диапазоне. Будет указано (0, 255, 255) 
        starting_point = (10, 10)
        newVal = (0, 255, 255)
        loDiff = (250, 250, 250)
        upDiff = (255, 255 ,255) 
        ret, dst3, _mask, rect = cv2.floodFill(img_1.copy(), mask, starting_point, newVal, loDiff, upDiff, cv2.FLOODFILL_FIXED_RANGE)
        
        # cv2.FLOODFILL_FIXED_RANGE: обрабатываемый пиксель сравнивается с начальной точкой, и пиксель заполняется в пределах диапазона
        starting_point = (0, 0)
        newVal = (0, 255, 255)
        mask = numpy.ones([img_1.shape[0]+2, img_1.shape[1]+2,1], dtype=numpy.uint8)
        mask[0:200, 0:200] = 0
        ret, dst4, _mask, rect = cv2.floodFill(img_1.copy(), mask, starting_point, newVal, cv2.FLOODFILL_MASK_ONLY)
        
        
        # plot
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGRA2RGB)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGRA2RGB)
        dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGRA2RGB)
        dst3 = cv2.cvtColor(dst3, cv2.COLOR_BGRA2RGB)
        dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGRA2RGB)
        titles = ['Original Image',
                  'floodFill (150, 150)',
                  'floodFill (250, 250)',
                  'floodFill FLOODFILL_FIXED_RANGE',
                  'FLOODFILL_MASK_ONLY'
                  ]
        images = [img_1, dst, dst2, dst3, dst4]
        
        plt.figure(figsize=[15,8]) 
        for i in range(int(len(titles))):
            # subplot(nrows, ncols, index, **kwargs)
            plt.subplot(2,3,i+1),plt.imshow(images[i],cmap='gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.savefig('/container_data/source/base/500x500_floodFill.png')   
        
        plt.show()
        

'''
Вырезать участок
'''           
def crop_example():        
    img = cv2.imread("/container_data/source/pazl/matrix-sunglasses.jpg",cv2.IMREAD_ANYCOLOR) 
    # вырезание учаска ----------------------------------------------------------------
    up = 40 # |
    down = 245 # __
    left =  520  # |
    right =  560  # __
    img_crop = img[up:down,left:right,...] # crop = img[30:130, 150:300,...]
    cv2.imshow('Crop',img_crop)
    
    # Сохраняем изображение на локальный диск ----------------------------------------
    # https://docs.opencv.org/4.5.5/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
    # запишем изображение на диск в формате png
    cv2.imwrite("/container_data/source/pazl/matrix_crop.png", img,[cv2.IMWRITE_PNG_COMPRESSION,1])
    
    key = cv2.waitKey(0) & 0xFF
    if key == 27 or key == ord( "q" ): # exit on ESC break 
        cv2.destroyAllWindows() 
        
'''
Поворот, отражение
'''    
def getRotationMatrix2D_example():  
    img = cv2.imread("/container_data/source/red_car_small.png",cv2.IMREAD_ANYCOLOR)  
    # поворот ------------------------------------------------------------------------
    # получим размеры изображения для поворота
    # и вычислим центр изображения
    (h, w) = img.shape[:2] # h=img.shape[0] w=img.shape[1]
    center = (w/ 2, h / 2)
    # повернем изображение на 250 градусов с коеф.увеличения 1.0
    M = cv2.getRotationMatrix2D(center, 250, 1.0)
    img_1 = cv2.warpAffine(img, M, (w, h))
   
    # или так ROTATE_90_CLOCKWISE,ROTATE_180,ROTATE_90_COUNTERCLOCKWISE 270 градусов
    img_2 = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
   
    # отзеркалить отображение по осям --------------------------------------------------
    # https://docs.opencv.org/4.5.5/d2/de8/group__core__array.html#gaca7be533e3dac7feb70fc60635adf441
    # Ось: 0 – (ось x)по вертикали, 1 – (ось y)по горизонтали, (-1) – по вертикали и по горизонтали.
    img_flip_y = cv2.flip(img,1)
    img_flip_x = cv2.flip(img,0)
    img_flip_xy = cv2.flip(img,-1)

    # plot
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img_flip_y = cv2.cvtColor(img_flip_y, cv2.COLOR_BGRA2RGB)
    img_flip_x = cv2.cvtColor(img_flip_x, cv2.COLOR_BGRA2RGB)
    img_flip_xy = cv2.cvtColor(img_flip_xy, cv2.COLOR_BGRA2RGB)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGRA2RGB)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGRA2RGB)
    titles = [
              'flip ось y',
              'flip ось x',
              'flip обое оси',
              'Original Image',
              'getRotationMatrix2D 250' ,
              'rotate 270']
    images = [img_flip_y,img_flip_x,img_flip_xy,img,img_1,img_2]
    for i in range(int(len(titles))):
        # subplot(nrows, ncols, index, **kwargs)
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.savefig('/container_data/source/rotate.png')    
    plt.show()
        
    key = cv2.waitKey(0) & 0xFF
    if key == 27 or key == ord( "q" ): # exit on ESC break 
        cv2.destroyAllWindows()  
        

'''
Изменение размера

 (h,w,channels)
 (row,col,channels) = img.shape
 
 resize(img,(ширина,высота))
'''
def resize_example():   
    window1 = "dsize INTER_LINEAR_EXACT"
    cv2.namedWindow(window1,flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
    window2 = "dsize INTER_CUBIC"
    cv2.namedWindow(window2,flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
    window3 = "dsize INTER_AREA"
    cv2.namedWindow(window3,flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
    window4 = "коэффициенты масштаба fx, fy"
    cv2.namedWindow(window4,flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
    
    # Изменение размера resize(img,(ширина,высота)) 
    img = cv2.imread("/container_data/source/red_car_small.png",cv2.IMREAD_ANYCOLOR)
    # Картинки и видео аналогичным образом
    
    # сохранить соотношение сторон, для этого считаем коэф. уменьшения стороны ---------
    # img_resized = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
    final_wide = img.shape[1]*2 # увеличить в 2 раза
    r = float(final_wide) / img.shape[1]
    dim = (final_wide, int(img.shape[0] * r)) 
  
    # INTER_LINEAR_EXACT хуже чем INTER_CUBIC
    img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR_EXACT)
    cv2.imshow(window1,img_resized)

    # самый плохой вариант INTER_AREA
    img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow(window3,img_resized)
      
    # лучший вариант INTER_CUBIC
    img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    cv2.imshow(window2,img_resized)
      
    # изменение размера через коэффициенты масштаба fx, fy -----------------------------
    # лучше чем INTER_LINEAR_EXACT
    resized_fx_fy = cv2.resize(img,None,fx=2, fy=2)
    cv2.imshow(window4,resized_fx_fy) 
    rgb_img = cv2.cvtColor(resized_fx_fy, cv2.COLOR_BGRA2RGB)
     
    key = cv2.waitKey(0) & 0xFF
    if key == 27 or key == ord( "q" ): # exit on ESC break 
        cv2.destroyAllWindows()   

'''
https://habr.com/ru/post/678570/
''' 
def matrix_operations():
    try:
        row = 5
        col = 10
        img = numpy.zeros((row,col,3),numpy.uint8) # 3 - bgr 

        # (h,w,channels)
        (row,col,channels) = img.shape
        # Доступ по индексу, первый [0], последний для [row-1] для [col-1]
        img1 = img.copy()
        img1[0,0] = 255 # верхний левый угол
        img1[row-1,0] = 255 # верхний правый угол
        img1[0,col-1] = 255 # нижний правый угол
        img1[row-1,col-1] = [255, 255, 255] # или 255 нижний левый угол
        
        img2 = img.copy()
        # img2[:] = [0, 0, 255] - закрасить всю матрицу
        img2[4,:] = [0, 0, 255] # или  img[4] т.е. в 4-й ряду по всем колонкам проставить красный цвет
        img2[:,0] = [0, 0, 255] # т.е. в 1-й колонке по всем рядам проставить красный
        
        # Доступ по диапазону,от первого [0:], до последнего [:row] или [:], до 3-го не включая(т.е 0,1,2) [:3] или [0:3]
        img3 = img2.copy()
        up = row-2 # от
        down = row # до 
        left = 0
        right = col
        img3 = img3[up:down,left:right,...]
        
        # Постоить индексы цвета(т.е. одного из 3-х каналов BGR пикселя) пикселя и задать им значение на основе выражения
        img4 = img.copy()
        # построить индексы на основе выражения (если пиксель в img2 соответствует выражению `current_pixel >= 250` True иначе False)
        indexes = img2 >= 250 # любой цвет соответствующий выражению
        # по всем индексам(цветам) True присвоить значение 75
        img4[indexes] = 75
        
      
        # plot
        # Для отображения plot преобразуем цвета BGR to RGB
        ''' 
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
             или
            img = img[:, :, [2, 1, 0]]
             или
            # (::-1) — взять каждый элемент, но в обратном порядке.
            img = img[:, :, ::-1]
        '''
        img = img[:, :, ::-1]
        img1 = img1[:, :, ::-1]
        img2 = img2[:, :, ::-1]
        img3 = img3[:, :, ::-1]
        img4 = img4[:, :, ::-1]
        
        plt.figure(figsize=[15,8])
        plt.subplot(2,3,1);plt.imshow(img);plt.title(f'Original Image row:{row} col:{col}');plt.xticks([0,1,2,3,4,5,6,7,8,9]);plt.yticks([0,1,2,3,4]);
        plt.subplot(2,3,2);plt.imshow(img1);plt.title('Углы [0,0],[w-1,0],[0,h-1],[w-1,h-1]');plt.xticks([0,1,2,3,4,5,6,7,8,9]);plt.yticks([0,1,2,3,4]);
        plt.subplot(2,3,4);plt.imshow(img2);plt.title('Ряд [4,:] и Колонка [:,0]');plt.xticks([0,1,2,3,4,5,6,7,8,9]);plt.yticks([0,1,2,3,4]);
        plt.subplot(2,3,5);plt.imshow(img3);plt.title(f'Crop [{up}:{down},{left}:{right},...]');plt.xticks([0,1,2,3,4,5,6,7,8,9]);plt.yticks([0,1]);
        plt.subplot(2,3,6);plt.imshow(img4);plt.title('[img >= 250] = 75');plt.xticks([0,1,2,3,4,5,6,7,8,9]);plt.yticks([0,1,2,3,4]);
        plt.savefig('/container_data/source/base/matrix_operations.png')  
        plt.show()
     
    except Exception as e: 
        print(e)
        cv2.destroyAllWindows()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)       
# ==========================================================================================================  
'''
Применение маски
https://www.youtube.com/watch?v=oXlwWbU8l2o&list=PLuudOZcE9EgnCN8cgw9FcPeQnjP1axXP_&index=4&t=176s&ab_channel=freeCodeCamp.org
'''
def bitwise_operations():
    img_circle_1 = numpy.zeros((50,50,3),numpy.uint8)             
    img_circle_1[:] = [0,0,0]        
    cv2.circle(img_circle_1,(0,0),25,[255,255,255],thickness=cv2.FILLED)# заливка        
    
    img_circle_2 = numpy.zeros((50,50,3),numpy.uint8)             
    img_circle_2[:] = [0,0,0]          
    cv2.circle(img_circle_2,(img_circle_2.shape[1]//2,img_circle_2.shape[0]//2),25,[255,255,255],thickness=cv2.FILLED)# заливка        
      
    img_and = cv2.bitwise_and(img_circle_1,img_circle_2)  
    img_or = cv2.bitwise_or(img_circle_1,img_circle_2)   
    img_xor = cv2.bitwise_xor(img_circle_1,img_circle_2)     
    img_not1 = cv2.bitwise_not(img_circle_1)
    img_not2 = cv2.bitwise_not(img_circle_2)
    
    # Наложение маски
    img = cv2.imread("/container_data/source/red_car_small.png",cv2.IMREAD_ANYCOLOR)

    blank = numpy.zeros(img.shape[:2],numpy.uint8)
    mask = cv2.circle(blank.copy(),(img.shape[1]//2,img.shape[0]//2),img.shape[0]//2,[255,255,255],thickness=cv2.FILLED)
    img_masked = cv2.bitwise_and(img,img,mask=mask) # нахождение пересекающейся области т.е. белый круг будет заменен картинкой
    
    # Наложение сложной маски
    blank = numpy.zeros(img.shape[:2],numpy.uint8)
    mask_circle = cv2.circle(blank.copy(),(img.shape[1]//2,img.shape[0]//2),img.shape[0]//3,[255,255,255],thickness=cv2.FILLED)
    mask_rectangle = cv2.rectangle(blank.copy(),(25,25),(img.shape[1]//2,img.shape[0]//2),[255,255,255],thickness=cv2.FILLED)# заливка
    mask = cv2.bitwise_and(mask_circle,mask_rectangle)
    img_masked2 = cv2.bitwise_and(img,img,mask=mask) 
    
    # plot
    img_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGRA2RGB)
    img_masked2 = cv2.cvtColor(img_masked2, cv2.COLOR_BGRA2RGB)

    titles = ['circle 1',
              'circle 2', 
              '1 and 2',
              '1 or 2',
              '1 xor 2',
              'not 1', 
              'not 2',
              'наложение маски',
              'сложная маска']

    images = [img_circle_1,
              img_circle_2,  
              img_and,
              img_or,
              img_xor,
              img_not1,
              img_not2,
              img_masked,
              img_masked2]
    plt.figure(figsize=[15,8])
    for i in range(int(len(images))):
        # subplot(nrows, ncols, index, **kwargs)
        plt.subplot(3,4,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.savefig('/container_data/source/bitwise_result.png')    
    plt.show()        
            
# ==========================================================================================================  
'''
Гистограммы позволяют визуализировать распределение интенсивности пикселей
'''
def histogram():

    img = cv2.imread('/container_data/source/red_car_small.png')
    
    '''
    Пики показывают количество пикселей(ось y) с определенной интенсивностью(ось x)
    Если пик ближе к интенсивности 255 тогда картинка более светлая чем пик ближе к 0-ю
    Расчитываем гистограмму для конкретной области т.е. используем маску
    '''
    # GRayscale histogram --------------------------------------------------------
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    gray_hist = cv2.calcHist([gray], [0], None, [256], [0,256] )
    plt.figure()
    plt.title('Grayscale Histogram mask')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    plt.plot(gray_hist)
    plt.xlim([0,256])# ограничение
    plt.savefig('/container_data/source/red_car_small_histogram_gray.png')
    plt.show()
    
    # use mask
    blank = numpy.zeros(img.shape[:2], dtype='uint8')
    mask = cv2.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)
    masked = cv2.bitwise_and(gray,gray,mask=mask)
    cv2.imshow('Mask gray', masked)
    
    gray_hist = cv2.calcHist([gray], [0], masked, [256], [0,256] )
    plt.figure()
    plt.title('Grayscale Histogram mask')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    plt.plot(gray_hist)
    plt.xlim([0,256])# ограничение
    plt.savefig('/container_data/source/red_car_small_histogram_mask_gray.png')
    plt.show()

    # Colour Histogram -----------------------------------------------------------
    
    plt.figure()
    plt.title('Colour Histogram full image')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    colors = ('b', 'g', 'r')
    for i,col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist, color=col)
        plt.xlim([0,256])
    plt.savefig('/container_data/source/red_car_small_histogram_full_bgr.png')
    plt.show()
    
    # use mask
    blank = numpy.zeros(img.shape[:2], dtype='uint8')
    masked = cv2.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)
  
    plt.figure()
    plt.title('Colour Histogram for mask')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    colors = ('b', 'g', 'r')
    for i,col in enumerate(colors):
        hist = cv2.calcHist([img], [i], masked, [256], [0,256])
        plt.plot(hist, color=col)
        plt.xlim([0,256])
    plt.savefig('/container_data/source/red_car_small_histogram_mask_bgr.png')
    plt.show()

    # Корректировка канала
    b,g,r = cv2.split(img)
    r = r+60  # усилить красный
    img_strong_red = cv2.merge((b,g,r))
    plt.figure()
    plt.title('Colour Histogram full image strong red')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    colors = ('b', 'g', 'r')
    for i,col in enumerate(colors):
        hist = cv2.calcHist([img_strong_red], [i], masked, [256], [0,256])
        plt.plot(hist, color=col)
        plt.xlim([0,256])
    plt.savefig('/container_data/source/red_car_small_histogram_full_strong_red.png')
    plt.show()
    
    # Общая картина ----------------------------------------------------------------- 
    img_original = cv2.imread('/container_data/source/red_car_small_histogram_full_bgr.png')
    img_strong_red_hist = cv2.imread('/container_data/source/red_car_small_histogram_full_strong_red.png')
    img_gray_mask = cv2.imread('/container_data/source/red_car_small_histogram_mask_gray.png')
    img_gray = cv2.imread('/container_data/source/red_car_small_histogram_gray.png')
    
    img_strong_red_hist = cv2.cvtColor(img_strong_red_hist, cv2.COLOR_BGRA2RGB)
    img_strong_red = cv2.cvtColor(img_strong_red, cv2.COLOR_BGRA2RGB)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGRA2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
      
    titles = ['Original hist', 'Strong red','Strong red image',  'Gray','Gray mask' ,'Original image']
    images = [img_original, img_strong_red_hist, img_strong_red,   img_gray, img_gray_mask, img ]
    plt.figure(figsize=[15,8])
    for i in range(int(len(images))):
        # subplot(nrows, ncols, index, **kwargs)
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.savefig('/container_data/source/red_car_small_histogram_total.png')    
    plt.show()       
     
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
        
# ==========================================================================================================  
             
            
    # ДОБАВИТЬ ПРИМЕР ОПЕРАЦИЙ С КАНАЛАМИ !!!!!!!!
    #cv2.split()  —  разделит многоканальный массив на несколько одноканальных.
    #cv2.merge()  —  объединит массивы в один многоканальный. Массивы должны быть одинакового размера.

    # https://docs.opencv.org/4.5.5/d2/de8/group__core__array.html#ga51d768c270a1cdd3497255017c4504be        
    # bitwise_and bitwise_no cv::mixChannels
    #b,g,r = cv2.split(img_5_bgr) # Делит многоканальный массив на несколько одноканальных массивов.
    #cv2.imshow('Red Channel',r)
    #cv2.imshow('Green Channel',g)
    #cv2.imshow('Blue Channel',b)
    #img_5 = cv2.bitwise_not(g)
    
    # Добавить выравнивание гистограммы gray
    # img = cv2.equalizeHist(img_gray)
    # cv2.imshow("equalizeHist",img)
    
    # Добавить
    # https://russianblogs.com/article/9615742843/
    #  функции преобразования:cv2.warpAffine с участием cv2.warpPerspective, cv2.resize(), cv2.getAffineTransform, cv2.getPerspectiveTransform
    # res = cv2.resize(img, None, fx=2, fy=3, interpolation=cv2.INTER_LANCZOS4)    
# ==========================================================================================================  
   
try:   

    if False: 
       base_example()
    if False:
       floodFill_example()    
    if False:
       resize_example()  
    if False:
       getRotationMatrix2D_example() 
    if False:
       matrix_operations()     
    if True:
       bitwise_operations()       
    if False:
       crop_example()     
    if False:
       histogram()        
except Exception as e:
    print(e)
    cv2.destroyAllWindows() 
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)       
    
# Run:    
# python  /container_data/image_example/base.py   









