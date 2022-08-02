# Как сканировать изображения, таблицы поиска
# https://docs.opencv.org/4.5.5/db/da5/tutorial_how_to_scan_images.html

import numpy as np
import cv2
from matplotlib import pyplot as plt
from platform import python_version

print('OpenCV',cv2.__version__)  
print('Python',python_version())
 
'''
Поскольку при обработке изображений часто выполняется преобразование каждого пикселя. 
Этот процесс преобразования может включать относительно большой объем вычислений. 
В настоящее время вы можете использовать предварительно рассчитанные данные преобразования 
вместо того, чтобы рассчитывать каждый пиксель один раз, 
что может значительно сократить количество вычислений

LUT указывает на таблицу поиска (таблица поиска также может быть одним каналом 
или 3 каналами, если входное изображение представляет собой один канал, 
таблица поиска должна быть одним каналом

Операции с маской над матрицами cv2.filter2D
Идея состоит в том, что мы пересчитываем значение каждого пикселя изображения в соответствии с матрицей маски (также известной как ядро). 
Эта маска содержит значения, которые регулируют степень влияния соседних пикселей (и текущего пикселя) на новое значение пикселя.

'''
 
 
# Гамма преобразование
def gamma_transform(image, gamma=1.0):
     
    table_search = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
     
    return cv2.LUT(image, table_search) # Использовать функцию таблицы поиска OpenCV

 
def test(image):
    # обнуления красного канала
    identity = np.arange(256, dtype = np.dtype('uint8'))
    zeros = np.zeros(256, np.dtype('uint8'))
    table_search = np.dstack((identity, identity, zeros))
    table_search[:,:,2] = 0
    
    # Установите синие значения больше, чем b_max, на b_max:
    # b_max = 20
    # img[img[:,:,0] > b_max, 0] = b_max
    return cv2.LUT(image, table_search)
# -----------------------------------------------------------
img = cv2.imread('/container_data/source/grey_books.jpg')
 
# gamma = gamma_transform(img, 0.5)
gamma = test(img)

cv2.imshow('img', img)
cv2.imshow('gamma', gamma)
 
# applyColorMap использует 13 таблиц поиска (cv2.COLORMAP_JET,...) для LUT 
# https://learnopencv.com/applycolormap-for-pseudocoloring-in-opencv-c-python/
color_image = cv2.applyColorMap(img, cv2.COLORMAP_JET)# Псевдоцвет
cv2.imshow('applyColorMap COLORMAP_JET', color_image) 
color_image = cv2.applyColorMap(img, cv2.COLORMAP_COOL)
cv2.imshow('applyColorMap COLORMAP_COOL', color_image) 
color_image = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
cv2.imshow('applyColorMap COLORMAP_RAINBOW', color_image) 

# Операции с маской над матрицами ---------------------------------------------------------------------
img = cv2.imread('/container_data/source/600px-Lenna.png')
kernal = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) # ядро для повышения контрастности изображения
filter2D = cv2.filter2D(img, -1, kernal)
cv2.imshow('filter2D', filter2D)   
cv2.imshow('Original', img)  
    
# Повышение яркости   ---------------------------------------------------------------------------------  
alpha = float(1.0)# 1.0-3.0
beta = int(100)  # 0-100  
new_image_brightness = np.zeros(img.shape, img.dtype)
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        for c in range(img.shape[2]):
            new_image_brightness[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)
cv2.imshow('new_image_brightness', new_image_brightness) 

    
cv2.waitKey(0)
cv2.destroyAllWindows()
