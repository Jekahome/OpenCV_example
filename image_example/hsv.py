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
 
'''
HSV

Канал H обозначает цвет. В зависимости от числа меняется оттенок. 
Канал S – насыщенность, при минимальном значении это белый цвет, при максимальном – цвет, соответствующий значению канала H. 
Канал V – это яркость. Минимальное значение – черный, максимальное – цвет, соответствующий комбинации H и S

https://habr.com/ru/post/664984/
'''
def main(params):
    img_original_1 = cv2.imread('/container_data/source/stop_summer.png',cv2.IMREAD_ANYCOLOR)
    img_original_2 = cv2.imread('/container_data/source/stop_winner.png',cv2.IMREAD_ANYCOLOR)
    
    # BGR -----------------------------------------------------------------------------------
    red_channel = img_original_1[:,:,2]
    bin_img_bgr_1 = numpy.zeros(img_original_1.shape)
    bin_img_bgr_1[red_channel > 200] = [0, 0, 255]
    # не сможет выделить цвет из-за формата, сразу три компоненты отвечают за цвет
    red_channel = img_original_2[:,:,2]
    bin_img_bgr_2 = numpy.zeros(img_original_2.shape)
    bin_img_bgr_2[red_channel > 200] = [0, 0, 255]
 
    # HSV -----------------------------------------------------------------------------------
    img_hsv = cv2.cvtColor(img_original_1, cv2.COLOR_BGR2HSV)
    #h_channel = img_hsv[:,:,0]
    #v_channel = img_hsv[:,:,2]
    h_channel,_s,v_channel = cv2.split(img_hsv)
    bin_img_hsv_1 = numpy.zeros(img_hsv.shape)
    bin_img_hsv_1[(h_channel < 10) * (h_channel > 0) * (v_channel>150) ] = [0, 0, 255]
    # HSV сможет выделить именно нужный цвет из-за влияния на цвет определенной компонентой без помех остальных компонент 
    img_hsv = cv2.cvtColor(img_original_2, cv2.COLOR_BGR2HSV)
    h_channel = img_hsv[:,:,0]
    v_channel = img_hsv[:,:,2]
    bin_img_hsv_2 = numpy.zeros(img_hsv.shape)
    bin_img_hsv_2[(h_channel < 10) * (h_channel > 1) * (v_channel>110) ] = [0, 0, 255]
    kernel = numpy.ones((3,3),numpy.uint8)
    
    # Для отображения plot преобразуем цвета BGR to RGB
    '''
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
            или
        img = img[:, :, [2, 1, 0]]
            или
        # (::-1) — взять каждый элемент, но в обратном порядке.
        img = img[:, :, ::-1]
    '''
    img_original_1 = img_original_1[:, :, ::-1]
    bin_img_bgr_1 = bin_img_bgr_1[:, :, ::-1]
    bin_img_hsv_1 = bin_img_hsv_1[:, :, ::-1]
    img_original_2 = img_original_2[:, :, ::-1]
    bin_img_bgr_2 = bin_img_bgr_2[:, :, ::-1]
    bin_img_hsv_2 = bin_img_hsv_2[:, :, ::-1]

    # plot
    titles = ['Original Image',
                'BGR',
                "HSV",
                'Original Image',
                'BGR',
                "HSV"
                ]
    images = [img_original_1, 
                bin_img_bgr_1,
                bin_img_hsv_1,
                img_original_2, 
                bin_img_bgr_2,
                bin_img_hsv_2
                ]
    for i in range(int(len(titles))): 
        # subplot(nrows, ncols, index, **kwargs)
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.savefig('/container_data/source/stop_detect_result.png') 
    plt.show() 
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main(sys.argv[1:])   
     
# Run:    
# python  /container_data/image_example/hsv.py   