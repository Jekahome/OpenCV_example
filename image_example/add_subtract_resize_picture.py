#!/usr/bin/python
# -*- coding: utf-8 -*-
from statistics import variance
import cv2
import numpy
import sys, os
import copy

from matplotlib import pyplot as plt # apt-get install -y python3-pip python3-tk&&  python -mpip install -U matplotlib && pip install pyqt5

    
'''
Сложение, вычитание, наложение изображений

cv2.resize      - изменить размер изображения
cv2.add         - сумировать изображения
cv2.subtract    - вычесть изображение
cv2.addWeighted - наложение изображений

При add сложении 250 + 10 = итоговое значение будет 255 (белый) т.е. не больше максимального
И получается что все области с переполнением сложения будут белыми
При subtract вычитании результат не меньше 0
'''
def main(argv):
    try:
        # https://docs.opencv.org/4.5.5/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
        # https://docs.opencv.org/4.5.5/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19
        # path = os.path.join("container_data", "source", "car.jpg")
        # print('[DEBUG] path:', path) # container_data/source/car.jpg

        path_img_1 = argv[0] if len(argv) > 0 else '/container_data/source/600px-Lenna.png'
        path_img_2 = argv[1] if len(argv) > 0 else '/container_data/source/600px-Lenna.png'

        img_1 = cv2.imread(path_img_1,cv2.IMREAD_UNCHANGED)
        if img_1 is None:
            print('Wrong path:', path_img_1)
            exit(0)
        else:
            img_1 = cv2.resize(img_1,(700,700))
            
        img_2 = cv2.imread(path_img_2,cv2.IMREAD_UNCHANGED)
        if img_1 is None:
            print('Wrong path:', path_img_2)
            exit(0)
        else:
            img_2 = cv2.resize(img_2,(700,700))

        # сумировать изображения
        cv2.namedWindow("add",cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
        result_img = cv2.add(img_1,img_2)
        cv2.imshow("add",result_img)
        
        # вычесть изображение
        cv2.namedWindow("subtract img_1 - img_2",cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
        result_img = cv2.subtract(img_1,img_2)
        cv2.imshow("subtract img_1 - img_2",result_img)
        
        # вычесть изображение
        cv2.namedWindow("subtract img_2 - img_1",cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
        result_img = cv2.subtract(img_2,img_1)
        cv2.imshow("subtract img_2 - img_1",result_img)
        
        # наложение на img_2
        cv2.namedWindow("addWeighed",cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
        beta_weight_img_1 = 0.5
        alpha_weight_img_2 = 1
        gamma = 0
        result_img = cv2.addWeighted(img_2,alpha_weight_img_2,img_1,beta_weight_img_1,gamma)
        cv2.imshow("addWeighed",result_img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord( "q" ): # exit on ESC break 
           cv2.destroyAllWindows()
        
    except Exception as e: 
        print(e)
        cv2.destroyAllWindows()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

if __name__ == "__main__":
    main(sys.argv[1:])
   
# Run:    
# python  /container_data/image_example/add_subtract_resize_picture.py   "/container_data/source/car.jpg" "/container_data/source/objects.png"