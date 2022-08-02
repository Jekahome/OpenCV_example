#!/usr/bin/python
# -*- coding: utf-8 -*-
from statistics import variance
import cv2
import numpy
import sys, os
import copy

from matplotlib import pyplot as plt # apt-get install -y python3-pip python3-tk&&  python -mpip install -U matplotlib && pip install pyqt5

  
        
def up_h(pos):
    global img,window,up,down,left,right
    up = pos
    crop = img[up:down,left:right,...] 
    cv2.imshow(window,crop)
    
def down_h(pos):
    global img,window,up,down,left,right
    down = pos
    crop = img[up:down,left:right,...] 
    cv2.imshow(window,crop) 
    
def left_h(pos):
    global img,window,up,down,left,right
    left = pos
    crop = img[up:down,left:right,...] 
    cv2.imshow(window,crop)
    
def right_h(pos):
    global img,window,up,down,left,right
    right = pos
    crop = img[up:down,left:right,...] 
    cv2.imshow(window,crop)
    
try:   
 
    # Debug crop img
    # y |
    # x __
    img = cv2.imread("/container_data/source/pazl/matrix-sunglasses.jpg",cv2.IMREAD_ANYCOLOR)
    #img.shape  (667, 1600, 3)  667 строк, 1600 столбцов и 3 канала цвета
    # (450, 450, 3)
    window = "crop"
    cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
    up=0
    down=img.shape[0]
    left=0
    right=img.shape[1]
    cv2.createTrackbar("up",window,0,img.shape[0],up_h)
    cv2.createTrackbar("down",window,0,img.shape[0],down_h)
    cv2.createTrackbar("left",window,0,img.shape[1],left_h)    
    cv2.createTrackbar("right",window,0,img.shape[1],right_h)
    cv2.setTrackbarPos("up",window,up)
    cv2.setTrackbarPos("down",window,down)
    cv2.setTrackbarPos("left",window,left)
    cv2.setTrackbarPos("right",window,right)
    cv2.imshow(window,img)
    
    print(">>> Enter ESC <<<")
    key = cv2.waitKey(0) & 0xFF
    if key == 27: # exit on ESC break 
        cv2.destroyAllWindows() # закрыть все окна или каждое cv2.destroyWindow(title)

except Exception as e:
    print(e)
    cv2.destroyAllWindows() 
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
    
    
# Run:    
# python  /container_data/image_example/crop.py      