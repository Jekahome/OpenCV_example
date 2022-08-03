#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy
import sys, os
from matplotlib import pyplot as plt
from platform import python_version

print('OpenCV',cv2.__version__)  
print('Python',python_version())

'''
Draw line
'''
def draw_line():
   
    img = numpy.zeros((15,30,3),dtype=numpy.uint8)
    (row,col,channels) = img.shape
    
    img[:] = 255  
   
    top_w = 0
    top_h = 0
    down_w = 30
    down_h = 30
    blue = 0
    green = 255
    red = 255
    thickness = 1
     
    cv2.line(img,(top_w,top_h),(down_w,down_h),(blue,green,red),thickness)
    cv2.line(img,(0,row//2),(col,row//2),(255,0,0),thickness)# горизонтальная по центру
    
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    titles = ['draw line']
    images = [img]
    plt.figure(figsize=[15,8])
    for i in range(int(len(images))):
        # subplot(nrows, ncols, index, **kwargs)
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.savefig('/container_data/source/draw_line.png')    
    plt.show()
    
    key = cv2.waitKey(0) & 0xFF
    if key == 27: # exit on ESC break 
        cv2.destroyAllWindows()   
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
'''
Draw rectangle
'''
def draw_rectangle():
    # https://docs.opencv.org/4.5.5/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9
   
    img = numpy.zeros((15,30,3),dtype=numpy.uint8)
    img[:] = 255
   
    top_w = 5
    top_h = 0
    down_w = 8
    down_h = 12
    color = [0,50,255]
    thickness = 1
   
    # rectangle мутирует img
    cv2.rectangle(img,(top_w,top_h),(down_w,down_h),color,thickness, cv2.LINE_8)# контур
    cv2.rectangle(img,(12,2),(20,13),(40,40,40),thickness=cv2.FILLED)# заливка
    
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    titles = ['draw rectangle']
    images = [img]
    plt.figure(figsize=[15,8])
    for i in range(int(len(images))):
        # subplot(nrows, ncols, index, **kwargs)
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.savefig('/container_data/source/draw_rectangle.png')    
    plt.show()
    
    key = cv2.waitKey(0) & 0xFF
    if key == 27: # exit on ESC break 
        cv2.destroyAllWindows()   
# ---------------------------------------------------------------------------------------------------------------------

'''
Draw circle 
'''
def draw_circle():
    # https://docs.opencv.org/4.5.5/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
    img = numpy.zeros((15,30,3),dtype=numpy.uint8)
    img_2 = img.copy()
    (row,col,channels) = img.shape
    img[:] = 255 
    img_2[:] = 255 
    
    centr_y = col//2
    centr_x = row//2
    radius = 5
    color = [0,50,255]
    thickness = 1
 
    # circle мутирует img (можно передеать копию img.copy() и забрать из результата его работы преобразованное изображение)
    img = cv2.circle(img.copy(),(centr_y,centr_x),radius,color,thickness, cv2.LINE_8)# контур
 
    cv2.circle(img_2,(centr_y,centr_x),radius,color,thickness=cv2.FILLED)# заливка
    
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGRA2RGB)
    titles = ['draw circle','draw circle fill']
    images = [img,img_2]
    plt.figure(figsize=[15,8])
    for i in range(int(len(images))):
        # subplot(nrows, ncols, index, **kwargs)
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i]);plt.xticks([]),plt.yticks([])
    plt.savefig('/container_data/source/draw_circle.png')    
    plt.show()
    
    
    key = cv2.waitKey(0) & 0xFF
    if key == 27: # exit on ESC break 
        cv2.destroyAllWindows()
# ---------------------------------------------------------------------------------------------------------------------
'''
Draw text
'''
def draw_text():
    # https://docs.opencv.org/4.5.5/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
    window = "draw text"
    cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) # Create window handler
    img = numpy.zeros((350,580,3),dtype=numpy.uint8)
    img[:] = 0   
    
    text = "Hello"
    start_y = 0
    start_x = 80
    fontFace = cv2.FONT_HERSHEY_SIMPLEX # 0-16 https://docs.opencv.org/4.5.5/d6/d6e/group__imgproc__draw.html#ga0f9314ea6e35f99bb23f29567fc16e11
    fontScale = 3 # size font
    color = [0,255,255]
    thickness = 1 # width line
 
    cv2.putText(img,"cv2.LINE_AA",(start_y,start_x),fontFace,fontScale,color,thickness,cv2.LINE_AA) # cv2.LINE_AA самый сглаженый шрифт
    cv2.putText(img,"cv2.LINE_4",(start_y,start_x+80),fontFace,fontScale,color,thickness,cv2.LINE_4)
    cv2.putText(img,"cv2.LINE_8",(start_y,start_x+160),fontFace,fontScale,color,thickness,cv2.LINE_8)
    cv2.putText(img,"cv2.FILLED",(start_y,start_x+240),fontFace,fontScale,color,thickness,cv2.FILLED) # cv2.FILLED самый резкий шрифт
    cv2.imwrite("/container_data/source/draw_text.png", img,[cv2.IMWRITE_PNG_COMPRESSION,1]) 
    cv2.imshow(window,img)
    key = cv2.waitKey(0) & 0xFF
    if key == 27: # exit on ESC break 
        cv2.destroyAllWindows() # закрыть все окна или каждое cv2.destroyWindow(title)

# ---------------------------------------------------------------------------------------------------------------------
(bgr,thickness,img,text,window,fontScale)=(None,None,None,None,None,None)

def event_choice_color_blue(pos):
    global bgr,thickness,img,text,window,fontScale
    bgr = tuple([pos,bgr[1],bgr[2]])
    cv2.putText(img,text,(0,80),cv2.FONT_HERSHEY_SIMPLEX,fontScale,bgr,thickness,cv2.LINE_AA)
    cv2.imshow(window,img)

def event_choice_color_green(pos):
    global bgr,thickness,img,text,window,fontScale
    bgr = tuple([bgr[0],pos,bgr[2]])
    cv2.putText(img,text,(0,80),cv2.FONT_HERSHEY_SIMPLEX,fontScale,bgr,thickness,cv2.LINE_AA)
    cv2.imshow(window,img) 

def event_choice_color_red(pos):
    global bgr,thickness,img,text,window,fontScale
    bgr = tuple([bgr[0],bgr[1],pos])
    cv2.putText(img,text,(0,80),cv2.FONT_HERSHEY_SIMPLEX,fontScale,bgr,thickness,cv2.LINE_AA)
    cv2.imshow(window,img) 

def event_choice_thickness(pos):
    global bgr,thickness,img,text,window,fontScale
    thickness = pos  
    cv2.putText(img,text,(0,80),cv2.FONT_HERSHEY_SIMPLEX,fontScale,bgr,thickness,cv2.LINE_AA)
    cv2.imshow(window,img)  

'''
Draw trackbar 
'''
def draw_trackbar():
    global bgr,thickness,img,text,window,fontScale
    # https://docs.opencv.org/4.5.5/d7/dfc/group__highgui.html#gaf78d2155d30b728fc413803745b67a9b
    window = "draw trackbar"
    cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) # Create window handler
    img = cv2.imread(cv2.samples.findFile("/container_data/source/list.jpeg"),cv2.IMREAD_UNCHANGED)
    if img is None:
        print('Could not open or find the image: ', "list.jpeg")
        exit(0) 
    
    bgr = tuple([0,0,0])  
    thickness = 1 # width line
    text = "cv2.LINE_AA"
    fontScale = 1 # size font
    cv2.putText(img,text,(0,80),cv2.FONT_HERSHEY_SIMPLEX,fontScale,bgr,thickness,cv2.LINE_AA)
    cv2.imshow(window,img)
    min_value = 0
    max_value = 255
            
    try:  
        trackbar_1 = "choice color blue"
        cv2.createTrackbar(trackbar_1,window,min_value,max_value,event_choice_color_blue)
        trackbar_2 = "choice color green"
        cv2.createTrackbar(trackbar_2,window,min_value,max_value,event_choice_color_green)
        trackbar_3 = "choice color red"
        cv2.createTrackbar(trackbar_3,window,min_value,max_value,event_choice_color_red)
        trackbar_4 = "choice thickness"
        cv2.createTrackbar(trackbar_4,window,min_value,50,event_choice_thickness)    
        #pos = cv2.getTrackbarPos(trackbar,window)
        cv2.imshow(window,img)

        key = cv2.waitKey(0) & 0xFF
        if key == 27: # exit on ESC break 
            cv2.destroyAllWindows() # закрыть все окна или каждое cv2.destroyWindow(title)

    except Exception as e:
        print(e)
        cv2.destroyAllWindows() 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)  

# ---------------------------------------------------------------------------------------------------------------------
 


def main(params):
    if False:
        draw_trackbar()
    if False:
        draw_rectangle()        
    if False:
        draw_line()    
    if False:
        draw_circle()    
    if True:
        draw_text()    
if __name__ == "__main__":
    main(sys.argv[1:])
            
# Run:    
# python  /container_data/image_example/draw.py    