#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import sys  
from platform import python_version
import matplotlib.pyplot as plt

print('OpenCV',cv2.__version__)  
print('Python',python_version())
 
def main(argv):
    '''
    https://docs.opencv.org/4.5.5/d7/dfc/group__highgui.html#ga5afdf8410934fd099df85c75b2e0888b

    Альфа канал
    cv2.IMREAD_UNCHANGED = -1 - вернуть загруженное изображение как есть (с альфа-каналом, иначе оно будет обрезано)
    cv2.IMREAD_GRAYSCALE = 0 - преобразовывать изображение в одноканальное изображение в градациях серого
    cv2.IMREAD_COLOR = 1 -  преобразовывать изображение в 3-канальное цветное изображение BGR
    cv2.IMREAD_ANYCOLOR = 4 - изображение читается в любом возможном цветовом формате.

    cv2.WINDOW_NORMAL - позволяет изменять размер окна 
    cv2.WINDOW_AUTOSIZE - автоматически настраивает размер окна в соответствии с отображаемым изображением
    cv2.WINDOW_GUI_EXPANDED - улучшенный графический интерфейс
    cv2.WINDOW_KEEPRATIO - сохраняет пропорции изображения
    По умолчанию cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED
    '''
    
    path_dir = argv[0] if len(argv) > 0 else '/container_data/source'
    
    title = "IMREAD_UNCHANGED"
    cv2.namedWindow(title,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
    img = cv2.imread(path_dir+"/list.jpeg",cv2.IMREAD_UNCHANGED)
    cv2.imshow(title,img)

    title = "IMREAD_GRAYSCALE"
    cv2.namedWindow(title,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
    img = cv2.imread(path_dir+"/list.jpeg",cv2.IMREAD_GRAYSCALE)
    cv2.imshow(title,img)

    title = "IMREAD_COLOR"
    cv2.namedWindow(title,cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) 
    img = cv2.imread(path_dir+"/list.jpeg",cv2.IMREAD_COLOR)
    cv2.imshow(title,img)
    
    
    # plot --------------------------------------------------
    #  Двумерный массив пикселей:
    smile = [[0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]]
    fig, ax = plt.subplots()
    ax.imshow(smile)
    fig.set_figwidth(6)  #  ширина и
    fig.set_figheight(6) #  высота "Figure"
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
    
# python  /container_data/image_example/show_picture.py  "/container_data/source"  