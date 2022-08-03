#!/usr/bin/python
# -*- coding: utf-8 -*-
 
import sys, os
import cv2 
import numpy
import socket

def server():
    tcp_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    host = socket.gethostname()# 'localhost'
    port = 1234
    addr = (host,port)

    tcp_socket.bind(addr)#bind - связывает адрес и порт с сокетом
    tcp_socket.listen(5)#listen - запускает прием TCP

    while True:
        #Если мы захотели выйти из программы
        question = input('Do you want to quit? y\\n: ')
        if question == 'y': break
        
        #accept - принимает запрос и устанавливает соединение, (по умолчанию работает в блокирующем режиме)
        #устанавливает новый сокет соединения в переменную conn и адрес клиента в переменную addr
        conn,addr = tcp_socket.accept()
        print(f"Connection from {addr} has been established!")
        
        data = conn.recv(1024) #recv - получает сообщение TCP
        if data != None:
            print(data)
        
        conn.send(bytes("Welcom to the server!","utf-8"))#send - передает сообщение TCP
        conn.close()#close - закрывает сокет
        
    tcp_socket.close()    
# ------------------------------------------------------------------
def client():
    background = cv2.imread("/container_data/source/background.jpg",cv2.IMREAD_GRAYSCALE)
    
    
    tcp_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    host = socket.gethostname()# 'localhost'
    port = 1234
    addr = (host,port)
    tcp_socket.connect(addr)

    tcp_socket.send(str.encode("hello server","utf-8"))

    full_msg = ''
    while True:
        msg = tcp_socket.recv(8)
        if len(msg) <= 0:
            break
        full_msg += msg.decode("utf-8") # encode - перекодирует введенные данные в байты, decode - обратно
    tcp_socket.close()

    print(full_msg)
    
    
# ------------------------------------------------------------------

def main(params):
    try:
        window = "Easy camera"
        delay_skip = 30
        
        cv2.namedWindow(window ) 
        cv2.moveWindow(window, 160, 120)
        # $ v4l2-ctl --info -d /dev/video0 --list-formats-ext 
        #  format YUYV (4:2:2) 640x480 30 fps
        pipeline_YUY2 = 'v4l2src device=/dev/video0 ! video/x-raw, width=(int)160, height=(int)120, format=YUY2 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1'
        cap = cv2.VideoCapture( pipeline_YUY2,cv2.CAP_GSTREAMER  ) # 0 - запущенная камера # cv2.CAP_VFW cv2.CAP_GSTREAMER
        print('Camera init')
        if cap.isOpened():  
            rval = True 
        else: 
            rval = False 
            
        #rval, frame = cap.read()  
        #cv2.imwrite("/container_data/source/background.jpg",frame,[cv2.IMWRITE_EXR_COMPRESSION_NO,1])
        background = cv2.imread("/container_data/source/background.jpg",cv2.IMREAD_GRAYSCALE)
        #print(background.shape)
        activ_object = numpy.zeros((120,160,1),dtype=numpy.uint8)
        
        pos = 0
        change_color = 10
        while rval:  
            pos+=1
            change_color+=2
            if change_color >= 255:
                change_color=10
            rval, frame = cap.read()  
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (h,w) = frame_gray.shape
            if pos >= w-10:
                pos = 0
            
            #activ_object = numpy.zeros((h,w,1),dtype=numpy.uint8)
            #activ_object_copy = cv2.circle(activ_object.copy(),(pos,h//2),w//5,255,thickness=cv2.FILLED)
            activ_object_copy = cv2.rectangle(activ_object.copy(),(pos,0),(pos+10,20),change_color,thickness=cv2.FILLED)
            mask = cv2.bitwise_or(activ_object_copy,background)
          
            # TODO. TCP передвать только координату прямоугольника и его матрицу т.е. сам движущийся обьект,а не весь фон
            # Найти контур движущегося обьекта и взять его в большего размера матрицу прямоугольник 
            # на клиенте принять матрицу и склеить с фоном
            
            cv2.imshow(window, mask) 
            key = cv2.waitKey(delay_skip) & 0xFF
            if key == 27: # exit on ESC break 
                cap.release()
                cv2.destroyWindow(window)
                return 0
            
        print("Finish\nEnter ESC") 
        key = cv2.waitKey(0) & 0xFF  
        if key == 27: # exit on ESC break 
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
# python  /container_data/camera_example/test/server.py