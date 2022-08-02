#!/usr/bin/python
# -*- coding: utf-8 -*-
from binascii import a2b_hex
import sys, os
import cv2 
import numpy
from platform import python_version
import colorsys
import urllib.request

 
print('OpenCV',cv2.__version__)  
print('Python',python_version())
# $ python -c "import cv2; print(cv2.__version__)"

'''
cap = cv2.VideoCapture(__gstreamer_pipeline(camera_id=0, flip_method=2), cv2.CAP_GSTREAMER)

'''
def __gstreamer_pipeline(
        camera_id,
        capture_width=160,
        capture_height=120,
        display_width=160,
        display_height=120,
        framerate=30,
        flip_method=0,
    ):
    return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True"
            % (
                    camera_id,
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    display_width,
                    display_height,
            )
    )
    
'''
https://docs.opencv.org/4.5.5/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d

Format pipeline:
https://forums.developer.nvidia.com/t/cannot-open-opencv-videocapture-with-gstreamer-pipeline/181639

apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
'''
def main(params):
    try:
        window = "Easy camera"
        delay_skip = 30
        
        cv2.namedWindow(window ) 
        #cv2.moveWindow(window, 640, 480)
         # $ v4l2-ctl --info -d /dev/video0 --list-formats-ext 
         #  format YUYV (4:2:2) 640x480 30 fps
         # width=(int)640, height=(int)480,
        pipeline_YUY2 = 'v4l2src device=/dev/video0 ! video/x-raw, format=YUY2 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1'
        # Или этот, использующий HW для преобразования YUV -> BGRx, так что преобразование видео на ЦП удаляет только 4-й байт.
        # pipeline_YUY2 = 'v4l2src device=/dev/video0 ! video/x-raw, format=YUY2 ! nvvidconv ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1'
        #pipeline_MJPG = " v4l2src device=/dev/video0 ! image/jpeg, format=MJPG ! jpegdec ! video/x-raw,format=BGR ! appsink drop=1"
        #pipeline_MJPG = "v4l2src device=/dev/video0 ! image/jpeg, format=MJPG ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        cap = cv2.VideoCapture( pipeline_YUY2,cv2.CAP_GSTREAMER  ) # 0 - запущенная камера # cv2.CAP_VFW cv2.CAP_GSTREAMER
       
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        #cap.set(cv2.CAP_PROP_FPS,30.0) # устанавливает частоту обновления оборудования камеры, обычно составляет 30 к/с
        print('Camera init')
        if cap.isOpened(): # try to get the first frame 
            rval = True 
        else: 
            rval = False 
          
        while rval: 
            rval, frame = cap.read() 
            #frame = cv2.bilateralFilter(frame, 9, 75, 75)
            frame = cv2.flip(frame,1)
             
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame,"FPS:{0}".format(fps),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            cv2.imshow(window, frame) 
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
# python  /container_data/camera_example/easy_camera.py