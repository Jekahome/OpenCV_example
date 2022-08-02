# https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

from __future__ import print_function

#from imutils.video import WebcamVideoStream
#from imutils.video import FPS
from utils.app_utils import FPS, WebcamVideoStream
import argparse
import imutils # python -mpip install -U imutils
import cv2

if __name__ == '__main__':
    # создаем аргумент parse и анализируем аргументы
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-frames", type=int, default=100,help="# Количество кадров для проверки FPS")
    ap.add_argument("-d", "--display", dest='display', type=int, default=-1,help="Должны ли отображаться кадры")
    ap.add_argument("-wd", "--width", dest='width', type=int,default=640, help='Width of the frames in the video stream.')
    ap.add_argument("-ht", "--height", dest='height', type=int,default=480, help='Height of the frames in the video stream.')
    args = vars(ap.parse_args())
 
    if True:
        # не использует многопоточность и использует блокировку ввода-вывода при чтении кадров из потока камеры
        # получаем указатель на видеопоток и инициализируем счетчик FPS
        print("[INFO] выборка кадров с веб-камеры...")
        stream = cv2.VideoCapture(0,cv2.CAP_VFW)
        fps = FPS().start()
        # цикл по некоторым кадрам
        while fps._numFrames < args["num_frames"]:
            # захватить кадр из потока и изменить его размер, чтобы он был максимальным
            # ширина 400 пикселей
            (grabbed, frame) = stream.read()
            frame = imutils.resize(frame, width=400)
            # проверяем, должен ли кадр отображаться на нашем экране
            if args["display"] > 0:
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
            # обновить счетчик FPS
            fps.update()
        # останавливаем таймер и отображаем информацию о FPS
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        # сделать небольшую очистку
        stream.release()
        cv2.destroyAllWindows()

    if True:
        # создал *многопоточный* видеопоток, разрешил датчику камеры прогреться,
        # и запускаем счетчик FPS
        print("[INFO] выборка THREADED кадров с веб-камеры...")
        #vs = WebcamVideoStream(src=0,width=int(args['width']),height=int(args['height'])).start()
        vs = WebcamVideoStream(src=0).start()
        fps = FPS().start()
        # зациклиться на некоторых кадрах... на этот раз с использованием многопоточного потока
        while fps._numFrames < args["num_frames"]:
            # захватить кадр из видеопотока и изменить его размер
            # иметь максимальную ширину 400 пикселей
            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            # проверяем, должен ли кадр отображаться на нашем экране
            if args["display"] > 0:
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
            # обновить счетчик FPS
            fps.update()
        # останавливаем таймер и отображаем информацию о FPS
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        # сделать небольшую очистку
        cv2.destroyAllWindows()
        vs.stop()
        
        
# Run:
'''
$ python /container_data/camera_example/thread_FPS/fps_demo.py  

    [INFO] выборка кадров с веб-камеры...
    [INFO] elasped time: 10.79
    [INFO] approx. FPS: 9.27

    [INFO] выборка THREADED кадров с веб-камеры...
    [INFO] elasped time: 0.52
    [INFO] approx. FPS: 191.86

--------------------------------------------------------------------------

$ python /container_data/camera_example/thread_FPS/fps_demo.py --display=1

    [INFO] выборка кадров с веб-камеры...
    [INFO] elasped time: 10.79
    [INFO] approx. FPS: 9.27

    [INFO] выборка THREADED кадров с веб-камеры...
    [INFO] elasped time: 0.77
    [INFO] approx. FPS: 129.16

'''
 