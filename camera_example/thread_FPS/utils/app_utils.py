# python -mpip install -U imutils
# From http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

import struct
import six
import collections
import cv2
import datetime
import subprocess as sp
import json 
import numpy
import time
from threading import Thread
from matplotlib import colors

import datetime

class FPS:
	def __init__(self):
		# сохранить время начала, время окончания и общее количество кадров
		# которые были проверены между начальным и конечным интервалами
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# запускаем таймер
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# остановить таймер
		self._end = datetime.datetime.now()
	def update(self):
		# увеличить общее количество фреймов, проверенных во время
		# начальный и конечный интервалы
		self._numFrames += 1
	def elapsed(self):
		# вернуть общее количество секунд между стартом и
		# конечный интервал
		return (self._end - self._start).total_seconds()
	def fps(self):
		# вычисляем (приблизительно) количество кадров в секунду
		return self._numFrames / self.elapsed()


class WebcamVideoStream:
	def __init__(self, src=0, width=640, height=480):
		# инициализируем поток видеокамеры и читаем первый кадр
		# из потока
		self.stream = cv2.VideoCapture(src,cv2.CAP_GSTREAMER)
		self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		(self.grabbed, self.frame) = self.stream.read()

		# инициализировать переменную, используемую для указания, должен ли поток
		# быть остановленным
		self.stopped = False

	def start(self):
		# запускаем поток для чтения кадров из видеопотока
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# продолжать цикл бесконечно, пока поток не будет остановлен
		while True:
			# если установлена ​​переменная индикатора потока, остановить поток
			if self.stopped:
				return

			# иначе читаем следующий кадр из потока
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# вернуть последний прочитанный фрейм
		return self.frame

	def stop(self):
		# указываем, что поток должен быть остановлен
		self.stopped = True