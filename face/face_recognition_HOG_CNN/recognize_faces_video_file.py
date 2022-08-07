#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
USAGE
	TODO:модель должна быть предварительно обучена с помощью файла encode_faces.py

	python /container_data/face/face_recognition_HOG_CNN/recognize_faces_video_file.py \
		--encodings /container_data/face/face_recognition_HOG_CNN/encodings.pickle \
		--detection-method hog \
		--output /container_data/source/face_train/face_recognition_HOG.avi
'''
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

def write_video_result(cap,output):
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	writer = cv2.VideoWriter(output, fourcc, 24, (cap.shape[1], cap.shape[0]), True)
	# записать кадр с распознанными лицами на диск
	if writer is not None: writer.write(cap)
	return writer 

def create_stream_with_piplene():
	pipeline_YUY2 = 'v4l2src device=/dev/video0 ! video/x-raw, format=YUY2 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1'
	stream = cv2.VideoCapture(pipeline_YUY2,cv2.CAP_GSTREAMER) 
	return stream

def create_stream_with_src(input):
	stream = cv2.VideoCapture(input) 
	return stream

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
	ap.add_argument("-i", "--input", required=False, help="path to input video")
	ap.add_argument("-o", "--output", type=str, required=False, help="path to output video")
	ap.add_argument("-d", "--detection-method", type=str, required=False, help="face detection model to use: either `hog` or `cnn`")
	args = vars(ap.parse_args())

	# load the known faces and embeddings
	print("[INFO] loading encodings...")
	data = pickle.loads(open(args["encodings"], "rb").read())

	print("[INFO] processing video...")
	# vs = VideoStream(src=0).start()
	if args["input"] is not None: 
		stream = create_stream_with_src(args["input"])
	else:
		stream = create_stream_with_piplene()
 
	writer = None
	time.sleep(20)
 
	while True:
		# rval, frame = vs.read()
		(grabbed, frame) = stream.read()

		# если кадр не захвачен, то мы дошли до конца потока
		if not grabbed:
			break

		# изменить размер до ширины 750 пикселей (для ускорения обработки)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		rgb = imutils.resize(frame, width=750)
		r = frame.shape[1] / float(rgb.shape[1])

		# определить (x, y)-координаты ограничивающих рамок, соответствующих каждому лицу во входном кадре, 
  		# а затем вычислить вложения лиц для каждого лица
		boxes = face_recognition.face_locations(rgb,model=args["detection_method"])
		encodings = face_recognition.face_encodings(rgb, boxes)
		names = []

		for encoding in encodings:
			# попытаться сопоставить каждое лицо на входном изображении с нашими известными кодировками
			matches = face_recognition.compare_faces(data["encodings"],encoding)
			name = "Unknown"

			if True in matches:
				# найти индексы всех совпадающих лиц, а затем инициализировать словарь 
    			# для подсчета общего количества совпадений каждого лица.
				matchedIdxs = [i for (i, b) in enumerate(matches) if b]
				counts = {}
				for i in matchedIdxs:
					name = data["names"][i]
					counts[name] = counts.get(name, 0) + 1

				# определить распознанное лицо с наибольшим количеством голосов
				name = max(counts, key=counts.get)
			# обновить список имен
			names.append(name)

		# loop over the recognized faces
		for ((top, right, bottom, left), name) in zip(boxes, names):
			# изменить масштаб координат лица
			top = int(top * r)
			right = int(right * r)
			bottom = int(bottom * r)
			left = int(left * r)

			# Взять лицо в рамку
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1, cv2.LINE_AA)

		# Запись видео кадра
		if args["output"] is not None:
			writer = write_video_result(frame,args["output"])
			 
		# Показать видео кадр
		cv2.imshow("Frame", frame); 
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
		
	# закрыть указатели видеофайлов
	cv2.destroyAllWindows()
	# vs.stop()
	stream.release()

	# проверьте, нужно ли освободить указатель записи видео
	if writer is not None:
		writer.release()
  
if __name__ == "__main__":
    main()