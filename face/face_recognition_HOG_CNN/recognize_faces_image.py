#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
USAGE

python /container_data/other_code/Face-recognition/Face_Recognition_CNN/recognize_faces_image.py \
		--encodings /container_data/other_code/Face-recognition/Face_Recognition_CNN/encodings.pickle \
		--detection-method hog \
		--input /container_data/source/face_train/face.png
'''
import face_recognition
import argparse
import pickle
import cv2

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--encodings", required=True,help="path to serialized db of facial encodings")
	ap.add_argument("-i", "--input", required=True,help="path to input image")
	ap.add_argument("-d", "--detection-method", type=str,required=True, help="face detection model to use: either `hog` or `cnn`")
	args = vars(ap.parse_args())

	# load the known faces and embeddings
	print("[INFO] loading encodings...")
	data = pickle.loads(open(args["encodings"], "rb").read())

	# загрузить входное изображение и преобразовать его из BGR в RGB
	image = cv2.imread(args["input"])
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# определить (x, y)-координаты ограничивающих рамок, соответствующих каждому лицу во входном кадре, 
	# а затем вычислить вложения лиц для каждого лица
	print("[INFO] recognizing faces...")
	boxes = face_recognition.face_locations(rgb,model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)

	# initialize the list of names for each face detected
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		#  попытаться сопоставить каждое лицо на входном изображении с нашими известными кодировками
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
		# Взять лицо в рамку
		cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1,cv2.LINE_AA)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

	cv2.imshow("Image", image)
	cv2.waitKey(0)

if __name__ == "__main__":
    main()