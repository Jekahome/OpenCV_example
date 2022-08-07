'''
https://www.youtube.com/watch?v=xX-3eEjWL94&list=PLVFGVo0DNh5cLN1UBTT4yQ-cABFEi8Q8x&index=12

HOG - Histogram of Oriented Gradients
Гистаграмма направленных градиентов - быстро распознает лица

Подсчитывает количество направлений градиента в локальных областях изображения
Работает быстрее чем свертачная нейронная сеть, но менее точно

Install libs: https://pyimagesearch.com/2017/03/27/how-to-install-dlib/
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev
python -mpip install numpy
python -mpip install scipy
python -mpip install scikit-image
python -mpip install dlib
python -mpip install face_recognition

USAGE HOG:

	Run train HOG:
	python  /container_data/face/face_recognition_HOG_CNN/encode_faces.py \
		--dataset /container_data/source/face_train/dataset \
		--encodings /container_data/face/face_recognition_HOG_CNN/encodings.pickle \
		--detection-method hog

	Run recognize HOG:
	python /container_data/face/face_recognition_HOG_CNN/recognize_faces_video_file.py \
		--encodings /container_data/face/face_recognition_HOG_CNN/encodings.pickle \
		--detection-method hog \
		--output /container_data/source/face_train/face_recognition_HOG.avi


USAGE CNN:

	Run train CNN:
	python  /container_data/face/face_recognition_HOG_CNN/encode_faces.py \
		--dataset /container_data/source/face_train/dataset \
		--encodings /container_data/face/face_recognition_HOG_CNN/encodings.pickle \
		--detection-method cnn

	Run recognize CNN:
	python /container_data/face/face_recognition_HOG_CNN/recognize_faces_video_file.py \
		--encodings /container_data/face/face_recognition_HOG_CNN/encodings.pickle \
		--detection-method cnn \
		--output /container_data/source/face_train/face_recognition_CNN.avi

''' 

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()