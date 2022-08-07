
## HOG - Histogram of Oriented Gradients

Гистаграмма направленных градиентов - быстро распознает лица

Подсчитывает количество направлений градиента в локальных областях изображения
Работает быстрее чем свертачная нейронная сеть, но менее точно

[BeyondCurriculum HOG](https://www.youtube.com/watch?v=xX-3eEjWL94&list=PLVFGVo0DNh5cLN1UBTT4yQ-cABFEi8Q8x&index=12)


## Dependencies

[Install libs](https://pyimagesearch.com/2017/03/27/how-to-install-dlib/)

```
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev
python -mpip install numpy
python -mpip install scipy
python -mpip install scikit-image
python -mpip install dlib
python -mpip install face_recognition
```

## USAGE:

```
USAGE HOG:

1.Run train HOG:
python  /container_data/other_code/Face-recognition/Face_Recognition_CNN/encode_faces.py \
    --dataset /container_data/source/face_train/dataset \
    --encodings /container_data/other_code/Face-recognition/Face_Recognition_CNN/encodings.pickle \
    --detection-method hog

2.Run recognize HOG:
python /container_data/other_code/Face-recognition/Face_Recognition_CNN/recognize_faces_video_file.py \
    --encodings /container_data/other_code/Face-recognition/Face_Recognition_CNN/encodings.pickle \
    --detection-method hog \
    --output /container_data/source/face_train/face_recognition_HOG.avi


USAGE CNN:

1.Run train CNN:
python  /container_data/other_code/Face-recognition/Face_Recognition_CNN/encode_faces.py \
    --dataset /container_data/source/face_train/dataset \
    --encodings /container_data/other_code/Face-recognition/Face_Recognition_CNN/encodings.pickle \
    --detection-method cnn

2.Run recognize CNN:
python /container_data/other_code/Face-recognition/Face_Recognition_CNN/recognize_faces_video_file.py \
    --encodings /container_data/other_code/Face-recognition/Face_Recognition_CNN/encodings.pickle \
    --detection-method cnn \
    --output /container_data/source/face_train/face_recognition_CNN.avi
```