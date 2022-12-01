## План

 

### Документация 

[docs.opencv.org 4.5.5](https://docs.opencv.org/4.5.5/d6/d00/tutorial_py_root.html)

[OpenCV tutorial](https://docs.opencv.org/4.5.5/d9/df8/tutorial_root.html)

### Базовые возможности OpenCV

[1. OpenCV. Beyond Robotics](https://www.youtube.com/playlist?list=PLVFGVo0DNh5duhps6KsiCQIoObyzcM2Cs)

[2. OpenCV Course - Full Tutorial with Python](https://www.youtube.com/watch?v=oXlwWbU8l2o&t=252s)

### Дополнительный материал OpenCV

[spmallick/learnopencv](https://github.com/spmallick/learnopencv/blob/master/README.md)

[Tensorflow and OpenCV](https://towardsdatascience.com/building-a-real-time-object-recognition-app-with-tensorflow-and-opencv-b7a2b4ebdc32)

[Increasing webcam FPS with Python and OpenCV](https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/)

[Tensorflow, OpenCV and Docker](https://towardsdatascience.com/real-time-and-video-processing-object-detection-using-tensorflow-opencv-and-docker-2be1694726e5)

 
[robocraft.ru C++](https://robocraft.ru/opencv)

[OpenCV tutorial pyimagesearch.com](https://pyimagesearch.com/category/opencv/page/2/)

[OpenCV university pyimagesearch.com](https://pyimagesearch.com/pyimagesearch-university/)

[opencv.org](https://opencv.org/opencv-free-course/)

[2.Нейронные сети. Beyond Robotics](https://www.youtube.com/watch?v=7jbAdB5lt9I&ab_channel=BeyondCurriculum)

[MediaPipe](https://google.github.io/mediapipe/getting_started/python.html)
Библиотека распознавания частей тела

[Boris Bochkarev](https://www.youtube.com/playlist?list=PLOjc9X-vV0SEHQDXm30Ts-3GGqn_PKaNu)

[russianblogs.com](https://russianblogs.com/article/44371526060/)

[russianblogs.com](https://russianblogs.com/article/2777646555/)

[ProgrammingKnowledge](https://www.youtube.com/playlist?list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K)

[Arboook](https://arboook.com/kompyuternoe-zrenie/operatsii-s-tsvetom-v-opencv3-i-python/)
 
[megabax](https://habr.com/ru/users/megabax/posts/)

[Библиотека для OpenCV](https://pyimagesearch.com/2015/02/02/just-open-sourced-personal-imutils-package-series-opencv-convenience-functions/)

[pyimagesearch.com](https://pyimagesearch.com/category/opencv/)

[ESP32 + OpenCV](https://how2electronics.com/color-detection-tracking-with-esp32-cam-opencv/)

[ESP32 + OpenCV](https://how2electronics.com/iot-projects/esp32-cam-projects/)

[Face recognition](https://github.com/L4HG/face_recognition)

[Face recognition](https://www.youtube.com/playlist?list=PLS1QulWo1RIbp_ImnSEWEMRLnJVfEc-GR)

[Нейронные сети](https://python-scripts.com/category/neural-network)

[PyTorch Adrian Rosebrock](https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/)

[PyTorch Adrian Rosebrock](https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/)

[PyTorch Adrian Rosebrock](https://pyimagesearch.com/2021/07/26/pytorch-image-classification-with-pre-trained-networks/)

[PyTorch](https://pyimagesearch.com/2021/07/05/what-is-pytorch/)

[Подборка материалов по OpenCV для Python](https://vk.com/wall-30666517_1493958)

[Машинное обучение](http://www.machinelearning.ru/wiki/index.php?title=Machine_Learning)

[Books](https://opencv.org/books/)

### [Examples code]

[Examples code](https://www.programcreek.com/python/example/89361/cv2.Canny)

### Use docker image [jekshmek/opencv_rep] from `less/` dir:

[jekshmek/opencv_rep]:(https://github.com/Jekahome/Docker_OpenCV)
```
Create mount:
$ docker volume create \
    --name host_data_source \
    --opt type=bind \
    --opt device=/home/jeka/Projects/OpenCV/less \
    --opt o=bind

TODO:Current dir `/home/jeka/Projects/OpenCV/less` == /container_data

Use mount dir:
$ docker run --rm \
    --name opencv_rep_less \
    --mount source=host_data_source,destination=/container_data \
    -it jekshmek/opencv_rep python /container_data/info.py

Use camera:
$ xhost +local:docker (or `$ xhost +`)
$ docker run --rm --env DISPLAY=$DISPLAY --privileged \
     --mount source=host_data_source,destination=/container_data \
     --volume /tmp/.X11-unix:/tmp/.X11-unix --volume /tmp/.docker.xauth:/tmp/.docker.xauth --env NO_AT_BRIDGE=1 \
     --name opencv_rep_less -it jekshmek/opencv_rep python /container_data/camera_example/camera.py   


TODO: maybe need use camera
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge - 
ocker run .... -v $XAUTH:$XAUTH
```

  
### Озвучка
https://gtts.readthedocs.io/en/latest/
 

## Посмотреть

### [Распознавание речи на Пайтон AI] 
[Распознавание речи на Пайтон AI] (https://www.youtube.com/playlist?list=PLxiU3nwEQ4PHLCwPFUdr3u2oScfIhUohe)

### [Распознавание контуров OpenCV]

[Распознавание контуров OpenCV](https://www.youtube.com/watch?v=Sr0KQftXcEg&feature=youtu.be)

### Алгоритм обнаружения лица RetinaFace

[RetinaFace](https://github.com/serengil/retinaface) 

## Python docs
https://docs.python.org/3/

## Course

[pyimagesearch.com](https://pyimagesearch.com/pyimagesearch-university/)
[github pyimagesearch.com](https://github.com/orgs/PyImageSearch/repositories)
[github search pyimagesearch](https://github.com/search?q=pyimagesearch&type=code)

## [Библиотеки для компьютерного зрения]

[Библиотеки для компьютерного зрения]:(https://arboook.com/kompyuternoe-zrenie/moj-top-7-bibliotek-dlya-python-dlya-kompyuternogo-zreniya/)

### NumPy

NumPy позволяет быстро и удобно работать с такими структурами данных как большие многомерные массивы.
```pip install numpy```

### [Matplotlib]

[Matplotlib]:(https://pyprog.pro/mpl/mpl_types_of_graphs.html)
Matplotlib это настоящий комбайн для отображения данных. 
```pip install matplotlib```

### IPython Jupyter

Среда программирования для Python
```pip install jupyter```

### OpenCV

OpenCV по сути стандарт в области обработки изображений и компьютерного зрения.

[Модели признаки haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)

### SciPy

Данный пакет расширяет возможности работы с векторами и матрицами.
```pip install scipy```

### scikit-learn

Начиная с этой библиотеки – все будет связано с машинным обучением. Тут уж ничего не поделаешь, если ты хочешь не только повернуть и уменьшить картинку, но и понять что на ней находится, ты попадаешь на территорию ML.  Scikit-learn это самый легкий способ начать свое знакомство с этой областью. Данная библиотека содержит различные реализации методов кластеризации, классификации и других. Устанавливается через менеджер пакетов pip командой: ```pip install scikit-learn```

### Tensor-flow

Для ее изучения, а не просто запуска демок, придется понять хотя бы теорию нейронных сетей, глубоких нейронных сетей. На самом деле, после получения базовых знаний работать с ней приятно а ее скорость и гибкость вас приятно удивят.
```pip install tensorflow```

### Caffe

Основной ее целью было создать инструмент для анализа мультимедийных данных с открытым исходным кодом который будет ориентирован на коммерческое применение. Вся библиотека написана на C++, в своей работе опирается на обработку данных графическим чипом, полностью поддерживает написание пользовательских алгоритмов на Python/NumPy, а также совместим с MATLAB.

### CatBoost

CatBoost отличается от Tensor-Flow и Caffe тем, что она реализует механизм глубокого обучения через градиентный бустинг, в отличии от нейронных сетей. Библиотека разработана и написана в Яндексе, поддерживает работу из Python и R, имеет открытый исходный код.
