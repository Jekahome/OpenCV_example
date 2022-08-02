# https://github.com/OlafenwaMoses/ImageAI
'''
 /home/jeka/.local/bin/python -m  pip install imageai --upgrade
'''
# $ /home/jeka/.local/bin/python imageai_example_2.py 
 

from imageai.Detection import Detection

detector = ObjectDetection()
detector.setModelTypeAsTinyTOLOv3()
detector.setModelPath("/container_data/source/tools/yolo-tiny.h5")
detector.loadModel() # загрузка в оперативную память

detector.detectObjectsFromImage(input_image="/container_data/source/objects.png",output_image="/container_data/source/new_objects.png")