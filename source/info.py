#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
import os 
with open("/container_data/source/new-file",'a+',encoding="utf8") as resource:
     text = resource.read()
     resource.write("hello world")
     resource.close()
'''

from platform import python_version
import cv2
print('OpenCV',cv2.__version__) # 4.5.4
print('Python',python_version()) # 3.10.4

