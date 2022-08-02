
### Image

С чего начать ?
Книги и уроки,документация в основном на C++
<<<<<<< HEAD
=======
 
1 [Гоша дударь](https://www.youtube.com/playlist?list=PL0lO_mIqDDFUAQ2RdAgLp6Tj_fREcxk6T)
>>>>>>> ac03be3047363d1ff75dbd8e9b8973212d3519dc
 
2 Image Processing (imgproc module) https://docs.opencv.org/4.5.5/d7/da8/tutorial_table_of_content_imgproc.html

3 Next ... https://docs.opencv.org/4.5.5/d9/df8/tutorial_root.html

------------------------------------------------------------------------------------------------------------------
 
Устарели https://robocraft.ru/opencv

------------------------------------------------------------------------------------------------------------------

Морфология:
    Добавить в morphological_transformations_hitmiss()
    - координаты положения объектов

------------------------------------------------------------------------------------------------------------------

Добавить пример операций с каналами bitwise_and, bitwise_no, cv::mixChannels
https://docs.opencv.org/4.5.5/d2/de8/group__core__array.html#ga51d768c270a1cdd3497255017c4504be        

------------------------------------------------------------------------------------------------------------------

Добавить выравнивание гистограммы gray img = cv2.equalizeHist(img_gray)

------------------------------------------------------------------------------------------------------------------

Добавить функции преобразования:cv2.warpAffine с участием cv2.warpPerspective, cv2.resize(), cv2.getAffineTransform, cv2.getPerspectiveTransform
https://russianblogs.com/article/9615742843/
res = cv2.resize(img, None, fx=2, fy=3, interpolation=cv2.INTER_LANCZOS4)  

------------------------------------------------------------------------------------------------------------------


2 Перспективное преобразование
2 Выравнивание картинки Афинное преобразование https://russianblogs.com/article/9615742843/
rows, cols, ch = img.shape
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

------------------------------------------------------------------------------------------------------------------

3  Обнаружения ключевых точек cv2.ORB_create
cv2.BFMatcher() Затем нам необходимо сопоставить(вычислить расстояние) дискрипторы первого изображения с дискрипторами второго и взять ближайший.
https://habr.com/ru/post/547218/
