
### Image

С чего начать ?
Книги и уроки,документация в основном на C++
 
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

------------------------------------------------------------------------------------------------------

### Deep learning freeCodeCamp.org
TODO: Слабый процент распознавания

video: https://www.youtube.com/watch?v=oXlwWbU8l2o&list=PLuudOZcE9EgnCN8cgw9FcPeQnjP1axXP_&index=7&t=176s
code: https://github.com/jasmcaus/opencv-course/blob/master/Section%20%234%20-%20Capstone/simpsons.py
kaggle: https://www.kaggle.com/

Add video freeCodeCamp.org:
https://www.youtube.com/watch?v=tPYj3fFJGjk&list=PLWKjhJtqVAblStefaz_YOVpDWqcRScc2s
https://www.youtube.com/watch?v=vo_fUOk-IKk&list=PLWKjhJtqVAbm3T2Eq1_KgloC7ogdXxdRa
https://www.youtube.com/watch?v=5ioMqzMRFgM&list=PLWKjhJtqVAbm5dir5TLEy2aZQMG7cHEZp