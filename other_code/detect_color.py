# Поиск цвета в обнаруженном контуре OpenCV python
# https://ru.stackoverflow.com/questions/1178677/%D0%9F%D0%BE%D0%B8%D1%81%D0%BA-%D1%86%D0%B2%D0%B5%D1%82%D0%B0-%D0%B2-%D0%BE%D0%B1%D0%BD%D0%B0%D1%80%D1%83%D0%B6%D0%B5%D0%BD%D0%BD%D0%BE%D0%BC-%D0%BA%D0%BE%D0%BD%D1%82%D1%83%D1%80%D0%B5-opencv-python

'''
Непосредственно сама задача: имеется код (смотреть ниже), который находит объекты при помощи метода вычитания фона. Мне необходимо найти определенный цвет в заданном контуре, который будет задаваться нижней и верхней границей цвета формата HSV.
'''

import cv2 as cv

cap = cv.VideoCapture('videos/5.mp4')
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    fgmask = fgbg.apply(frame)
    contours, hierarchy = cv.findContours(fgmask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        # area = cv2.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        x, y, w, h = cv.boundingRect(contour)
        col = cv.mean(frame, fgmask)
        print(int(col[0]))

        if perimeter > 400:
            M = cv.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(f"{cX}:{cY}")
            cv.circle(frame, (cX, cY), 3, (255, 255, 255), -1)
        # cv.putText(frame, "color", (x, y), frame.FONT_HERSHEY_SIMPLEX, 0.1, (int(col[0]), int(col[1]), int(col[2])))
            cv.rectangle(frame, (x, y), (x + w, y + w), (int(col[0]), int(col[1]), int(col[2])), 2)
            print(cv.mean(frame, fgmask))
    cv.imshow("Frame", frame)
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    
cap.release() 
cv.destroyAllWindows()