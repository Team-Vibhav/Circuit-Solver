import numpy as np
import cv2
import pandas as pd
from math import isnan
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

src = cv2.imread("Circuit 5.jpeg")
src = cv2.resize(src,(640,640))

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(gray,(9,9),0)
th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv2.THRESH_BINARY,5,2)
# th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


custom_config = r'--oem 3 --psm 6 outputbase digits'
d = pytesseract.image_to_data(th, output_type=Output.DICT, config = '--psm 6')

print(d)



n_boxes = len(d['text'])

for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('res', th)
cv2.waitKey(0)
cv2.destroyAllWindows()