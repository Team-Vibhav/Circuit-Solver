import numpy as np
import cv2
import pandas as pd
from math import isnan
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_values(src):

	boxes_val = []

	src = cv2.resize(src, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

	gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(gray, (3,3), 0)
	ret, th = cv2.threshold(img,1010, 200, cv2.THRESH_OTSU, cv2.THRESH_BINARY)
	# th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
	# ret3,th = cv2.threshold(img,127,200,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 7)

	custom_config = r'--oem 3 --psm 6 outputbase digits'
	d = pytesseract.image_to_data(th, output_type=Output.DICT, config = custom_config)

	n_boxes = len(d['text'])

	for i in range(n_boxes):
		if int(d['conf'][i]) > 60:
			(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
			img = cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
			boxes_val.append([(x,y,w,h),int(d['text'][i])])
	
	# print(boxes_val)

	# cv2.imshow('res', th)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	return boxes_val

def detect_oth(src):

	boxes_oth = []
	img = src.copy()

	gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	blur = cv2.medianBlur(gray,7)
	th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
	# th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	# 	cv2.THRESH_BINARY,9,2)

	custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789 outputbase digits'
	d = pytesseract.image_to_string(th, config=custom_config)
	print(d)
	# n_boxes = len(d['text'])

	# for i in range(n_boxes):
	# 	if int(d['conf'][i]) > 60:
	# 		(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
	# 		img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
	# 		boxes_oth.append((x,y,w,h))
	# 		print(d['text'][i])
	# 		print((x,y,w,h))
	
	# cv2.imshow('res', img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	return boxes_oth


if __name__ == "__main__":

	src = cv2.imread("Circuit 6.jpg")
	src = cv2.resize(src, (640,640))
	
	# boxes_oth = detect_oth(src)
	
	boxes_val = detect_values(src)
	