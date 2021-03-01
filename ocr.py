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
	img = cv2.medianBlur(gray, 3)
	th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
		cv2.THRESH_BINARY,7,2)
	# th = cv2.adaptiveThreshold(cv2.medianBlur(gray, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

	custom_config = r'--oem 3 --psm 6 outputbase digits'
	d = pytesseract.image_to_data(th, output_type=Output.DICT, config = custom_config)

	n_boxes = len(d['text'])

	for i in range(n_boxes):
		if int(d['conf'][i]) > 60:
			(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
			img = cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
			boxes_val.append((x,y,w,h))

	cv2.imshow('res', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return boxes_val

def detect_oth(src):

	boxes_oth = []

	gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	# img = cv2.GaussianBlur(gray,(7,7),0)
	# th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	# 	cv2.THRESH_BINARY,5,2)
	# th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
	th = cv2.adaptiveThreshold(cv2.bilateralFilter(gray, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)

	custom_config = r'--oem 3 --psm 6 outputbase digits'
	d = pytesseract.image_to_data(th, output_type=Output.DICT, config=custom_config)
	print(d)

	n_boxes = len(d['text'])

	for i in range(n_boxes):
		if int(d['conf'][i]) > 60:
			(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
			img = cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)
			boxes_oth.append((x,y,w,h))

	cv2.imshow('res', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return boxes_oth


if __name__ == "__main__":

	src = cv2.imread("Circuit 6.jpg")
	src = cv2.resize(src, (640,640))
	boxes_oth = detect_oth(src)
	print(boxes_oth)
	# boxes_val = detect_values(src)
	# print(boxes_val)