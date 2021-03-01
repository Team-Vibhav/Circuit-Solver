kernel = np.ones((1,1), np.uint8)
	img = cv2.dilate(gray, kernel, iterations=1)
	img = cv2.erode(img, kernel, iterations=1)