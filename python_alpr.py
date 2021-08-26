import sys
import cv2
import numpy as np
import gpyocr
from PIL import Image
print("--- Starting Computer Vision ---")
plateCascade = cv2.CascadeClassifier("/home/pi/haarcascade_eu_plate_number.xml")
color = (255,120,255)
arg = float(sys.argv[1])
bbox_x = float(sys.argv[2])
bbox_y = float(sys.argv[3])
bbox_w = float(sys.argv[4])
bbox_h = float(sys.argv[5])

outText = ""

def getText(filename):
	outText = gpyocr.tesseract_ocr('morph.png',psm=7,config='tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')[0]
	return outText

def resize(img):
	imgCopy = img.copy()
	resized_img = cv2.resize(imgCopy, (int(imgCopy.shape[1] * 220/100), int(imgCopy.shape[0] * 220/100)), interpolation = cv2.INTER_AREA)
	return resized_img

def getMorph(img):
	kernel = np.ones((3, 3), np.uint8)
	_, morph = cv2.threshold(cv2.GaussianBlur(img,(5,5),0),0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	morph = cv2.erode(morph,kernel)
	morph = cv2.dilate(morph,kernel)
	morph = cv2.bitwise_not(morph)
	pad = cv2.copyMakeBorder(morph, 1,1,1,1, cv2.BORDER_CONSTANT, value=255)
	h, w = pad.shape
	mask = np.zeros([h+2, w+2], np.uint8)
	img_flood = cv2.floodFill(pad, mask, (0,0), 0, (5), (0), flags=8)[1]
	img_flood = img_flood[1:h-1, 1:w-1]
	morph = cv2.bitwise_not(img_flood)
	morph = cv2.dilate(morph, kernel)
	morph = cv2.erode(morph, kernel, iterations=3)
	morph = cv2.dilate(morph, kernel)
	return morph

def getBox(img):
	height, width, _ = img.shape
	return (bbox_x*width, bbox_y*height, bbox_w*width, bbox_h*height)

def detect(img):
	errorFlag = 1
	global outText
	global bbox_x, bbox_y, bbox_w, bbox_h
	x = 0
	y = 0
	h = 0
	w = 0
	max_level = 0
	bbox_x, bbox_y, bbox_w, bbox_h = getBox(img)
	print("BBox of car:", bbox_x, bbox_y, bbox_w, bbox_h)
	temp = img
	imgGray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
	plates = plateCascade.detectMultiScale3(imgGray, arg, outputRejectLevels=True)
	print(plates)
	if len(plates[0]) > 0:
		for i in range(len(plates[0])):
			(x_temp, y_temp, w_temp, h_temp) = plates[0][i]
			level_temp = plates[2][i]
			if x_temp >= bbox_x and y_temp >= bbox_y and (x_temp + w_temp) <= (bbox_x + bbox_w) and (y_temp + h_temp) <= (bbox_y + bbox_h) and level_temp > max_level:
				print("BBox of plate:", x_temp, y_temp, w_temp, h_temp)
				max_level = level_temp
				x = x_temp
				y = y_temp
				w = w_temp
				h = h_temp
				errorFlag = 0
	if errorFlag:
		raise Exception("No plates found!")
	else:
		roi_gray = imgGray[y:y+h, x:x+w]
		roi_color = img[y+2:y+h-2, x+2:x+w-2]
		cv2.rectangle(temp, (x, y), (x + w, y + h), color, 2)
		cv2.putText(temp, "Tablica", (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
		cv2.imwrite("result.png", roi_color)
		resized = resize(roi_gray)
		morph = getMorph(resized)
		cv2.imwrite("morph.png", morph)
		outText = getText("/home/pi/morph.png")


	cv2.imwrite("bbox.png",temp)
print("--- Reading Image ---")
img = cv2.imread("image.png")
print("--- Detecting Plates ---")
detect(img)
if len(outText)==0:
	print("Numbers not detected!")
else:
	print(outText)
file = open("resultText.txt","w")
file.write(outText)
file.close()
