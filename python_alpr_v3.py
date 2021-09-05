import sys
import cv2
import numpy as np
import gpyocr
plateCascade = cv2.CascadeClassifier("/home/pi/haarcascade_eu_plate_number.xml")
color = (255,120,255)
arg = float(sys.argv[1])

outText = ""

def getText(filename):
	outText = gpyocr.tesseract_ocr(filename,psm=7)[0]
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

def getPlates(img):
    plates = plateCascade.detectMultiScale3(img, arg, outputRejectLevels=True)
    if len(plates[0]) > 0:
        totalPlates = zip(plates[0], plates[2])
        totalPlates = list(totalPlates)
        totalPlates.sort(key = lambda x: x[1][0], reverse=True)
        return [totalPlates[0][0][0], totalPlates[0][0][1], totalPlates[0][0][2], totalPlates[0][0][3]]
    else:
        raise Exception("No plates found!")
    
def doALPR(img):
	foundText = ""
	x = 0
	y = 0
	h = 0
	w = 0
	temp = img
	imgGray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
	[x, y, w, h] = getPlates(imgGray)
	roi_gray = imgGray[y:y+h, x:x+w]
	roi_color = img[y+2:y+h-2, x+2:x+w-2]
	cv2.rectangle(temp, (x, y), (x + w, y + h), color, 2)
	cv2.putText(temp, "Tablica", (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
	cv2.imwrite("bbox.png",temp)
	cv2.imwrite("result.png", roi_color)
	resized = resize(roi_gray)
	morph = getMorph(resized)
	cv2.imwrite("morph.png", morph)
	foundText = getText("/home/pi/morph.png")
	return foundText

img = cv2.imread("image.png")
outText = doALPR(img)
if len(outText)==0:
	raise Exception("Numbers not detected!")
else:
	print(outText)

