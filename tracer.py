import cv2
import os
import numpy as np

img = cv2.imread("pfp.png")
imgBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thres = cv2.threshold(imgBW, 50, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8) #square image kernel used for erosion

erosion = cv2.erode(thres, kernel,iterations = 3) #refines all edges in the binary image
opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #this is for further removing small noises and holes in the image

contours, _ = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #find contours with simple approximation

#test difference in biggest_contour & contours
img_copy = img.copy()
cv2.drawContours(img_copy, contours, -1, (255, 255, 0), cv2.FILLED)

biggest_contour = max(contours, key=cv2.contourArea)
cv2.drawContours(img, [biggest_contour], -1, (0, 255, 0), cv2.FILLED)

cv2.imshow('biggest', img) 
cv2.imshow('contours', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
