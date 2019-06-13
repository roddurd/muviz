import cv2
import os
import numpy as np

img = cv2.imread("pfp3.jpg")
imgBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thres = cv2.threshold(imgBW, 30, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8) #square image kernel used for erosion
erosion = cv2.erode(thres, kernel,iterations = 1) #refines all edges in the binary image

opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #this is for further removing small noises and holes in the image

contours, _ = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #find contours with simple approximation

biggest_contour = max(contours, key=cv2.contourArea)
print("biggest contour: ", str(biggest_contour))
cv2.drawContours(closing, [biggest_contour], 0, (0, 255, 0), 3)
print("thres shape:", thres.shape)
print("closing shape:", closing.shape)
print("closing[0] shape:", closing[0].shape)
print("biggest_contour shape:", biggest_contour.shape)
print("biggest_contour[0] shape:", biggest_contour[0].shape)

##closing[:] =255 
#for i in range(closing.shape[0]):
#	for j in range(closing.shape[1]):
#		if [[i, j]] in biggest_contour:
#			closing[i][j] = 0
cv2.imshow('cleaner', closing) 
cv2.waitKey(0)
cv2.destroyAllWindows()
