import cv2 
import os
import numpy as np
img = cv2.imread("pfp3.jpg")
height, width, _ = img.shape

#img zoom (warpPerspective)
pts1 = np.float32([[20, 20],[width-20, 20], [20, height-20],[width-20, height-20]])
pts2 = np.float32([[0, 0],  [width, 0], [0, height],[width, height]])
M = cv2.getPerspectiveTransform(pts1, pts2)

zimg = cv2.warpPerspective(img, M, (width, height))


#red,blue, green -scale
def color(img, color):
	colors = {"blue":0, "green":1, "red":2}
	index = colors.get(color)
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	red_img = np.zeros_like(img)
	red_img[:, :, index] = gray_img
	return red_img

blue = color(img, 'blue')
cv2.imshow('blue', blue)
cv2.waitKey(0)

