import cv2 
import os
import numpy as np
img = cv2.imread("pfp3.jpg")
height, width, _ = img.shape



#img zoom (warpPerspective)
def corners(img):
	height, width, _ = img.shape
	return [[0, 0], [width, 0], [0, height], [width, height]]
def zoom_corners(img, percent_zoom):
	cnrs = corners(img)
	percent_zoom /= 100
	cnrs = np.array(cnrs)
	zoom_corners = [corner*(1 + percent_zoom) for corner in cnrs]
	return zoom_corners	
def warp(img, pts1, pts2):
	pts1, pts2 = map(np.float32, (pts1, pts2))
	M = cv2.getPerspectiveTransform(pts1, pts2)
	zimg = cv2.warpPerspective(img, M, (width, height))
	return zimg
def zoom(img, percent_zoom):
	cnrs = corners(img)
	zoom_cnrs = zoom_corners(img, percent_zoom)
	return warp(img, cnrs, zoom_cnrs)

#red,blue, green -scale
def color(img, color):
	colors = {"blue":0, "green":1, "red":2}
	index = colors.get(color)
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	red_img = np.zeros_like(img)
	red_img[:, :, index] = gray_img
	return red_img

img_zoom = zoom(img, 100)
cv2.imshow('zoom', img_zoom)
cv2.waitKey(0)

