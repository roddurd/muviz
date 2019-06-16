import cv2 
import os
import numpy as np
img = cv2.imread("pfp3.jpg")



#img zoom (warpPerspective)
def corners(img):
	height, width, _ = img.shape
	return [[0, 0], [width, 0], [0, height], [width, height]]
def dim_corners(cnrs):
	"""returns an np array [width, height] of a region bounded by corners in format [[x1, y1], ... [x4, y4]]"""
	cnrs = np.array(cnrs)
	width = np.linalg.norm(cnrs[1]-cnrs[0])
	height = np.linalg.norm(cnrs[2]-cnrs[0])
	return np.array([width, height])	
def zoom_corners(img, percent_zoom):
	"""returns an array, the set of corners comprising the box which is percent_zoom% zoomed into the center of the image, e.g., if percent_zoom is 100, then the width and height of img will be 100% bigger than the width and height of the box bounded by zoom_corners(img, 100)""" 
	height, width, _ = img.shape
	cnrs = corners(img)
	percent_zoom /= 100
	cnrs = np.array(cnrs)
	zoom_cnrs = [corner/(1 + percent_zoom) for corner in cnrs]
	offset = ((width - dim_corners(zoom_cnrs)[0])/2, ((height - dim_corners(zoom_cnrs)[1]))/2)
	zoom_cnrs = [corner+offset for corner in zoom_cnrs]	
	return zoom_cnrs	
def warp(img, pts1, pts2):
	height, width, _ = img.shape
	pts1, pts2 = map(np.float32, (pts1, pts2))
	M = cv2.getPerspectiveTransform(pts2, pts1)
	zimg = cv2.warpPerspective(img, M, (width, height))
	return zimg
def zoom(img, percent_zoom):
	cnrs = corners(img)
	zoom_cnrs = zoom_corners(img, percent_zoom)
	return warp(img, cnrs, zoom_cnrs)

#red,blue, green -scale
def color(img, color):
	"""returns img in red-, green-, or blue-scale depending on if color is 'red', 'blue', or 'green'"""
	colors = {"blue":0, "green":1, "red":2}
	index = colors.get(color)
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	red_img = np.zeros_like(img)
	red_img[:, :, index] = gray_img
	return red_img

#test/starter code
img_zoom = zoom(img, 40)
cv2.imshow('zoom', img_zoom)
cv2.waitKey(0)
