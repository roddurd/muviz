import cv2 
import os
import numpy as np
import re
import random
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

#img fracture effect
def fracture(img):
	red=(0,0,255)
	green=(0,255,0)
	blue=(255,0,0)
	height, width, _ = img.shape
	center = (width/2, height/2)
	polygons = []
	for i in range(4):
		left=[0,random.randint(20,height-20)]#left
		top=[random.randint(20,width-20),0]#top
		right=[width,random.randint(20,height-20)]#right
		bottom=[random.randint(20,width-20),height]#bottom
		polygons.append([left,top,center,[0,0]])
		polygons.append([left,bottom,center,[0,height]])
		polygons.append([right,top,center,[width,0]])
		polygons.append([right,bottom,center,[width,height]])
	for polygon in polygons:
		clr = random.choice([red, green, blue])
		cv2.fillPoly(img, np.int32([polygon]),clr) 	
			


#for file sorting
def atoi(text):
	try:
		return int(text)
	except:
		return text
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    Otherwise, files are sorted like [1, 11, 2, 22, 3, etc]
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ] 


#test/starter code
beat_skip = 4

proj_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(proj_dir, "data")
output_dir = os.path.join(proj_dir, "output")

"""
for _, _, files in os.walk(output_dir):
	files.sort(key=natural_keys)
	for i, file in enumerate(files):
		if not i%beat_skip:
			print(file)
			imgo = cv2.imread("output/"+file)
			fracture(imgo)
			cv2.imwrite("output/"+file,imgo)
"""

from vidsplice import Vidsplicer

vid = Vidsplicer(output_dir)
vid.join()					
"""	
img_zoom = zoom(img, 40)
cv2.imshow('zoom', img_zoom)
cv2.waitKey(0)
"""






