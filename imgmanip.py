import cv2 
import os
import numpy as np
import re
import random
import sys
from colors import colors
img = cv2.imread("pfp3.jpg")



def blur(img, intensity):
    img = img.copy()
    intensity += 1 if not intensity%2 else 0
    cv2.GaussianBlur(img, (intensity, intensity), 3, img)
    return img
def rotate(img, theta):
    img = img.copy()
    height, width, _ = img.shape
    M = cv2.getRotationMatrix2D((width/2, height/2), theta,1)
    img = cv2.warpAffine(img, M, (width, height))
    return img      

#red,blue, green -scale
def color(img, color):
    colors = {"blue":0, "green":1, "red":2}
    if color == 'random':
        color = random.choice(list(colors.keys()))
        print("color: ", color)
    index = colors.get(color)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clr_img = np.zeros_like(img)
    clr_img[:, :, index] = gray_img
    return clr_img

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

#img fracture effect
def fracture(img):
    """adds a slightly transparent set of polygons which all connect to the center, and each have a different color"""
    img = img.copy()
    red=(0,0,255)
    green=(0,255,0)
    blue=(255,0,0)
    height, width, _ = img.shape
    center = (width/2, height/2)
    polygons = []
    for i in range(10):
        left=[0,random.randint(20,height-20)]#left
        top=[random.randint(20,width-20),0]#top
        right=[width,random.randint(20,height-20)]#right
        bottom=[random.randint(20,width-20),height]#bottom
        polygons.append([left,center,top,[0,0]])
        polygons.append([left,center,bottom,[0,height]])
        polygons.append([right,center,top,[width,0]])
        polygons.append([right,center,bottom,[width,height]])
    for polygon in polygons:
        clr = random.choice(list(colors.values()))
        cv2.fillPoly(imgcopy, np.int32([polygon]),clr)  
        alpha = 0.5
        cv2.addWeighted(imgcopy, alpha, img, 1-alpha, 0, img)   
    return img

def circle(img, center, radius, color='random',thickness=5,alpha=1):
    img = img.copy()
    img_copy = img.copy() #second copy for adding weighted onto img; for transparency
    if color=='random':
        clr = random.choice(list(colors.values()))
    else:
        clr = colors.get(color.upper())
    cv2.circle(img_copy, center, radius, clr,thickness) 
    cv2.addWeighted(img_copy,alpha,img,1-alpha,0,img)
    return img

def tesselate(img, center, radius):
    img = img.copy()
    for i in range(-radius, radius):
        for j in range(-radius, radius):
            for k in range(5):
                img[center[0]+radius+i][center[1]+k*radius+j]=img[center[0]-radius+i][center[1]-radius+j]
                return img      

def brighten(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def darken(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v[v <= value] = 0
    v[v > value] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def faded(img,percent_zoom):
    img = img.copy()
    img2 = img.copy()
    img2 = zoom(img2, percent_zoom)
    alpha = 0.4
    cv2.addWeighted(img2, alpha, img, 1-alpha, 0, img)
    return img
def shuffle(img, percent):
    #Shuffle the rows of pixels in the image. Specify the percentage of shuffling from 1-100.
    skip = 100 - percent
    img = img.copy()
    height, width, _ = img.shape
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            if not j % skip and i % percent :
              #img[random.randint(0,height-1),:] = img[i, :]
              img[i, :] = img[random.randint(0,height-1),:]
    return img

def horizontal_glitch(img, bands, coords=None):
#coords for the glitch bands can be passed, so that subsequent glitches are nearby.
    img = img.copy()
    height, width, _ = img.shape
    if coords:
        coords = [x+5 if x+5< height else round(x-(height/2)) for x in coords]
    else:
        coords = [random.randint(20,height-20) for _ in range(bands)]
    for x in coords:
        for i in range(width):
            if i-50 < 0:
                img[x:x+10,i] = img[x:x+10,i+50]
            else:
                img[x:x+10,i-50] = img[x:x+10,i]
                img[x:x+10,i] = img[x:x+10,width-i]
    return img, xcoords

def scramble(img, center, radius):
#scramble all of the pixels within a certain square radius of the given center.
    img = img.copy()
    columns = list(range(center[0]-radius, center[0]+radius)) 
    rows = list(range(center[1]-radius, center[1]+radius))
    for c in columns :
        for r in rows :
            img[c][r] = img[random.choice(columns)][random.choice(rows)]
    return img


def vert_offset(img, offset):
#vertical offset; shift all pixels up by offset
#(there will be a horizontal line across the screen where the frame is stitched at its ends).
    imgo = img.copy()
    height, width, _ = img.shape
    offset = offset % height #in case offset is too high, just mod it
    for r in range(height):
        if r >= offset:
            imgo[:][r] = img[:][r-offset]
        else:
            imgo[:][r] = img[:][height-offset+r]   
    return imgo
    

def hor_offset(img, offset):
#horizontal offset; shift all pixels right by offset
#(there will be a vertical line across the screen where the frame is stitched at its ends).
    imgo = img.copy()
    height, width, _ = img.shape
    offset = offset % width #in case offset is too high, just mod it
    for c in range(width):
        if c >= offset:
            imgo[:,c] = img[:,c-offset]
        else:
            imgo[:,c] = img[:,width-offset+c]
    return imgo

#for file sorting
def atoi(text):
    return int(text) if text.isdigit() else False
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in numerical order
    Otherwise, files are sorted like [1, 11, 2, 22, 3, etc]
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ] 

#test/starter code

proj_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(proj_dir, "data")
output_dir = os.path.join(proj_dir, "output")


"""
from vidsplit import Vidsplitter
vid = Vidsplitter("video.mp4", 6)
vid.split()
"""
"""  FOR DOING SEQUENTIAL STUFF ON IMAGES IN data_dir
print("working...")
sys.stdout.flush()
for _, _, files in os.walk(data_dir):
    files.sort(key=natural_keys)
    skip = 1
    offset = 50 
    for i, file in enumerate(files):
        img = cv2.imread('data/' + file)
        img = hor_offset(img, offset)
        cv2.imwrite("output/" + file, img)
        offset += 150
print("done!")

"""
zimg = vert_offset(img,370)
cv2.imshow('offset', zimg)

cv2.waitKey(0)
"""
print("joining...")
sys.stdout.flush()
from vidsplice import Vidsplicer
vid = Vidsplicer(output_dir, 24)
vid.join()
print("done.")
"""





