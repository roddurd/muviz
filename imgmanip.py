import cv2 
import os
import numpy as np
import re
import random
img = cv2.imread("parrot.jpg")


colors = {
"ALICEBLUE" : (240, 248, 255),
"ANTIQUEWHITE" : (250, 235, 215),
"ANTIQUEWHITE4" : (139, 131, 120),
"AQUA" : (0, 255, 255),
"AQUAMARINE3" : (102, 205, 170),
"AQUAMARINE4" : (69, 139, 116),
"AZURE1" : (240, 255, 255),
"BANANA" : (227, 207, 87),
"BEIGE" : (245, 245, 220),
"BISQUE3" : (205, 183, 158),
"BISQUE4" : (139, 125, 107),
"BLACK" : (0, 0, 0),
"BLUE" : (0, 0, 255),
"BLUE2" : (0, 0, 238),
"BLUE4" : (0, 0, 139),
"BLUEVIOLET" : (138, 43, 226),
"BRICK" : (156, 102, 31),
"BROWN" : (165, 42, 42),
"BROWN1" : (255, 64, 64),
"BROWN4" : (139, 35, 35),
"BURLYWOOD" : (222, 184, 135),
"BURLYWOOD4" : (139, 115, 85),
"CADETBLUE" : (95, 158, 160),
"CADETBLUE1" : (152, 245, 255),
"CADETBLUE4" : (83, 134, 139),
"CARROT" : (237, 145, 33),
"CHARTREUSE3" : (102, 205, 0),
"CHARTREUSE4" : (69, 139, 0),
"CHOCOLATE" : (210, 105, 30),
"CHOCOLATE1" : (255, 127, 36),
"CHOCOLATE4" : (139, 69, 19),
"COBALTGREEN" : (61, 145, 64),
"COLDGREY" : (128, 138, 135),
"CORAL" : (255, 127, 80),
"CORAL1" : (255, 114, 86),
"CORAL4" : (139, 62, 47),
"CORNSILK1" : (255, 248, 220),
"CORNSILK4" : (139, 136, 120),
"CRIMSON" : (220, 20, 60),
"CYAN4" : (0, 139, 139),
"DARKGOLDENROD" : (184, 134, 11),
"DARKGOLDENROD4" : (139, 101, 8),
"DARKGRAY" : (169, 169, 169),
"DARKGREEN" : (0, 100, 0),
"DARKOLIVEGREEN1" : (202, 255, 112),
"DARKOLIVEGREEN4" : (110, 139, 61),
"DARKORANGE1" : (255, 127, 0),
"DARKORANGE4" : (139, 69, 0),
"DARKORCHID1" : (191, 62, 255),
"DARKORCHID4" : (104, 34, 139),
"DARKSALMON" : (233, 150, 122),
"DARKSEAGREEN" : (143, 188, 143),
"DARKSEAGREEN1" : (193, 255, 193),
"DARKSLATEBLUE" : (72, 61, 139),
"DARKSLATEGRAY" : (47, 79, 79),
"DARKSLATEGRAY1" : (151, 255, 255),
"DARKSLATEGRAY4" : (82, 139, 139),
"DARKTURQUOISE" : (0, 206, 209),
"DARKVIOLET" : (148, 0, 211),
"DEEPPINK1" : (255, 20, 147),
"DEEPSKYBLUE2" : (0, 178, 238),
"DEEPSKYBLUE4" : (0, 104, 139),
"DIMGRAY" : (105, 105, 105),
"DODGERBLUE1" : (30, 144, 255),
"DODGERBLUE2" : (28, 134, 238),
"FIREBRICK4" : (139, 26, 26),
"FLESH" : (255, 125, 64),
"GAINSBORO" : (220, 220, 220),
"GHOSTWHITE" : (248, 248, 255),
"GOLD2" : (238, 201, 0),
"GOLD4" : (139, 117, 0),
"GOLDENROD" : (218, 165, 32),
"GOLDENROD4" : (139, 105, 20),
"GRAY" : (128, 128, 128),
"GRAY1" : (3, 3, 3),
"GRAY5" : (13, 13, 13),
"GRAY55" : (140, 140, 140),
"GRAY56" : (143, 143, 143),
"HONEYDEW2" : (224, 238, 224),
"HOTPINK3" : (205, 96, 144),
"HOTPINK4" : (139, 58, 98),
"INDIANRED" : (176, 23, 31),
"INDIANRED1" : (255, 106, 106),
"INDIANRED2" : (238, 99, 99),
"INDIGO" : (75, 0, 130),
"IVORY4" : (139, 139, 131),
"IVORYBLACK" : (41, 36, 33),
"KHAKI2" : (238, 230, 133),
"LAVENDER" : (230, 230, 250),
"LAWNGREEN" : (124, 252, 0),
"LEMONCHIFFON1" : (255, 250, 205),
"LIGHTBLUE" : (173, 216, 230),
"LIGHTBLUE1" : (191, 239, 255),
"LIGHTBLUE4" : (104, 131, 139),
"LIGHTCYAN1" : (224, 255, 255),
"LIGHTGOLDENROD1" : (255, 236, 139),
"LIGHTGOLDENRODYELLOW" : (250, 250, 210),
"LIGHTGREY" : (211, 211, 211),
"LIGHTPINK1" : (255, 174, 185),
"LIGHTSALMON1" : (255, 160, 122),
"LIGHTSALMON4" : (139, 87, 66),
"LIGHTSEAGREEN" : (32, 178, 170),
"LIGHTSKYBLUE" : (135, 206, 250),
"LIGHTSKYBLUE1" : (176, 226, 255),
"LIGHTSKYBLUE4" : (96, 123, 139),
"LIGHTSLATEBLUE" : (132, 112, 255),
"LIGHTSLATEGRAY" : (119, 136, 153),
"LIGHTSTEELBLUE" : (176, 196, 222),
"LIGHTSTEELBLUE4" : (110, 123, 139),
"LIGHTYELLOW1" : (255, 255, 224),
"LIGHTYELLOW4" : (139, 139, 122),
"LIMEGREEN" : (50, 205, 50),
"LINEN" : (250, 240, 230),
"MAGENTA" : (255, 0, 255),
"MAGENTA2" : (238, 0, 238),
"MAROON" : (128, 0, 0),
"MAROON1" : (255, 52, 179),
"MAROON4" : (139, 28, 98),
"MEDIUMORCHID1" : (224, 102, 255),
"MEDIUMPURPLE" : (147, 112, 219),
"MEDIUMPURPLE4" : (93, 71, 139),
"MEDIUMSPRINGGREEN" : (0, 250, 154),
"MEDIUMVIOLETRED" : (199, 21, 133),
"MIDNIGHTBLUE" : (25, 25, 112),
"MINT" : (189, 252, 201),
"MINTCREAM" : (245, 255, 250),
"MISTYROSE2" : (238, 213, 210),
"MISTYROSE4" : (139, 125, 123),
"MOCCASIN" : (255, 228, 181),
"NAVAJOWHITE1" : (255, 222, 173),
"NAVY" : (0, 0, 128),
"ORANGE" : (255, 128, 0),
"ORANGE1" : (255, 165, 0),
"ORANGERED1" : (255, 69, 0),
"ORCHID" : (218, 112, 214),
"ORCHID1" : (255, 131, 250),
"PALEGOLDENROD" : (238, 232, 170),
"PALEGREEN4" : (84, 139, 84),
"PALETURQUOISE1" : (187, 255, 255),
"PALEVIOLETRED" : (219, 112, 147),
"PALEVIOLETRED4" : (139, 71, 93),
"PAPAYAWHIP" : (255, 239, 213),
"PEACHPUFF2" : (238, 203, 173),
"PEACHPUFF4" : (139, 119, 101),
"PEACOCK" : (51, 161, 201),
"PINK" : (255, 192, 203),
"PINK1" : (255, 181, 197),
"PLUM" : (221, 160, 221),
"PLUM1" : (255, 187, 255),
"POWDERBLUE" : (176, 224, 230),
"PURPLE1" : (155, 48, 255),
"PURPLE4" : (85, 26, 139),
"RASPBERRY" : (135, 38, 87),
"RED1" : (255, 0, 0),
"ROSYBROWN" : (188, 143, 143),
"ROYALBLUE" : (65, 105, 225),
"ROYALBLUE1" : (72, 118, 255),
"SALMON1" : (255, 140, 105),
"SALMON4" : (139, 76, 57),
"SANDYBROWN" : (244, 164, 96),
"SAPGREEN" : (48, 128, 20),
"SEAGREEN4" : (46, 139, 87),
"SEASHELL4" : (139, 134, 130),
"SEPIA" : (94, 38, 18),
"SGICHARTREUSE" : (113, 198, 113),
"SGIGRAY12" : (30, 30, 30),
"SKYBLUE" : (135, 206, 235),
"SKYBLUE1" : (135, 206, 255),
"SLATEBLUE1" : (131, 111, 255),
"SLATEGRAY" : (112, 128, 144),
"SNOW1" : (255, 250, 250),
"SPRINGGREEN" : (0, 255, 127),
"SPRINGGREEN1" : (0, 238, 118),
"STEELBLUE1" : (99, 184, 255),
"TAN" : (210, 180, 140),
"TEAL" : (0, 128, 128),
"THISTLE2" : (238, 210, 238),
"THISTLE3" : (205, 181, 205),
"TOMATO1" : (255, 99, 71),
"TOMATO3" : (205, 79, 57),
"TURQUOISE1" : (0, 245, 255),
"TURQUOISEBLUE" : (0, 199, 140),
"VIOLETRED" : (208, 32, 144),
"VIOLETRED4" : (139, 34, 82),
"WHEAT2" : (238, 216, 174),
"WHEAT4" : (139, 126, 102),
"WHITE" : (255, 255, 255),
"WHITESMOKE" : (245, 245, 245),
"YELLOW1" : (255, 255, 0)}

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
        """returns img in red-, green-, or blue-scale depending on if color is 'red', 'blue', or 'green'"""
        colors = {"blue":0, "green":1, "red":2}
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
        imgcopy = img.copy()
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

def faded(img):
	img = img.copy()
	img2 = img.copy()
	img2 = zoom(img2, 20)
	alpha = 0.4
	cv2.addWeighted(img2, alpha, img, 1-alpha, 0, img)
	return img

#for file sorting
def atoi(text):
        return int(text) if text.isdigit() else text
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in numerical order
    Otherwise, files are sorted like [1, 11, 2, 22, 3, etc]
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ] 


#test/starter code
beat_skip = 4

proj_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(proj_dir, "data")
output_dir = os.path.join(proj_dir, "output")

img = faded(img)
cv2.imshow('timg', img)
cv2.waitKey(0)

"""
for _, _, files in os.walk(output_dir):
        files.sort(key=natural_keys)
        for i, file in enumerate(files):
                if not i%beat_skip:
                        print(file)
                        imgo = cv2.imread("output/"+file)
                        fracture(imgo)
                        cv2.imwrite("output/"+file,imgo)
from vidsplice import Vidsplicer

vid = Vidsplicer(output_dir, fps=6)
vid.join()                                      

img_zoom = zoom(img, 40)
cv2.imshow('zoom', img_zoom)
cv2.waitKey(0)
"""






