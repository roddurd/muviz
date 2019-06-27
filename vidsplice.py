import cv2
import os
import re

def atoi(text):
	return int(text) if text.isdigit() else text
def natural_keys(text):
	return [atoi(c) for c in re.split(r'(\d+)', text)]


class Vidsplicer:
	def __init__(self, dir, fps=24):
		self.directory = dir
		self.FPS = fps
	def join(self):
		sample_img = cv2.imread("output/0slapper.jpg")
		height, width, _, = sample_img.shape	
		video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'DIVX'), self.FPS, (width, height))
		for _, _, files in os.walk(self.directory):
			files.sort(key=natural_keys)
			for file in files:	
				image = cv2.imread("output/"+file)
				video.write(image)	
		cv2.destroyAllWindows()
		video.release()
