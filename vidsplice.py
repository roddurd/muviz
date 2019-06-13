import cv2
import os

class Vidsplicer:
	def __init__(self, dir):
		self.directory = dir
	def join(self):
		sample_img = cv2.imread("output/0slapper.jpeg")
		height, width, _, = sample_img.shape	
		video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 6, (width, height))
		for _, _, files in os.walk(self.directory):
			for file in files:	
				image = cv2.imread("output/" + file)
				video.write(image)	
		cv2.destroyAllWindows()
		video.release()
