import cv2
import os

class Vidsplitter:
	def __init__(self, filename, frameskip = 4):
		self.video = cv2.VideoCapture(filename)
		self.factor = frameskip
	def split(self):
		try:
			if not os.path.exists('data'):
				os.makedirs('data')
		except OSError:
			print('Error creating directory of data')
		#TODO: check if last frame file exists to save computation?
		success, frame = self.video.read()
		currentFrame = 0
		while (success):
			success, frame = self.video.read()
			name = './data/' + str(currentFrame) + 'frame.jpg'
			if not os.path.isfile(name) and currentFrame % self.factor == 0:

				print('Creating...' + name)
				cv2.imwrite(name, frame)
			currentFrame += 1
		self.video.release()
		cv2.destroyAllWindows()
