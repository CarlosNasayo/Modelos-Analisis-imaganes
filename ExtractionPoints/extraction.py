import numpy as np 
import cv2 as cv
from HandlingSI.utils import utils

class extraction(object):
	def __init__(self,image,nameclass,path_to_csv):
		self.image = image
		self.nameclass = nameclass
		self.path_to_csv = path_to_csv
		self.pointers = []
		if(len(self.image.shape) < 3):
			self.image = cv.cvtColor(self.image, cv.COLOR_GRAY2BGR)

	def run(self,savePoints=None):
		if savePoints is None:
			self.markpoints()
		else:
			self.markpoints(savePoints)

	def set_points(self, points):
		self.pointers = points

	def set_nameclass(self, nameclass):
		self.nameclass = nameclass
	
	def markpoints(self, savePoints=True):
		cv.namedWindow('markpoints',cv.WINDOW_NORMAL)
		cv.setMouseCallback('markpoints', self.click_and_crop)
		while True:
			cv.namedWindow('markpoints',cv.WINDOW_NORMAL)
			cv.imshow('markpoints', self.image)
			key = cv.waitKey(1) & 0xFF
			if key == ord("q"):
				break
		cv.destroyAllWindows()
		if savePoints == True:
			self.save_pointers()

	def click_and_crop(self, event, x, y, flags, param):
		if event == cv.EVENT_LBUTTONDOWN:
			refPt = [x, y, self.nameclass]
			self.pointers.append(refPt)
			cv.circle(self.image,(x,y),1,(0,0,255),-1)

	def save_pointers(self):
		log = utils(self.path_to_csv)
		log.write_pointers(self.pointers)