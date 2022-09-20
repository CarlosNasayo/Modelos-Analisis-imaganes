from HandlingSI.utils import utils
from HandlingSI.handlingSI import handlingSI
import os
import numpy as np
import cv2 as cv

class buildDataset(object):
	def __init__(self, path_to_dataset, image, raster_from_gdal, kernel = 7, points = None):
		self.path_to_dataset = path_to_dataset
		self.image = image
		self.raster = raster_from_gdal
		self.kernel = kernel
		if points is None:
			self.points = None
		else:
			self.points = points
			self.build()

	def get_points(self, path_to_csv):
		obutils = utils(path_to_csv)
		self.points = obutils.read_pointers()
		for i in range(len(self.points)):
			self.points[i][0] = int(self.points[i][0])
			self.points[i][1] = int(self.points[i][1])
		self.build()

	def build(self):
		obutils = utils()
		obhand = handlingSI()
		obhand.set_raster(self.raster)
		print("COMPROBAR: ",len(obhand.Bands))
		for i in range(len(self.points)):
			foldername = self.path_to_dataset+"/"+str(self.points[i][2])
			if not os.path.exists(foldername):
				os.makedirs(foldername)
			saved_id=foldername+"/"+str(self.points[i][2])+"_"+str(self.points[i][0])+str(self.points[i][1])+".tif"
			refPt = [self.points[i][0],self.points[i][1]]
			delta_kernel=int(np.trunc(self.kernel/2))
			if self.raster.RasterCount == 1:
				roi=obutils.set_ROI(self.image,refPt,delta_kernel)
				row=roi.shape[0]
				col=roi.shape[1]
				if row == col and row == self.kernel:
					print(roi.shape)
					obhand.saveSI(roi,saved_id,NoDataValue = -32768)
				else:
					print("no guardado")
			else:
				multiband=cv.merge(obhand.Bands)
				roi=obutils.set_ROI(multiband,refPt,delta_kernel)
				row=roi.shape[0]
				col=roi.shape[1]
				print("ROI_size: ",roi.shape)
				if row == col and row == self.kernel:
					print(roi.shape)
					obhand.saveSI(roi,saved_id,NoDataValue = -32768)
				else:
					print("no guardado")
