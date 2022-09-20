from HandlingSI.utils import utils
from HandlingSI.handlingSI import handlingSI
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2 as cv
from osgeo import gdal
import time
np.seterr(divide='ignore', invalid='ignore')

class inference(object):
	def __init__(self, image, rasterfromgdal, path_to_model, path_to_result, NoDataValue, kernel, lbl, colors):
		self.image = image
		self.rasterfromgdal = rasterfromgdal
		self.path_to_model = path_to_model
		self.path_to_result = path_to_result
		self.kernel = kernel
		self.result_image = None
		self.NoDataValue = NoDataValue
		self.obutil = utils()
		self.obhand = handlingSI()
		self.obhand.set_raster(self.rasterfromgdal)
		self.lbl = lbl
		self.colors = colors

	def test_mod1(self):
		height = self.image.shape[0]
		width = self.image.shape[1]
		print(height,width)
		total_px = height*width
		self.result_image = np.ones((height,width), np.float32)*(self.NoDataValue)
		model = load_model(self.path_to_model, compile=False)
		delta_kernel=int(np.trunc(self.kernel/2))
		proceso=(height-2*delta_kernel)*(width-2*delta_kernel)
		cont=0
		self.obutil.printProgressBar(0, total_px, prefix = 'Progress:', suffix = 'Complete', length = 50)
		for i in range(delta_kernel,height-delta_kernel):
			for j in range(delta_kernel,width-delta_kernel):
				refPt = (j, i)
				sample=self.obutil.set_ROI(self.image,refPt,delta_kernel)
				sample = img_to_array(sample)
				sample = np.expand_dims(sample, axis=0)
				prediction = model.predict(sample)[0]
				value_predicted = self.colors[np.argmax(prediction)]
				self.result_image.itemset((i,j),value_predicted)
				time.sleep(0.000001)
				self.obutil.printProgressBar(cont, total_px, prefix = 'Progress:', suffix = 'Complete', length = 50)
				cont = cont + 1
		self.obhand.saveSI(image=self.result_image,outFileName=self.path_to_result,NoDataValue=-32768)

	def test_mod2(self):
		height = self.image.shape[0]
		width = self.image.shape[1]
		total_px = height*width
		self.result_image = np.zeros((height,width), np.float32)
		model = load_model(self.path_to_model, compile=False)
		delta_kernel=int(np.trunc(self.kernel/2))
		proceso=(height-2*delta_kernel)*(width-2*delta_kernel)
		cont=0
		self.obutil.printProgressBar(0, total_px, prefix = 'Progress:', suffix = 'Complete', length = 50)
		for i in range(delta_kernel,height-delta_kernel):
			for j in range(delta_kernel,width-delta_kernel):
				if self.image.item(i,j) == -32768:
					self.result_image.itemset((i,j),-32768)
				else:
					refPt = (j, i)
					sample=self.obutil.set_ROI(self.image,refPt,delta_kernel)
					sample = img_to_array(sample)
					sample = np.expand_dims(sample, axis=0)
					prediction = model.predict(sample)[0]
					value_predicted = self.colors[np.argmax(prediction)]
					self.result_image.itemset((i,j),value_predicted)
				time.sleep(0.000001)
				self.obutil.printProgressBar(cont, total_px, prefix = 'Progress:', suffix = 'Complete', length = 50)
				cont = cont + 1
		self.obhand.saveSI(image=self.result_image,outFileName=self.path_to_result,NoDataValue=-32768)