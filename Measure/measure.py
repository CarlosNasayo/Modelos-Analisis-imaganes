import numpy as np
import cv2 as cv
from Visualization.visual import visual
from HandlingSI.handlingSI import handlingSI

class measure(object):
	def __init__(self):
		self.afectacion = None
		self.shorelineBef = None
		self.shorelineAft = None
	def shoreline_detection(self,path_to_image):
		path_to_SAR_VV = path_to_image
		handSI = handlingSI(path_to_SAR_VV)
		obvisual = visual()
		imgconv = obvisual.contrast(handSI.SARimage,minvalue=None,maxvalue=None,method=1)
		imgconv = cv.Canny(imgconv,100,200)
		cv.namedWindow('conv',cv.WINDOW_NORMAL)
		cv.imshow("conv",imgconv)
		cv.waitKey(0)
	def calc_afectacion(self, path_to_image_bef, path_to_image_aft):
		path_to_SAR_VV_bef = path_to_image_bef
		path_to_SAR_VV_aft = path_to_image_aft

		handSI_bef = handlingSI(path_to_SAR_VV_bef)
		img_bef = handSI_bef.SARimage

		handSI_aft = handlingSI(path_to_SAR_VV_aft)
		img_aft = handSI_aft.SARimage
		height = img_bef.shape[0]
		width = img_bef.shape[1]
		self.afectacion = np.zeros((height,width), np.float32)
		area_perdida = 0
		area_ganada = 0
		for i in range(height):
			for j in range(width):
				value = img_bef.item(i,j) - img_aft.item(i,j)
				if(value==255):
					self.afectacion.itemset((i,j),255)
					area_perdida = area_perdida + 1
				elif(value == -255):
					self.afectacion.itemset((i,j),100)
					area_ganada = area_ganada + 1
		handSI_bef.saveSI(self.afectacion,"afectacion_2019-2020.tif",NoDataValue = 0)
		size_px = 0.01 #10m - 0.01Km
		total_area_perdida = size_px*size_px*area_perdida
		print("Área perdida de costa: ",total_area_perdida," Km2")
		total_area_ganada = size_px*size_px*area_ganada
		print("Área ganada de costa: ",total_area_ganada," Km2")