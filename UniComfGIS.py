import numpy as np
import cv2 as cv
from Visualization.visual import visual
from HandlingSI.handlingSI import handlingSI
from ExtractionPoints.extraction import extraction
from BuildDataset.buildDataset import buildDataset
#from Classification.Training import training
#from Classification.Inference import inference

path_to_SAR_VV = r"source\sar.tif"


handSI = handlingSI(path_to_SAR_VV)

obvisual = visual()
SARimage_visual = obvisual.contrast(handSI.SARimage,minvalue=0.0035,maxvalue=0.12,method=3)
ACM = obvisual.ColorMapMethod(id_colormap=1)
extractor = extraction(SARimage_visual,"cultivo","CIATCEBOLLA2.txt")
extractor.run(savePoints=False)
extractor.set_nameclass("surco")
extractor.run()


buildDS = buildDataset("dataset_v2_CEBOLLA", handSI.SARimage, handSI.raster, kernel = 3, points = extractor.pointers)


'''handSI = handlingSI(path_to_SAR_VV)
path_to_points = r"puntos.txt"
buildDS = buildDataset("dataset_v1", handSI.SARimage, handSI.raster, kernel = 21, points = None)
buildDS.get_points(path_to_points)'''







#handSI.saveSI(SARimage_visual,"Kuwait_contrast_improved.tif")
#handSI.saveImage(ACM,"Kuwait_contrast_improved.png")
'''cv.namedWindow('raster SAR',cv.WINDOW_NORMAL)
cv.imshow('raster SAR',SARimage_visual)
cv.namedWindow('raster SAR ACM',cv.WINDOW_NORMAL)
cv.imshow('raster SAR ACM',ACM)
cv.waitKey(0)'''
