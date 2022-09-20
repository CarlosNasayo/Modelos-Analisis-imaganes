import numpy as np
import cv2 as cv
import os
from osgeo import gdal
'''from Visualization.visual import visual
from HandlingSI.handlingSI import handlingSI
from ExtractionPoints.extraction import extraction
from BuildDataset.buildDataset import buildDataset'''

#path_to_SAR_VV = r"source\sample.tif"
path_to_SAR_VV = r"source\sar.tif"

'''handSI = handlingSI(path_to_SAR_VV)

print(type(handSI.SARimage.shape))'''

in_ds = gdal.Open(path_to_SAR_VV)
print(in_ds.RasterCount)
in_band = in_ds.GetRasterBand(1)
in_data = in_band.ReadAsArray()
print(in_data.shape)