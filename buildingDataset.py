import numpy as np
import cv2 as cv
from Visualization.visual import visual
from HandlingSI.handlingSI import handlingSI
from ExtractionPoints.extraction import extraction
from BuildDataset.buildDataset import buildDataset
from Classification.Training import training
from Classification.Inference import inference

#path_to_SAR_VV = r"source\Kuwait_SAR.tif"
#path_to_csv = "dataset_shoreline_kuwait_crop1.txt"
path_to_SAR_VV = r"D:\UNICOMFACAUCA_2020_II\Proyectos_DIMAR\ShoreLineProject\source\Shoreline_roi_Cartagena\zone01_20191106.tif"
path_to_csv = R"D:\UNICOMFACAUCA_2020_II\Proyectos_DIMAR\ShoreLineProject\dataset\dataset_shoreline_Cartagena_2019.txt"

handSI = handlingSI(path_to_SAR_VV)
buildDS = buildDataset("DATASET_SHORELINE", handSI.SARimage, handSI.raster, kernel = 7, points = None)
buildDS.get_points(path_to_csv)