import numpy as np
import cv2 as cv
from Visualization.visual import visual
from HandlingSI.handlingSI import handlingSI
from Classification.Inference.inference import inference

'''path_to_SAR_VV = r"source\Kuwait_SAR_lite.tif"
model_dir = r"Results\DATASET_KUWAIT_model_odf.h5"
name_of_classification = "ClassificationKuwait_2020-11-13.tif"'''

path_to_SAR_VV = r"D:\UNICOMFACAUCA_2020_II\Proyectos_DIMAR\ShoreLineProject\source\Shoreline_roi_Cartagena_2020.tif"
model_dir = r"D:\UNICOMFACAUCA_2020_II\Proyectos_DIMAR\ShoreLineProject\model\Cartagena_model_sld.h5"
name_of_classification = "ClassificationCartagena2020.tif"

handSI = handlingSI(path_to_SAR_VV)

lbldic = {
		"continental":0,
		"ocean":1
	}
nameclasses = [255,0]

tester = inference(image=handSI.SARimage, 
	rasterfromgdal=handSI.raster, 
	path_to_model=model_dir, 
	path_to_result=name_of_classification, 
	NoDataValue=-32768, 
	kernel=7, 
	lbl=lbldic,
	colors=nameclasses)
tester.test_mod2()