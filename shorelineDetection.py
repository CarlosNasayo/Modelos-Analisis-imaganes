from Measure.measure import measure

path_to_SAR_VV_bef = r"D:\UNICOMFACAUCA_2020_II\Proyectos_DIMAR\ShoreLineProject\results\ClassificationCartagena2019f.tif"
path_to_SAR_VV_aft = r"D:\UNICOMFACAUCA_2020_II\Proyectos_DIMAR\ShoreLineProject\results\ClassificationCartagena2020f.tif"
m = measure()
#m.shoreline_detection(path_to_SAR_VV)
m.calc_afectacion(path_to_SAR_VV_bef, path_to_SAR_VV_aft)