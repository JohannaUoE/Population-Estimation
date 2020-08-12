import numpy as np
import gdal, ogr, osr, os
import os


'''
Set an given value of no data to np.nan in a raster file

Raster reading and writing from:
#https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html

Script by Johanna Kauffer
'''


def raster2array(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    return band.ReadAsArray()

def getNoDataValue(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    return band.GetNoDataValue()

def array2raster(rasterfn,newRasterfn,array):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)  #gdal.GDT_Float32 #gdal.GDT_Int16
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.SetNoDataValue(np.nan)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()



rootdir = '/exports/csce/datastore/geos/groups/MSCGIS/s1937352/Sentinel1_Coherence/CoherenceTiffs'
# data has only one band
for image in os.listdir(rootdir):
    if image.endswith("nce.tif"):
        imagename = image[:-4]
        print(imagename)
        #input directory
        imgdir = f'{rootdir}/{image}'
        print(image)
        #outputdirecotry
        outimg = f"/exports/csce/datastore/geos/groups/MSCGIS/s1937352/Sentinel1_Coherence/Coherence_nan/{imagename}_nan.tif"
        

        # convert Raster to array
        rasterArray = raster2array(imgdir)
        #trabslate all 0 to np.nan
        rasterArray= np.where(rasterArray == 0, np.nan, rasterArray)


        # Write updated array to new raster
        array2raster(imgdir,outimg,rasterArray)
        
        #check if the new raster has nan values
        print(getNoDataValue(outimg))



