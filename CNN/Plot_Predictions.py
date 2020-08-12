import numpy as np
import gdal
import tqdm
import os
import json
import argparse
import time
import gdal, ogr, osr, os
import matplotlib.pyplot as plt

mapdir = f'/exports/eddie/scratch/s1937352' 

mapdir = f'/exports/csce/datastore/geos/groups/MSCGIS/s1937352/ResultsJuly' 
boundsfileP =f'/exports/csce/datastore/geos/groups/MSCGIS/s1937352/UgandaCensus/UgandaCensus.tif'

'''
This script reads in an specified dictionary and plots the predictions to the extent of a given Raster file 

Script by Johanna Kauffert
'''


def readCommands():
  '''
  Read commandline arguments
  '''
  p = argparse.ArgumentParser(description=("Get parameters for training the CNN"))
  p.add_argument("--pred_img", dest ="pred_img", type=str, default="pred.tif", help=("Specify a dictionary that will be used"))
  p.add_argument("--dict_pred", dest ="dict_pred", type=str, default="dictionary_p2_pred.json", help=("Specify a dictionary that will be used"))
  cmdargs = p.parse_args()
  return cmdargs


def array2raster(rasterfn,newRasterfn,array,r,c):
    #write array to raster given parameters of the intial raster
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    print(pixelWidth)
    pixelHeight = geotransform[5]
    print(pixelHeight)
    cols = raster.RasterXSize
    rows = raster.RasterYSize


    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, c, r, 1, gdal.GDT_Int16) 
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY - 1000, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.SetNoDataValue(np.nan)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def crop_up(x):
    '''
        function to find the greater next number
        which can be divided by 1000
    '''
    return int(1000 * (np.ceil(x / 1000)))

def crop_down(x):
    '''
        function to find the lower next number
        which can be divided by 1000
    '''
    return int(1000 * (np.floor(x / 1000)))

def get_bounds(ds):
    '''
        function to get xmin, xmax, ymin, ymax
        and translate them to number which are dividable by 1000
    '''
    gt = ds.GetGeoTransform()
    minx = crop_up(gt[0])
    maxy = crop_down(gt[3])                           
    miny = crop_up(maxy + gt[5] * ds.RasterYSize)
    maxx = crop_down(minx + gt[1] * ds.RasterXSize)
    return minx, miny, maxx, maxy, ds.RasterYSize, ds.RasterXSize


def plot(pred_img,dictionary):
    #open dictionary with predictions
    with open(dictionary) as json_file:
        dictTiles = json.load(json_file)
    # for the worldpop file open dictionary holding worldpop valuzes
    with open("/exports/csce/datastore/geos/groups/MSCGIS/s1937352/Eddiebackup/dictionary_pWorldPop.json") as json_file:
        dictTileswp = json.load(json_file)
    
    #open boundsfile (Raster Uganda) and get the extent of the raster as well as its columns and rows
    ds = gdal.Open(boundsfileP)
    minx, miny, maxx, maxy,rY, rX = get_bounds(ds)
    print(minx, miny, maxx, maxy,rY, rX)
    band = ds.GetRasterBand(1)
    uganda = band.ReadAsArray()
    total = 0
    columns = 0
    for x in range(minx, maxx, 1000):
        columns +=1
        for y in range(miny, maxy, 1000):
            total +=1
    rows = total / columns
    print(rows)
    #create empty list
    ugandapopList= []
    print(total)
    print(columns)

    with tqdm.tqdm(total=total) as bar:
        bar.set_description(f"Write Prediction: ")
        # loop over the extent in 1000 m steps
        for y in range(miny, maxy, 1000):
            for x in range(minx, maxx, 1000):    
                #check if the boundsfile holds data         
                v = max(
                    0,      
                    int(
                        gdal.Translate('',
                                        ds,
                                        projWin=[x, y, x + 1000, y - 1000],
                                        format='VRT').ReadAsArray()[0, 0]))

                if v > 0:
                    #create the key of the dictionary with the extent
                    xystring = f'{x}_{y}'
                    #check if the key is in the dictionary and if a prediction value is saved
                    if xystring in dictTiles:
                        if dictTiles[xystring].get("pred") != None:
                            # get the prediction value 
                            pred = dictTiles[xystring].get("pred")
                            #pred = dictTileswp[xystring].get("pred")

                            #append prediction value to list
                            ugandapopList.append(pred)
                        # else append np.nan to the list to get the same extent of the boundsfile    
                        else:
                            ugandapopList.append(np.nan)
                    else:
                        ugandapopList.append(np.nan)
                else:
                    ugandapopList.append(np.nan)
                bar.update()

    print(len(ugandapopList))
    #reshape list to array with specified rows and columns
    ugandaarray = np.array(ugandapopList)
    ugandaarray = ugandaarray.reshape(int(rows),int(columns))
    ugandaarray = np.flip(ugandaarray, axis=0)
    #write array to raster
    array2raster(boundsfileP,pred_img,ugandaarray,int(rows),int(columns))

def main():

    cmd=readCommands()
    dict_pred = cmd.dict_pred
    pred_img=cmd.pred_img
    dictionary = f'{mapdir}/{dict_pred}'
    pred_img = f'{mapdir}/{pred_img}'
    plot(pred_img,dictionary)

main()
