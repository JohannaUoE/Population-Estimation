#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import gdal
import glob
import tqdm
import os
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from constants import tiledir, mapdir
import json


def crop_up(x):
    return int(1000 * (np.ceil(x / 1000)))


def crop_down(x):
    return int(1000 * (np.floor(x / 1000)))


def get_bounds(ds):
    gt = ds.GetGeoTransform()
    minx = crop_up(gt[0])
    maxy = crop_down(gt[3])
    miny = crop_up(maxy + gt[5] * ds.RasterYSize)
    maxx = crop_down(minx + gt[1] * ds.RasterXSize)
    return minx, miny, maxx, maxy


def map_pop_to_img(pop, img_arrs, imgdir, uk_bounds, dictTiles, show=False):
    """
        This function reads an image and tiles it into 1km2 tiles. The extent and the corresonding 
        Population is read into a dictionary.  
    """

    if "VV" in imgdir:
        bands = ['vv']
    elif "VH" in imgdir:
        bands = ['vh']    
    elif "S2" in imgdir:
        bands = ['S2']
        print("S2")
    else:
        print("SAD! :( {}".format(imgdir))
        exit()
    

    minx, miny, maxx, maxy = get_bounds(img_arrs[0])
    print(minx, miny, maxx, maxy)
    print(maxy-miny)
    print(maxx-minx)
    total = 0
    for x in range(minx, maxx, 1000):
        for y in range(miny, maxy, 1000):
            total +=1

    for i, img in enumerate(img_arrs):
        print(i, img)
        with tqdm.tqdm(total=total) as bar:
            bar.set_description(f"Band: {bands[i]}")
            for x in range(minx, maxx, 1000):
                for y in range(miny, maxy, 1000):
                    land = max(
                        0,
                        int(
                            gdal.Translate('',
                                            uk_bounds,
                                            projWin=[x, y, x + 1000, y - 1000],
                                            format='VRT').ReadAsArray()[0, 0]))
                    
                    if not land:
                        bar.update(1)
                        continue
                    
                    v = max(
                        0,
                        int(
                            gdal.Translate('',
                                            pop,
                                            projWin=[x, y, x + 1000, y - 1000],
                                            format='VRT').ReadAsArray()[0, 0]))
                    
                    tile = gdal.Translate('',
                                            img,
                                            projWin=[x, y, x + 1000, y - 1000],
                                            format='VRT')

                    tile_arr = tile.ReadAsArray() 

                    #check if there is actually data in there, since S1 images are tilted
                    complete_tile = (not np.all(tile_arr == tile_arr[0, 0])
                                    ) and np.max(tile_arr) < 65534 and np.min(tile_arr) > -99  #what is this number

                    if complete_tile:
                        xystring = f'{x}_{y}'
                        if xystring not in dictTiles:
                            dictTiles[xystring] = {"extent": [x, y, x + 1000, y - 1000],
                                                    "pop": v,
                                                    bands[i]: imgdir}
                        elif xystring in dictTiles:
                            if dictTiles[xystring].get(bands[i]) == None: 
                                dictTiles[xystring].update({bands[i]: imgdir})
                            else:
                                continue
                    bar.update(1)
    return dictTiles                   
                        

def convert(dir,name):
    extension = '.tiff'
    outName=f'{mapdir}/UgandaCensus/{name}_P.tiff'
    # EPSG to project oo
    outEPSG='21036'
    # reproject to new file (could output to an object instead)
    gdal.Warp(outName,dir,dstSRS='EPSG:'+outEPSG)

def convert_coord_systems(imgdir):
    dstSRS = '+proj=utm +zone=36 +south +ellps=clrk80 +towgs84=-160,-6,-302,0,0,0,0 +units=m +no_defs '

    converted = []
    ds = gdal.Warp('',
                imgdir,
                targetAlignedPixels=True,
                xRes=10,
                yRes=10,
                dstSRS='EPSG: 21036',
                dstNodata=65534,
                format='VRT')
    converted.append(ds)
    print("Converted")

    return converted
    
def main():
    popfile = f'{mapdir}/UgandaCensus/UgandaCen1000.tif'
    boundsfile = f'{mapdir}/UgandaCensus/UgandaRaster.tif'
    imgdirs = f'{mapdir}SentinelCNN'
    boundsname = boundsfile[66:-4]
    popname = popfile[66:-4]
    print(popname)
    print(boundsname)
    #convert(popfile,popname)
    #convert(boundsfile,boundsname)
    popfileP = f'{mapdir}/UgandaCensus/{popname}_P.tiff'
    boundsfileP =f'{mapdir}/UgandaCensus/{boundsname}_P.tiff'
    pop = gdal.Open(popfileP)
    uganda_bounds = gdal.Open(boundsfileP)



    dictTiles = {}
    for idx, imgdir in enumerate(os.listdir(imgdirs)):
        if imgdir.endswith('.tif'):
            imgdir = f'{imgdirs}/{imgdir}'
            print(imgdir)
            tifname = imgdir[64:-4]
            print(tifname)
            print("Processing image {} of {} ({} bands)".format((idx + 1), len(imgdirs),
                                                                2 if "S1B" in imgdir else 3))
            # possibly not using this bc I want ot keep everything in 4326                                                      
            arss = convert_coord_systems(imgdir)
            dictTiles = map_pop_to_img(pop, arss, imgdir, uganda_bounds, dictTiles, show=False)
            



    with open('dictionary.json', 'w') as fp:
        json.dump(dictTiles, fp)         

if __name__ == "__main__":
    main()
