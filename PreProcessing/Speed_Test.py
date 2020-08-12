import gdal
import numpy as np
import os
import tqdm
import time

'''
This script performs perfromance tests on different ways to read imagery into RAM

Script by Johanna Kauffert

'''

dirS1 = '/exports/csce/datastore/geos/groups/MSCGIS/s1937352/Speed_Test'

#reproject the given files for the speed test
def reporject():
    for idx, imgdir in enumerate(os.listdir(dirS1)):
        print(imgdir)
        imgname = imgdir[:-4]
        print(imgname)
        gdal.Warp(f'{dirS1}/{imgname}_p.tif',
                f'{dirS1}/{imgdir}',
                targetAlignedPixels=True,
                xRes=10,
                yRes=10,
                dstSRS='EPSG: 21036',
                dstNodata=65534)

#crop_up, crop_down and get_bounds functions are taken from the create_dictionary script
def crop_up(x):
    return int(1000 * (np.ceil(x / 1000)))


def crop_down(x):
    return int(1000 * (np.floor(x / 1000)))


def get_bounds(raster):
    ds = gdal.Open(raster)
    gt = ds.GetGeoTransform()
    minx = crop_up(gt[0])
    maxy = crop_down(gt[3])   
    miny = crop_up(maxy + gt[5] * ds.RasterYSize)  
    maxx = crop_down(minx + gt[1] * ds.RasterXSize)
    return minx, miny, maxx, maxy



def find_tiles():
    # get the coordinates of the smaller tile file to make it comparable
    for idx, imgdir in enumerate(os.listdir(dirS1)):
        if imgdir.endswith('Speed_p.tif'):
            print(imgdir)
            minx, miny, maxx, maxy = get_bounds(f'{dirS1}/{imgdir}')
            print(minx, miny, maxx, maxy)      
            print(maxy-miny)
            
            total = 0
            for x in range(minx, maxx, 1000):
                for y in range(miny, maxy, 1000):
                    total +=1
            
            with tqdm.tqdm(total=total) as bar:
                for x in range(minx, maxx, 1000):
                    for y in range(miny, maxy, 1000):
                        print(f'[{x}, {y}, {x + 1000}, {y - 1000}]')
            

def test_speed_onetile():
    #test spped in agiven extent for the small file and the whole file
    starts1 = time.time()
    tile = gdal.Translate('',
                            f'{dirS1}/S1a_Speed_p.tif',
                            projWin=[301000, 10306000, 302000, 10305000],
                            format='VRT')
    tile_arr = tile.ReadAsArray()
    print(f'small file: {time.time()-starts1}')

    starts1 = time.time()
    tile = gdal.Translate('',
                            f'{dirS1}/S1A_IW_GRDH_1SDV_20150913T161322_20150913T161347_007699_00AB01_15B6_vv_nan_p.tif',
                            projWin=[301000, 10306000, 302000, 10305000],
                            format='VRT')
    tile_arr = tile.ReadAsArray()
    print(f'whole file: {time.time()-starts1}')
    
    #test if Warp is faster than translate
    starts1 = time.time()
    tile = gdal.Warp('',
                            f'{dirS1}/S1a_Speed_p.tif',
                            outputBounds=[301000, 10306000, 302000, 10305000],
                            format='VRT')
    tile_arr = tile.ReadAsArray()
    print(f'small file WARP: {time.time()-starts1}')

    starts1 = time.time()
    tile = gdal.Warp('',
                            f'{dirS1}/S1A_IW_GRDH_1SDV_20150913T161322_20150913T161347_007699_00AB01_15B6_vv_nan_p.tif',
                            outputBounds=[301000, 10306000, 302000, 10305000],
                            format='VRT')
    tile_arr = tile.ReadAsArray()
    print(f'whole file WARP: {time.time()-starts1} ')

    #test how fast if the tile was already saved wih the given extent
    starts1 = time.time()
    tile = gdal.Open(f'{dirS1}/tile.tif')
    bands = tile.GetRasterBand(1)
    tile_arr = bands.ReadAsArray()
    print(f'tile file: {time.time()-starts1}')



test_speed_onetile()