''' This script reprojects the data to the specified coordinate system
    and normalises the resolution to 10 * 10 mean.
    The resampling is done per default with a nearest neighbour interpolation

    Script by Johanna Kauffert
'''

#import packages
import gdal
import numpy as np
import os
import time 

#specify directories of images
dirS2 = "/exports/eddie/scratch/s1937352/Sentinel2_1"
dirVV = "/exports/eddie/scratch/s1937352/VV_Backscatter"
dirVH = "/exports/eddie/scratch/s1937352/VH_Backscatter"
dirCoh = "/exports/eddie/scratch/s1937352/Coherence"

#specify out directories
outdirS2 = "/exports/eddie/scratch/s1937352/Sentinel2_1_p"
outdirVV = "/exports/eddie/scratch/s1937352/VV_Backscatter_p"
outdirVH = "/exports/eddie/scratch/s1937352/VH_Backscatter_p"
outdirCoh = "/exports/eddie/scratch/s1937352/Coherence_p"

overall = time.time()

#loop over all foldern and apply the gdal warp function

for idx, imgdir in enumerate(os.listdir(dirVV)):
    starttime = time.time()
    imgname = imgdir[:-4]
    gdal.Warp(f'{outdirVV}/{imgname}_p.tif',
            f'{dirVV}/{imgdir}',
            targetAlignedPixels=True,
            xRes=10,
            yRes=10,
            dstSRS='EPSG: 21036',
            dstNodata=65534)
    process = time.time() - starttime
    print(f"Image {imgname} was warped and took {process} seconds")

for idx, imgdir in enumerate(os.listdir(dirVV)):
    starttime = time.time()
    imgname = imgdir[:-4]
    gdal.Warp(f'{outdirVV}/{imgname}_p.tif',
            f'{dirVV}/{imgdir}',
            targetAlignedPixels=True,
            xRes=10,
            yRes=10,
            dstSRS='EPSG: 21036',
            dstNodata=65534)
    process = time.time() - starttime
    print(f"Image {imgname} was warped and took {process} seconds")

for idx, imgdir in enumerate(os.listdir(dirVH)):
    starttime = time.time()
    imgname = imgdir[:-4]
    gdal.Warp(f'{outdirVH}/{imgname}_p.tif',
            f'{dirVH}/{imgdir}',
            targetAlignedPixels=True,
            xRes=10,
            yRes=10,
            dstSRS='EPSG: 21036',
            dstNodata=65534)
    process = time.time() - starttime
    print(f"Image {imgname} was warped and took {process} seconds")

for idx, imgdir in enumerate(os.listdir(dirCoh)):
    starttime = time.time()
    imgname = imgdir[:-4]
    gdal.Warp(f'{outdirCoh}/{imgname}_p.tif',
            f'{dirCoh}/{imgdir}',
            targetAlignedPixels=True,
            xRes=10,
            yRes=10,
            dstSRS='EPSG: 21036',
            dstNodata=65534)
    process = time.time() - starttime
    print(f"Image {imgname} was warped and took {process} seconds")

time = (time.time()-overall)/60
print(f'Finished all scenes in {time} minutes')