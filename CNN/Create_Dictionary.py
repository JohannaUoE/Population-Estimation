'''
    This script generates a dictionary that is constructed like following:
    xmin_ymax : {'extent': [xmin, ymax, xmax, ymin],
                  'pop': number of population in the extent,
                  'vv': filepath to the vv image,
                  'vh': filepath to the vh image,
                  'S2': filepath to the S2 image,
                  'Coherence': filepath to the MLI image}

    Example:
    {"557000_10285000": {"extent": [557000, 10285000, 558000, 10284000],
                        "pop": 42, 
                        "S2": "/exports/eddie/scratch/s1937352/Sentinel2_1_p/UgandaS2_73.tif",
                        "vv": "/exports/eddie/scratch/s1937352/VV_Backscatter_p/S1A_IW_GRDH_1SDV_20150908T160508_20150908T160533_007626_00A907_5A06_vv_nan.tif",
                        "vh": "/exports/eddie/scratch/s1937352/VH_Backscatter_p/S1A_IW_GRDH_1SDV_20150908T160508_20150908T160533_007626_00A907_5A06_vh_nan.tif",
                        "Coherence": "/exports/eddie/scratch/s1937352/Coherence_p/20151002_20151026_5_coherenceMLI_Cl.tif"}

Script by Johanna Kauffert (initial implementation McCabe)
'''

#import necessary modules
import numpy as np
import gdal
import tqdm
import os
import json
import argparse
import time
import random

#set the main directory
mapdir = f'/exports/eddie/scratch/s1937352' 

def readCommands():
  '''
  Read commandline arguments
  '''
  p = argparse.ArgumentParser(description=("Make a dictionary from data"))
  p.add_argument("--dictName", dest ="dictname", type=str, default="dictionary_c.json", help=("Specify a name for the dictionary"))
  p.add_argument("--outEpsg", dest ="outEPSG", type=int, default=21036, help=("Output EPSG code"))
  p.add_argument("--Pop-Layer", dest ="pop_layer", type=str, default=f'/exports/eddie/scratch/s1937352/UgandaCensus.tif', help=("Specify the population layer"))
  p.add_argument("--Pop-Name ", dest ="pop_name", type=str, default="UgandaCensus", help=("Specify a name for the new popualtion layer"))
  cmdargs = p.parse_args()
  return cmdargs





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


def get_bounds(raster):
    '''
        function to get xmin, xmax, ymin, ymax
        and translate them to number which are dividable by 1000
    '''
    ds = gdal.Open(raster)
    gt = ds.GetGeoTransform()
    minx = crop_up(gt[0])
    maxy = crop_down(gt[3])                           
    miny = crop_up(maxy + gt[5] * ds.RasterYSize)
    maxx = crop_down(minx + gt[1] * ds.RasterXSize)
    return minx, miny, maxx, maxy


def map_pop_to_img(pop, imgdir, bounds, dictTiles, idfile,show=False):
    """
        This function reads an image and tiles it into 1km2 tiles. 
        The extent and the corresonding Population is read into a dictionary.
    """

    #see what kind of image it is 
    # and save the name of the image into a list
    if "VV" in imgdir:
        bands = ['vv']
    elif "VH" in imgdir:
        bands = ['vh']
    elif "S2" in imgdir:
        bands = ['S2']
    elif "MLI" in imgdir:
        bands = ['Coherence']
    else:
        print("This image is not recognised")
        exit()

    # get the boundaries of the image
    minx, miny, maxx, maxy = get_bounds(imgdir)

    #calculate the total number of tiles 
    #within the whole image
    total = 0
    for x in range(minx, maxx, 1000):
        for y in range(miny, maxy, 1000):
            total +=1

    #loop over each tile
    with tqdm.tqdm(total=total) as bar:
        bar.set_description(f"Band: {bands[0]}")
        for x in range(minx, maxx, 1000):
            for y in range(miny, maxy, 1000):

                #check if the tile is within the boundaries of Uganda
                land = max(
                    0,
                    int(
                        gdal.Translate('',
                                        bounds,
                                        projWin=[x, y, x + 1000, y - 1000],
                                        format='VRT').ReadAsArray()[0, 0]))

                if not land:
                    bar.update(1)
                    continue

                # acquire the popualtion count for the tile
                # if nodata value is found it will be set to zero
                v = max(
                    0,      
                    int(
                        gdal.Translate('',
                                        pop,
                                        projWin=[x, y, x + 1000, y - 1000],
                                        format='VRT').ReadAsArray()[0, 0]))
                id = max(
                    0,      
                    int(
                        gdal.Translate('',
                                        idfile,
                                        projWin=[x, y, x + 1000, y - 1000],
                                        format='VRT').ReadAsArray()[0, 0]))

                #read in the image as an array of the specified extent
                tile = gdal.Translate('',
                                        imgdir,
                                        projWin=[x, y, x + 1000, y - 1000],
                                        format='VRT')

                tile_arr = tile.ReadAsArray()

                #check if there is actually data in there, since S1 images are tilted
                complete_tile = (not np.all(tile_arr == tile_arr[0, 0])
                                ) and np.max(tile_arr) < 65534 and np.min(tile_arr) > -99  

                if complete_tile:
                    #generate dictionary key from xmin and ymax
                    xystring = f'{x}_{y}'

                    # dump the filepath of the img and the population into the dictionary
                    if xystring not in dictTiles:
                        dictTiles[xystring] = {"extent": [x, y, x + 1000, y - 1000],
                                                "pop": v,
                                                "id": id,
                                                bands[0]: imgdir}
                    elif xystring in dictTiles:
                        if dictTiles[xystring].get(bands[0]) == None:
                            dictTiles[xystring].update({bands[0]: imgdir})
                    else:
                        continue
                bar.update(1)          
    return dictTiles


def convert(dir,name,outepsg):
    '''
        convert the file to the given epsg
    '''
    extension = '.tiff'
    outName=f'{mapdir}/{name}_P.tiff'
    outEPSG= outepsg
    gdal.Warp(outName,dir,dstSRS='EPSG:'+str(outepsg))

def run_trough_folders(dict_name, outepsg, pop_layer, pop_name):
    '''
        This function iterates over each image folder and calls 
        map_pop_to_img for each image. The dictionary is always passed 
        to the next iteration and also loop
    '''
    #specify folder names of images
    popfileP = f'{pop_layer}'
    boundsfile = f'{mapdir}/UgandaRaster.tif'
    idfile = f'{mapdir}/UgandaParishID.tif'
    imgdirsS2 = f'{mapdir}/Sentinel2'
    imgdirsVV = f'{mapdir}/VV_Backscatter'
    imgdirsVH = f'{mapdir}/VH_Backscatter'
    imgdirsCoherence = f'{mapdir}/Coherence'

    #convert bounds and pop file to specifies coordinate system
    #and open them with gdal
    boundsname = boundsfile[32:-4]
    popname = pop_name
    print(popname)
    print(boundsname)
    '''
    if not os.path.exists(f'{mapdir}/{popname}_P.tiff'):
        convert(popfile,popname,outepsg)
    '''
    if not os.path.exists(f'{mapdir}/{boundsname}_P.tiff'):
        convert(boundsfile,boundsname,outepsg)
    #popfileP = f'{mapdir}/{popname}_P.tiff'
    
    boundsfileP =f'{mapdir}/{boundsname}_P.tiff'
    pop = gdal.Open(popfileP)
    uganda_bounds = gdal.Open(boundsfileP)
    idfile = gdal.Open(idfile)


    #create an empty dictionary
    dictTiles = {}

    #loop over Sentinel 2 image directory
    for idx, imgdir in enumerate(os.listdir(imgdirsS2)):       
        if imgdir.endswith('.tif'):
            imgdir = f'{imgdirsS2}/{imgdir}'
            print(f'Processing {imgdir}. Number {idx+1} from 96')
            dictTiles = map_pop_to_img(pop, imgdir, uganda_bounds, dictTiles, idfile, show=False)

    #loop over Sentinel 1 VV image directory
    for idx, imgdir in enumerate(os.listdir(imgdirsVV)):
        if imgdir.endswith('.tif'):
            imgdir = f'{imgdirsVV}/{imgdir}'
            print(f'Processing {imgdir}.  Number {idx+1} from 14')
            dictTiles = map_pop_to_img(pop, imgdir, uganda_bounds, dictTiles, idfile,show=False)

    #loop over Sentinel 1 VH image directory 
    for idx, imgdir in enumerate(os.listdir(imgdirsVH)):
        if imgdir.endswith('.tif'):                       
            imgdir = f'{imgdirsVH}/{imgdir}'
            print(f'Processing {imgdir}.  Number {idx+1} from 14')
            dictTiles = map_pop_to_img(pop, imgdir, uganda_bounds, dictTiles, idfile,show=False)

    #loop over Sentinel 1 Coherence image directory
    for idx, imgdir in enumerate(os.listdir(imgdirsCoherence)):
        if imgdir.endswith('.tif'):
            imgdir = f'{imgdirsCoherence}/{imgdir}'
            print(f'Processing {imgdir}.  Number {idx+1} from 13')
            dictTiles = map_pop_to_img(pop, imgdir, uganda_bounds, dictTiles, idfile,show=False)

    #dumb the dictionary into json file
    with open(f'{mapdir}/{dict_name}', 'w') as fp:
        json.dump(dictTiles, fp)



def split_train_test(dictname):
    #after creating the dictionary, it is split into training and test data dictionary based on Parish ID
    random.seed(430)
    parishlist = []
    # create a list with all Parish IDs
    for i in range(1,7467):
        parishlist.append(i)
    data = len(parishlist)
    training = int(data * 0.7)
    test =int(data *0.3)
    random.shuffle(parishlist)
    print(training,test)
    # split into training and testing 
    train_data = parishlist[:training]
    test_data = parishlist[training:]

    print(len(train_data),len(test_data))
    dictTraining = {}
    dictTest = {}
    with open(f'{mapdir}/{dictname}') as json_file:
        dictTiles = json.load(json_file)
    #create train dictionary
    for i in train_data:
        for element in dictTiles:
            if dictTiles[element].get("id") == i:
                elm = dictTiles[element]
                dictTraining.update({element:elm})

    #create test dictionary      
    for i in test_data:
        for element in dictTiles:
            if dictTiles[element].get("id") == i:
                elm = dictTiles[element]
                dictTest.update({element:elm})
                
    with open(f'{mapdir}/Train_{dictname}', 'w') as fp:
        json.dump(dictTraining, fp)
    with open(f'{mapdir}/Test_{dictname}', 'w') as fp:
        json.dump(dictTest, fp)


if __name__ == "__main__":

    cmd=readCommands()
    dictname=cmd.dictname
    outEPSG=cmd.outEPSG
    pop_layer=cmd.pop_layer
    pop_name=cmd.pop_name
    start = time.time()

    run_trough_folders(dict_name = dictname , outepsg = outEPSG, pop_layer = pop_layer, pop_name = pop_name)
    split_train_test(dictname)
    print(f'Finished after {(time.time()-start)/60} Minutes')





