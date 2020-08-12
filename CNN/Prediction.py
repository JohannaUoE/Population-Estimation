import matplotlib.pyplot as plt
import os
import sys
import json
import torch
import tqdm
from math import sqrt
from osgeo import gdal
import random
import numpy as np
import time
from statistics import median
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import random
import argparse

mapdir = "/exports/eddie/scratch/s1937352"
logdir = "/exports/eddie/scratch/s1937352/logs"


'''
This script is the first step of the CNN test workflow. The test data is pushed through the model and the 
prediction is written to a dictionary

Script by Johanna Kauffert
'''


def readCommands():
  '''
  Read commandline arguments
  '''
  p = argparse.ArgumentParser(description=("Get parameters for training the CNN"))
  p.add_argument("--log_name", dest ="log_name", type=str, default="CNN_number", help=("Specify a name for the CNN"))
  p.add_argument("--dict", dest ="dict", type=str, default="dictionary_p2.json", help=("Specify a dictionary that will be used"))
  p.add_argument("--dict_pred", dest ="dict_pred", type=str, default="dictionary_p2_pred.json", help=("Specify a dictionary that will be used"))
  p.add_argument("--CNN_type", dest ="cnn", type=int, default=0, help=("normal ConvNet (0) 3 Branch (1), 2 Branch S2 Gamma (2), 2 Branch S2 Coherence (3)"))
  cmdargs = p.parse_args()
  return cmdargs



def smooth(out, y):
    #loss function L1Smooth
    loss = F.smooth_l1_loss(out, y)
    acc = loss.data.item()
    return loss, acc


def get_test_set(ds):
    #get only the test dataset
    split_lengths = [int(0.1 * len(ds))] * 2
    split_lengths = [len(ds) - sum(split_lengths)] + split_lengths
    t, v, test_split = random_split(ds, split_lengths)
    test = DataLoader(test_split, batch_size=1, shuffle=True, pin_memory=True, num_workers=6)
    return test

def get_whole_set(ds):
    #get the whole dataset with DataLoader
    dataset = DataLoader(ds, batch_size=1, shuffle=True, pin_memory=True, num_workers=6)
    return dataset


class FileDatasetPred(Dataset):
    """
        This class combines Population data and image data to feed it into the CNN. 
        The image data is a 8 dimensional numpy array that is calculated based on the entries of the 
        dictionary.
        It is not imported as it also returns the key
    """

    images = ['vv', 'vh', 'S2', 'Coherence']
    bands = ['B4', 'B3', 'B2', 'B8', 'B11', 'vv', 'vh', 'Coherence']
    
    def __init__(self,dictionary):
        # create a list with all keys that have data in all bands
        self.keys_list = []
        with open(f'{mapdir}/{dictionary}') as json_file:
            self.dictTiles = json.load(json_file)

        for element in self.dictTiles:
            if self.dictTiles[element].get("pop") != None:
                if self.dictTiles[element].get("S2") != None:
                    if self.dictTiles[element].get("vv") != None:
                        if self.dictTiles[element].get("vh") != None:
                            if self.dictTiles[element].get("Coherence") != None:
                                self.keys_list.append(element)
        
        #check how long the list is
        print(len(self.keys_list))
        
        #get width and height of a tile so that it can be further used
        for i in self.keys_list:
            imgdir = self.dictTiles[i]["S2"]
            coords = self.dictTiles[i]["extent"]
            tile = gdal.Translate('',
                        imgdir,
                        projWin=coords,
                        format='VRT')
            tile_arr = tile.ReadAsArray()
            self.S2bands, self.width, self.height = tile_arr.shape
            print(self.S2bands, self.width, self.height)
            break
        
        #shuffle the list so that tiles next to each other are apart
        random.seed(430)
        random.shuffle(self.keys_list)
      
    def __len__(self):
        #function needed by PyTorch
        return len(self.keys_list)

    def __getitem__(self,idx):
        # here is more information about this:
        #https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

        #get the key for the dict
        key = self.keys_list[idx]
        coords = self.dictTiles[key]["extent"]
        pop = self.dictTiles[key]["pop"]
        pops = np.array(pop, dtype=np.float32)
        pop = pops

        #create empty 8D list
        return_img = np.empty((8, self.width, self.height), dtype=np.float32)

        #loop thorugh the image names to generate the array from different images
        for band_id, band_name in enumerate(self.images):
            #get the imagedir
            imgdir = self.dictTiles[key][self.images[band_id]]

            if band_id == 0:
                tile = gdal.Translate('',
                            imgdir,
                            projWin=coords,
                            format='VRT')
                tile_arr = tile.ReadAsArray()
                return_img[5, :, :] = tile_arr

            elif band_id == 1:
                tile = gdal.Translate('',
                            imgdir,
                            projWin=coords,
                            format='VRT')
                tile_arr = tile.ReadAsArray()
                return_img[6, :, :] = tile_arr
            
            elif band_id == 3:
                tile = gdal.Translate('',
                            imgdir,
                            projWin=coords,
                            format='VRT')
                tile_arr = tile.ReadAsArray()
                return_img[7, :, :] = tile_arr

            else:
                tile = gdal.Translate('',
                            imgdir,
                            projWin=coords,
                            format='VRT')
                noS2bands = tile.RasterCount
                for s2b in range(noS2bands):
                    bandx = tile.GetRasterBand(s2b+1)
                    bandxarray = bandx.ReadAsArray()
                    return_img[s2b, :, :] = bandxarray

        return return_img, pop, key



def write_pred(modelpath,parampath,dict,dict_pred):
    #function to write the prediction to a dictionary
    #load dataset
    dataset = FileDatasetPred(dict)
    print(dataset)
    #load model
    model = torch.load(modelpath)
    model.to(torch.device('cuda'))
    model.eval()
    #use the whoel dataset since it is only the test dictionary
    dataset = get_whole_set(dataset)
    #open the dictionary
    with open(f'{mapdir}/{dict}') as json_file:
        dictTiles = json.load(json_file)
    with tqdm.tqdm(total=len(dataset)) as bar:
        for x, y, f in dataset:
            # push the data through the model
            out = model.forward(x.to(torch.device('cuda')))

            xystring = f[0]
            if dictTiles[xystring].get("pred") == None: 
                #write prediction to the dictionary
                dictTiles[xystring].update({"pred": out.data.item()})

            bar.update(1) 
    #save the new dictionary
    with open(f'{mapdir}/{dict_pred}', 'w') as fp:
        json.dump(dictTiles, fp) 


def main(log_name,dictionary,dict_pred):
    # get data from paramfile and specify model path
    parampath = f"{logdir}/{log_name}/paramfile.json"
    modelpath = f"{logdir}/{log_name}/model.pt"
    torch.manual_seed(12345)


    if os.path.isfile(modelpath):
        model = torch.load(modelpath)
        model.eval()
        model.to(torch.device('cuda'))
        write_pred(parampath=parampath, modelpath=modelpath,dict=dict,dict_pred=dict_pred)
    else:
        print("Error: No data found for experiment '{}'".format(sys.argv[1]))


if __name__ == "__main__":

    cmd=readCommands()
    dict=cmd.dict
    dict_pred = cmd.dict_pred
    log_name=cmd.log_name
    cnn = cmd.cnn
    start = time.time()
    # decide which CNN architecture to choose
    if cnn == 0:
        from CNN_Uganda_ConvNet import CNN
    elif cnn == 1:
        from CNN_Uganda_3-Branches-ConvNet import CNN
    elif cnn == 2:
        from CNN_Uganda_2-Branches-ConvNet-S2-Gamma0 import CNN
    elif cnn == 3:
        from CNN_Uganda_2-Branches-ConvNet-S2-Coherence import CNN

    main(log_name=log_name,dictionary=dict,dict_pred=dict_pred)

    print(f'Finished after {(time.time()-start)/60} Minutes')
