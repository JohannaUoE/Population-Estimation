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
from CNN_Uganda_ConvNet import CNN
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import random
import argparse

mapdir = "/exports/eddie/scratch/s1937352"
logdir = "/exports/eddie/scratch/s1937352/logs"

'''
This script is the first step of the CNN test workflow Uganda ConvNet only based on Sentinel 2 data. The test data is pushed through the model and the 
prediction is written to a dictionary
For further details refer to Prediction.py

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
  cmdargs = p.parse_args()
  return cmdargs



def smooth(out, y):
    loss = F.smooth_l1_loss(out, y)
    acc = loss.data.item()
    return loss, acc


def get_seed(parampath):
    # Get the correct seed, otherwise this won't use the proper test set
    if os.path.isfile(parampath):
        with open(parampath, 'r') as f:
            return json.load(f)['seed']
    else:
        return 123

def get_test_set(ds):

    split_lengths = [int(0.1 * len(ds))] * 2
    split_lengths = [len(ds) - sum(split_lengths)] + split_lengths
    t, v, test_split = random_split(ds, split_lengths)
    test = DataLoader(test_split, batch_size=1, shuffle=True, pin_memory=True, num_workers=6)
    return test

def get_whole_set(ds):
    dataset = DataLoader(ds, batch_size=1, shuffle=True, pin_memory=True, num_workers=6)
    return dataset

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


class FileDatasetPred(Dataset):
    """
        This class combines Population data and image data to feed it into the CNN. 
        The image data is a 8 dimensional numpy array that is calculated based on the entries of the 
        dictionary.
    """

    images = ['S2','vv', 'vh', 'Coherence']
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
        return_img = np.empty((5, self.width, self.height), dtype=np.float32)

        #loop thorugh the image names to generate the array from different images
        for band_id, band_name in enumerate(self.images):
            #get the imagedir
            imgdir = self.dictTiles[key][self.images[band_id]]

            if band_id == 1:
                continue

            elif band_id == 2:
                continue
            
            elif band_id == 3:
                continue

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

        return return_img, pop,key



def write_pred(modelpath,parampath,dict,dict_pred):
    dataset = FileDatasetPred(dict)
    print(dataset)
    model = torch.load(modelpath)
    model.to(torch.device('cuda'))
    model.eval()
    dataset = get_whole_set(dataset)
    with open(f'{mapdir}/{dict}') as json_file:
        dictTiles = json.load(json_file)
    with tqdm.tqdm(total=len(dataset)) as bar:
        for x, y, f in dataset:
            #print(x,y)

            out = model.forward(x.to(torch.device('cuda')))

            xystring = f[0]
            if dictTiles[xystring].get("pred") == None: 
                dictTiles[xystring].update({"pred": out.data.item()})
                #print(dictTiles[xystring])
            bar.update(1) 
            
    with open(f'{mapdir}/{dict_pred}', 'w') as fp:
        json.dump(dictTiles, fp) 


def main(log_name,dictionary,dict_pred):

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
    start = time.time()

    main(log_name=log_name,dictionary=dict,dict_pred=dict_pred)

    print(f'Finished after {(time.time()-start)/60} Minutes')
