import numpy as np
import os
import tqdm
import gdal
import random
from torch.utils.data import Dataset
import json

''' 
The DataLoader is a helper class for training and testing the neural network. It pushes randomly input and true data 
into the architecture

Script by Johanna Kauffert (initial implementation McCabe)
'''
mapdir =  "/exports/eddie/scratch/s1937352"

class FileDataset(Dataset):
    """
        This class combines Population data and image data to feed it into the CNN. 
        The image data is a 8 dimensional numpy array that is calculated based on the entries of the 
        dictionary.
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

        return return_img, pop
