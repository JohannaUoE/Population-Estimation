import tqdm
import time
import sys
import re
import os
import gdal
import json
import argparse
import shutil
import inspect
import itertools
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import random
import cProfile
import pstats
from CNN_Uganda_ConvNet import CNN

'''
This training script is adopted from train_Uganda and fit to the purpose of only training a CNN on Sentinel 2 bands.
For more details refer to the Technical report and train_Uganda script

Script by Johanna Kauffert 
'''

mapdir = "/exports/eddie/scratch/s1937352"
logdir = "/exports/eddie/scratch/s1937352/logs"


class FileDataset(Dataset):
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

        return return_img, pop



def smooth(out, y):
    '''
    Creates a criterion that uses a squared term if the absolute element-wise error 
    falls below 1 and an L1 term otherwise. It is less sensitive to outliers than
     the MSELoss and in some cases prevents exploding gradients
    '''
    loss = F.smooth_l1_loss(out, y)
    acc = loss.data.item()
    return loss, acc


class Job(object):

    def __init__(
            self,
            exp_name,
            dataset=None,
            batch_size=64,
            seed=12345,
            num_layers=4,
            num_filters=128,
            num_epochs=5,
            learning_rate=1e-5,
            dropout_rate=0.1,
            l2=0.2,
            kernel_size=5,
            loss_fn=smooth,
            early_stop_mem=0,
            alt_cnn=False,
            kfold=0,
            start=0,
            small = False,
            parish = 3
    ):
        self.exp_name = exp_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2 = l2
        self.loss_fn = loss_fn
        self.early_stop_mem = early_stop_mem
        self.alt_cnn = alt_cnn
        self.kfold = kfold
        self.params = locals()
        self.start = start
        self.small = small
        self.parish = parish

        del self.params['self']
        del self.params['dataset']
        self.params['loss_fn'] = self.params['loss_fn'].__name__

        self.train_duration = None
        self.best_loss = ("-1", float("inf"), float("inf"))

        self.jobpath = self.initialise_log_dir()

        self.device = self.initialise_pytorch()
        if self.parish == 0:
            self.train, self.val, self.data_shape= self.initialise_datasets_parish_train()
        elif self.parish == 1:
            self.test, self.data_shape = self.initialise_datasets_parish_test()
        else:
            self.train, self.val, self.test, self.data_shape = self.initialise_datasets_orig()

        #self.train, self.val, self.test, self.data_shape = self.initialise_datasets_orig()
        self.model = self.initialise_cnn()
        self.optimiser = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.l2)

    def save_model(self):
        # Save the model to disk
        self.model.cpu()
        torch.save(self.model, "{}/model.pt".format(self.jobpath))
        self.model.to(self.device)

    def train_model(self):
        # Perform training on the model
        train_start_time = time.time()
        best_val_loss = float("inf")
        for i in range(self.start, self.start + self.num_epochs, 1):
            train_losses = 0.0
            print("I am here")
            self.model.train()
            with tqdm.tqdm(total=len(self.train) * self.train.batch_size) as bar:
                bar.set_description("Training  |Epoch {}".format(i))
                for j, (x, y) in enumerate(self.train):
                    x, y = [torch.Tensor(z).to(device=self.device) for z in [x, y]]
                    self.optimiser.zero_grad()
                    out = self.model.forward(x).view(-1)
                    #print(out)
                    loss, acc = self.loss_fn(out, y)
                    #print(loss,acc)
                    train_losses += acc
                    loss.backward()
                    self.optimiser.step()
                    bar.update(self.train.batch_size)
            val_losses = self.validate_model(i)
            train_loss = train_losses / len(self.train)
            val_loss = val_losses / len(self.val)
            if val_loss < self.best_loss[2]:
                self.best_loss = (i, train_loss, val_loss)
            elif (self.early_stop_mem != 0 and i - self.best_loss[0] >= self.early_stop_mem):
                break
            print("Training  |RMS loss {} : {:0.2f}".format(i, train_loss))
            print("Validation|RMS loss {} : {:0.2f}\n".format(i, val_loss))
            best_val_loss = self.record_data(i, train_loss, val_loss, best_val_loss)
            print(*self.best_loss)
        self.train_duration = (time.time() - train_start_time)/60
        t_ofile = '{}/duration.txt'.format(self.jobpath)

        with open(t_ofile, 'w') as f:
            f.write(str(self.train_duration))

    def validate_model(self, epoch):
        # Perform validation on the model
        losses = 0.0
        self.model.eval()
        with tqdm.tqdm(total=len(self.val) * self.val.batch_size) as bar:
            bar.set_description("Validation|Epoch {}".format(epoch))
            for j, (x, y) in enumerate(self.val):
                x, y = [torch.Tensor(z).to(device=self.device) for z in [x, y]]
                out = self.model.forward(x)
                loss = F.l1_loss(out, y).view(-1)
                losses += loss.data.item()
                bar.update(self.val.batch_size)
        return losses

    def record_data(self, epoch, train_loss, val_loss, best_val_loss):
        # Record training data to the current job's log file
        ofile = '{}/data.csv'.format(self.jobpath)
        with open(ofile, 'a') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
            writer.writerow([epoch, train_loss, val_loss])
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            self.save_model()
        return best_val_loss

    def initialise_cnn(self):
        # Set up the model to be trained, either by loading an existing model
        # or creating one from scratch.
        if self.start == 0:
            if self.alt_cnn:
                model = models.vgg11(pretrained=False)
                num_fts = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_fts, 1)
            else:
                data_shape = (self.batch_size, self.data_shape[0], self.data_shape[1],
                              self.data_shape[2])
                model = CNN(data_shape, self.num_layers, self.num_filters, self.dropout_rate,
                            self.kernel_size)
                
        elif os.path.exists(f"{self.jobpath}/model.pt"):
            model = torch.load("{}/model.pt".format(self.jobpath))
        
            
            
        else:
            data_shape = (self.batch_size, self.data_shape[0], self.data_shape[1], self.data_shape[2])
            model = CNN(data_shape, self.num_layers, self.num_filters, self.dropout_rate, self.kernel_size)


        model.to(self.device)
        return model




    def initialise_datasets_parish_train(self):
        if self.dataset is None:
            self.dataset = FileDataset()
        data_shape = self.dataset[0][0].shape
        len_orig = len(self.dataset)
        split_lengths = [int(0.15 * len(self.dataset))]
        split_lengths = [len(self.dataset) - sum(split_lengths)] + split_lengths
        sets = random_split(self.dataset, split_lengths)
        train, val, = [
            DataLoader(x, batch_size=b, shuffle=True, pin_memory=True, num_workers=4)
            for x, b in zip(sets, [self.batch_size, self.batch_size])
        ]
        return train, val, data_shape

    def initialise_datasets_parish_test(self):
        if self.dataset is None:
            self.dataset = FileDataset()
        data_shape = self.dataset[0][0].shape

        test = DataLoader(self.dataset, batch_size=b, shuffle=True, pin_memory=True, num_workers=4)

        return  test, data_shape

    def initialise_datasets_orig(self):
        # Initialise data, partitioning randomly into 80% train, 10% val, 10% t$
        if self.dataset is None:
            self.dataset = FileDataset()
        data_shape = self.dataset[0][0].shape
        len_orig = len(self.dataset)
        if self.small:

            slit_half = [int(0.25 * len(self.dataset))]*4
            dataset,d2,d3,d4 = random_split(self.dataset, slit_half)
            #datatset = dataset[0]
            print(len(dataset))
            split_length = [int(0.2*len(dataset))]*2
            split_length = [len(dataset)-sum(split_length)]+split_length
            sets = random_split(dataset, split_length)

        else:
            #make a list of twice 10 % of the dataset length
            split_lengths = [int(0.1 * len(self.dataset))] * 2
            # add to the list 80% of the dataset length
            split_lengths = [len(self.dataset) - sum(split_lengths)] + split_lengt
            #randomly split the dataset into 80,10,10
            sets = random_split(self.dataset, split_lengths)
        
        train, val, test = [
            DataLoader(x, batch_size=b, shuffle=True, pin_memory=True, num_workers=4)
            for x, b in zip(sets, [self.batch_size, self.batch_size, 1])
        ]
        return train, val, test, data_shape

    def initialise_log_dir(self):
        # Set up a log directory for the current job.
        print(logdir)
        jobpath = f'{logdir}/{self.exp_name}'
        print(jobpath)

        if self.start == 0:
            if os.path.isdir(jobpath):
                shutil.rmtree(jobpath)
            os.mkdir(jobpath)
        elif not os.path.isdir(jobpath):
            os.mkdir(jobpath)
        with open("{}/paramfile.json".format(jobpath), 'w') as f:
            f.write(json.dumps(self.params))
        return jobpath

    def initialise_pytorch(self):
        # Set up pytorch parameters.
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            print("Training CNN with GPU")
            return torch.device('cuda')
        else:
            print("Training CNN with CPU")
            return torch.device('cpu')


class JobScheduler(object):

    def __init__(self,dict):
        self.dataset = FileDataset(dict)

    def name(self, val):
        return val.__name__ if hasattr(val, "__name__") else str(val)


    def train_model_list(self,log_name,num_epochs,start,learnr,small,l2,parish):
        # Train a number of specified jobs.
        if os.path.isdir(f'{logdir}/{log_name}'):
            with open(f'{logdir}/{log_name}/paramfile.json') as json_file:
                paramdict = json.load(json_file)
            start = paramdict.get("start")
            num_epochs = paramdict.get("num_epochs")
            start = start+num_epochs
        else:
            start = start

        jobs = [
            Job(log_name,
                dataset=self.dataset,
                num_epochs=num_epochs,
                early_stop_mem=0,
                kfold=0,
                loss_fn=smooth,
                learning_rate=learnr,
                start=start,
                l2 = l2,
                small = small,
                parish=parish)]

        for i, job in enumerate(jobs):
            job.train_model()


def readCommands():
  '''
  Read commandline arguments
  '''
  p = argparse.ArgumentParser(description=("Get parameters for training the CNN"))
  p.add_argument("--log_name", dest ="log_name", type=str, default="CNN_number", help=("Specify a name for the CNN"))
  p.add_argument("--num_epochs", dest ="num_epochs", type=int, default=5, help=("Number of epochs"))
  p.add_argument("--start", dest ="start", type=int, default=0, help=("Specify a epoch number"))
  p.add_argument("--learning-rate", dest ="learning_rate", type=float, default=0.00001, help=("Specify a learning rate"))
  p.add_argument("--l2", dest ="l2", type=float, default=0.01, help=("Specify l2"))
  p.add_argument("--parish", dest ="parish", type=int, default=0, help=("Train on Paris (0:Train, 1:Test), train on whole dictionary (3)"))
  p.add_argument("--dict", dest ="dict", type=str, default="dictionary_p2.json", help=("Specify a dictionary that will be used"))
  p.add_argument("--smalldata", dest ="small", type=bool, default=False, help=("Use Small dataset"))
  cmdargs = p.parse_args()
  return cmdargs



def main():
    cmd=readCommands()
    log_name=cmd.log_name
    num_epochs=cmd.num_epochs
    start=cmd.start
    learnr=cmd.learning_rate
    dict=cmd.dict
    small=cmd.small
    parish=cmd.parish
    l2=cmd.l2
    js = JobScheduler(dict)
    js.train_model_list(log_name,num_epochs,start,learnr,small,l2,parish)



if __name__ == "__main__":

    PROFILE = 'process.profile'
    prf = cProfile.run('main()', PROFILE)

    p = pstats.Stats(PROFILE)
    p.sort_stats('cumtime').print_stats(20)
