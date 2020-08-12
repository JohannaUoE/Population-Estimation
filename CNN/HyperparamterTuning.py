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
from CNN_Johanna import CNN
from DataLoader_Johanna import FileDataset


mapdir = "/exports/eddie/scratch/s1937352"
logdir = "/exports/eddie/scratch/s1937352/logs"

'''
This script performs a grid search to determine the best hyperparamters for Uganda ConvNet
It is based on train_Uganda and not explained in detail here.

The function train_model_grid_search runs the hyperparamter tuning over a couple of paramters

Script by Johanna Kauffert (initial implementation McCabe)
'''

def smooth(out, y):
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

        del self.params['self']
        del self.params['dataset']
        self.params['loss_fn'] = self.params['loss_fn'].__name__

        self.train_duration = None
        self.best_loss = ("-1", float("inf"), float("inf"))

        self.jobpath = self.initialise_log_dir()

        self.device = self.initialise_pytorch()
        self.train, self.val, self.test, self.data_shape = self.initialise_datasets_orig()
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

    def initialise_datasets_orig(self):
        # Initialise data, partitioning randomly into 80% train, 10% val, 10% t$
        if self.dataset is None:
            self.dataset = FileDataset()
        data_shape = self.dataset[0][0].shape
        len_orig = len(self.dataset)

        #make a list of twice 10 % of the dataset length
        split_lengths = [int(0.1 * len(self.dataset))] * 2
        # add to the list 80% of the dataset length
        split_lengths = [len(self.dataset) - sum(split_lengths)] + split_lengths
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
    
    def get_job_args_dict(self):
        # Initialise a dict of Job arguments
        job_args = {
            k: v.default
            for k, v in inspect.signature(Job).parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        del job_args['dataset']
        del job_args['num_epochs']
        del job_args['early_stop_mem']

        return job_args


    def train_model_grid_search(self):
        # Perform a grid search on a number of hyperparameters.
        params = {"learning_rate": [1e-4], "l2": [0.1, 0.01, 0.001, 0.0001 ]}

        job_args = self.get_job_args_dict()

        permutations = list(itertools.product(params["learning_rate"], params["l2"]))
        print(permutations)
        # combine the different lists 
        for idx, per in enumerate(permutations):
            permutation = [("learning_rate", per[0]), ("l2",per[1])]
            print(permutation)
            for name, val in permutation:
                job_args[name] = val
            #name the log according to the paramters and run the model for 4 epochs
            expname = re.sub(
                '\.', '', "02-07_Hyper_{}".format("_".join(
                    ["{}-{}".format(n, self.name(v)) for n, v in permutation])))
            print("Running job {} ({} of {})".format(expname, idx + 1, len(permutations)))
            job = Job(expname, dataset=self.dataset, num_epochs=4, early_stop_mem=10, **job_args)
            job.train_model()




def readCommands():
  '''
  Read commandline arguments
  '''
  p = argparse.ArgumentParser(description=("Get parameters for training the CNN"))
  p.add_argument("--dict", dest ="dict", type=str, default="dictionary_p2.json", help=("Specify a dictionary that will be used"))
  cmdargs = p.parse_args()
  return cmdargs



def main():
    cmd=readCommands()
    dict=cmd.dict
    js = JobScheduler(dict)
    js.train_model_grid_search()




if __name__ == "__main__":

    PROFILE = 'process.profile'
    prf = cProfile.run('main()', PROFILE)

    p = pstats.Stats(PROFILE)
    p.sort_stats('cumtime').print_stats(20)
