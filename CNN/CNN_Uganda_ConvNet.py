import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Class that holds the architecture of the CNN and forwards the input data trough it.
The layout of teh CNN is given in the research paper and is therefore not explained in this script
Script by Johanna Kauffert
'''
class CNN(nn.Module):

    def __init__(self, input_shape, num_layers, num_filters, dropout_rate, kernel_size):
        super(CNN, self).__init__()

        # input_shape format: (batch_size, n_channels, im_height, im_width)
        # assign attributes of the class
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()
        self.build_module()
        self.init_weights()

    def build_module(self):
        x = torch.zeros(self.input_shape)
        out = x

        print("Feature shapes:")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print(out.shape)
        print(out.shape[0])
        print(out.shape[1])
        #vgga 11 weights
        # the shape of the input data decides upon the number of input channels
        self.conv1 = nn.Conv2d(in_channels=out.shape[1],
                        kernel_size = 5,
                        out_channels = 64,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out = self.conv1(out)
        print(out.shape)
        self.maxpool1 = nn.MaxPool2d(2)
        out = self.maxpool1(out)
        print(out.shape)

        self.conv2 = nn.Conv2d(in_channels=64,
                        kernel_size = 5,
                        out_channels = 128,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out = self.conv2(out)
        print(out.shape)
        self.maxpool2 = nn.MaxPool2d(2)
        out = self.maxpool2(out)
        print(out.shape)
        self.conv3 = nn.Conv2d(in_channels= 128,
                        kernel_size = 5,
                        out_channels = 256,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out = self.conv3(out)
        print(out.shape)
        self.conv4 = nn.Conv2d(in_channels= 256,
                        kernel_size = 5,
                        out_channels = 256,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out = self.conv4(out)
        print(out.shape)
        self.maxpool3 = nn.MaxPool2d(2)
        out = self.maxpool3(out)
        print(out.shape)

        self.conv5 = nn.Conv2d(in_channels= 256,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out = self.conv5(out)
        print(out.shape)

        self.conv6 = nn.Conv2d(in_channels= 512,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out = self.conv6(out)
        print(out.shape)
        self.maxpool4 = nn.MaxPool2d(2)
        out = self.maxpool4(out)
        print(out.shape)
        self.conv7 = nn.Conv2d(in_channels= 512,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out = self.conv7(out)
        print(out.shape)
        self.conv8 = nn.Conv2d(in_channels= 512,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        
        out = self.conv8(out)
        print(out.shape)
        self.maxpool5 = nn.MaxPool2d(2)
        out = self.maxpool5(out)
        print(out.shape) #64,512,3,3 
        out = out.view(-1,4608)  #torch.Size([64, 4608])
        print(out.shape) 

        self.fc1 = nn.Linear(in_features=4608, out_features=1024, bias=False) #4608
        out = self.fc1(out)
        print(out.shape)
        self.fc2 = nn.Linear(in_features=1024, out_features=256, bias=False)
        out = self.fc2(out)
        print(out.shape)
        self.fc3 = nn.Linear(in_features=256, out_features=1, bias=False)
        out = self.fc3(out)
        print(out.shape)


    def init_weights(self):
        for param in self.parameters():
            if len(param.size()) > 1:
                torch.nn.init.xavier_uniform_(param.data)
            else:
                torch.nn.init.zeros_(param.data)

    def forward(self, x):
        #forward function that pushes the input data through teh architecture
        out = x
        out = F.relu(self.conv1(out))
        out = self.maxpool1(out)
        out = F.relu(self.conv2(out))
        out = F.dropout2d(out, p=self.dropout_rate, training=self.training)
        out = self.maxpool2(out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.maxpool3(out)
        out = F.relu(self.conv5(out))
        out = F.dropout2d(out, p=self.dropout_rate, training=self.training)
        out = F.relu(self.conv6(out))
        out = self.maxpool4(out)
        out = F.relu(self.conv7(out))
        out = F.relu(self.conv8(out))
        out = F.dropout2d(out, p=self.dropout_rate, training=self.training)
        out = self.maxpool5(out)

        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.dropout2d(out, p=self.dropout_rate, training=self.training)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        out = out.view(-1)
        return out