import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
This architecture was written for the 3 Branches CNN and repeats the CNN from Uganda ConvNet 
3 times Sentinel 2, Backscatter and Coherence. The branches are concatenated before the last
2 fully connected layers
Script by Johanna Kauffert
'''

class CNN(nn.Module):
    liste = [5,2,1]
    def __init__(self, input_shape, num_layers, num_filters, dropout_rate, kernel_size):
        super(CNN, self).__init__()

        # input_shape format: (batch_size, n_channels, im_height, im_width)
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
        #m#########model sentinel 2 #######

        x = torch.zeros(self.input_shape)
        print("Feature shapes Sentinel 2:")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        out_S2 = x[:,:5,:,:]
        print(out_S2.shape)
        print(out_S2.shape[0])
        print(out_S2.shape[1])
        #vgga 11 weights
        
        self.conv1_S2 = nn.Conv2d(in_channels=out_S2.shape[1],
                        kernel_size = 5,
                        out_channels = 64,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_S2 = self.conv1_S2(out_S2)
        print(out_S2.shape)
        self.maxpool1_S2 = nn.MaxPool2d(2)
        out_S2 = self.maxpool1_S2(out_S2)
        print(out_S2.shape)

        self.conv2_S2 = nn.Conv2d(in_channels=64,
                        kernel_size = 5,
                        out_channels = 128,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_S2 = self.conv2_S2(out_S2)
        print(out_S2.shape)
        self.maxpool2_S2 = nn.MaxPool2d(2)
        out_S2 = self.maxpool2_S2(out_S2)
        print(out_S2.shape)
        self.conv3_S2 = nn.Conv2d(in_channels= 128,
                        kernel_size = 5,
                        out_channels = 256,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_S2 = self.conv3_S2(out_S2)
        print(out_S2.shape)
        self.conv4_S2 = nn.Conv2d(in_channels= 256,
                        kernel_size = 5,
                        out_channels = 256,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_S2 = self.conv4_S2(out_S2)
        print(out_S2.shape)
        self.maxpool3_S2 = nn.MaxPool2d(2)
        out_S2 = self.maxpool3_S2(out_S2)
        print(out_S2.shape)

        self.conv5_S2 = nn.Conv2d(in_channels= 256,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_S2 = self.conv5_S2(out_S2)
        print(out_S2.shape)

        self.conv6_S2 = nn.Conv2d(in_channels= 512,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_S2 = self.conv6_S2(out_S2)
        print(out_S2.shape)
        self.maxpool4_S2 = nn.MaxPool2d(2)
        out_S2 = self.maxpool4_S2(out_S2)
        print(out_S2.shape)
        self.conv7_S2 = nn.Conv2d(in_channels= 512,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_S2 = self.conv7_S2(out_S2)
        print(out_S2.shape)
        self.conv8_S2 = nn.Conv2d(in_channels= 512,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        
        out_S2 = self.conv8_S2(out_S2)
        print(out_S2.shape)
        self.maxpool5_S2 = nn.MaxPool2d(2)
        out_S2 = self.maxpool5_S2(out_S2)
        print(out_S2.shape) #64,512,3,3 
        out_S2 = out_S2.view(-1,4608)  #torch.Size([64, 4608])
        print(out_S2.shape) 
        self.fc1_S2 = nn.Linear(in_features=4608, out_features=1024, bias=False) #4608
        out_S2 = self.fc1_S2(out_S2)
        print(out_S2.shape)

        #m######### model sentinel 1 #######
        out_S1 = x[:,5:7,:,:]
        print(out_S1.shape)
        print(out_S1.shape[0])
        print(out_S1.shape[1])
        #vgga 11 weights
        
        self.conv1_S1 = nn.Conv2d(in_channels=out_S1.shape[1],
                        kernel_size = 5,
                        out_channels = 64,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_S1 = self.conv1_S1(out_S1)
        print(out_S1.shape)
        self.maxpool1_S1 = nn.MaxPool2d(2)
        out_S1 = self.maxpool1_S1(out_S1)
        print(out_S1.shape)

        self.conv2_S1 = nn.Conv2d(in_channels=64,
                        kernel_size = 5,
                        out_channels = 128,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_S1 = self.conv2_S1(out_S1)
        print(out_S1.shape)
        self.maxpool2_S1 = nn.MaxPool2d(2)
        out_S1 = self.maxpool2_S1(out_S1)
        print(out_S1.shape)
        self.conv3_S1 = nn.Conv2d(in_channels= 128,
                        kernel_size = 5,
                        out_channels = 256,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_S1 = self.conv3_S1(out_S1)
        print(out_S1.shape)
        self.conv4_S1 = nn.Conv2d(in_channels= 256,
                        kernel_size = 5,
                        out_channels = 256,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_S1 = self.conv4_S1(out_S1)
        print(out_S1.shape)
        self.maxpool3_S1 = nn.MaxPool2d(2)
        out_S1 = self.maxpool3_S1(out_S1)
        print(out_S1.shape)

        self.conv5_S1 = nn.Conv2d(in_channels= 256,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_S1 = self.conv5_S1(out_S1)
        print(out_S1.shape)

        self.conv6_S1 = nn.Conv2d(in_channels= 512,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_S1 = self.conv6_S1(out_S1)
        print(out_S1.shape)
        self.maxpool4_S1 = nn.MaxPool2d(2)
        out_S1 = self.maxpool4_S1(out_S1)
        print(out_S1.shape)
        self.conv7_S1 = nn.Conv2d(in_channels= 512,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_S1 = self.conv7_S1(out_S1)
        print(out_S1.shape)
        self.conv8_S1 = nn.Conv2d(in_channels= 512,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        
        out_S1 = self.conv8_S1(out_S1)
        print(out_S1.shape)
        self.maxpool5_S1 = nn.MaxPool2d(2)
        out_S1 = self.maxpool5_S1(out_S1)
        print(out_S1.shape) #64,512,3,3 
        out_S1 = out_S1.view(-1,4608)  #torch.Size([64, 4608])
        print(out_S1.shape) 
        self.fc1_S1 = nn.Linear(in_features=4608, out_features=1024, bias=False) #4608
        out_S1 = self.fc1_S1(out_S1)
        print(out_S1.shape)

        #m######### model sentinel 1 #######
        out_Coh = x[:,7:8,:,:]
        print(out_Coh.shape)
        print(out_Coh.shape[0])
        print(out_Coh.shape[1])
        #vgga 11 weights
        
        self.conv1_Coh = nn.Conv2d(in_channels=out_Coh.shape[1],
                        kernel_size = 5,
                        out_channels = 64,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_Coh = self.conv1_Coh(out_Coh)
        print(out_Coh.shape)
        self.maxpool1_Coh = nn.MaxPool2d(2)
        out_Coh = self.maxpool1_Coh(out_Coh)
        print(out_Coh.shape)

        self.conv2_Coh = nn.Conv2d(in_channels=64,
                        kernel_size = 5,
                        out_channels = 128,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_Coh = self.conv2_Coh(out_Coh)
        print(out_Coh.shape)
        self.maxpool2_Coh = nn.MaxPool2d(2)
        out_Coh = self.maxpool2_S2(out_Coh)
        print(out_Coh.shape)
        self.conv3_Coh = nn.Conv2d(in_channels= 128,
                        kernel_size = 5,
                        out_channels = 256,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_Coh = self.conv3_Coh(out_Coh)
        print(out_Coh.shape)
        self.conv4_Coh = nn.Conv2d(in_channels= 256,
                        kernel_size = 5,
                        out_channels = 256,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_Coh = self.conv4_Coh(out_Coh)
        print(out_Coh.shape)
        self.maxpool3_Coh = nn.MaxPool2d(2)
        out_Coh = self.maxpool3_Coh(out_Coh)
        print(out_Coh.shape)

        self.conv5_Coh = nn.Conv2d(in_channels= 256,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_Coh = self.conv5_Coh(out_Coh)
        print(out_Coh.shape)

        self.conv6_Coh = nn.Conv2d(in_channels= 512,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_Coh = self.conv6_Coh(out_Coh)
        print(out_Coh.shape)
        self.maxpool4_Coh = nn.MaxPool2d(2)
        out_Coh = self.maxpool4_Coh(out_Coh)
        print(out_Coh.shape)
        self.conv7_Coh = nn.Conv2d(in_channels= 512,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        out_Coh = self.conv7_Coh(out_Coh)
        print(out_Coh.shape)
        self.conv8_Coh = nn.Conv2d(in_channels= 512,
                        kernel_size = 5,
                        out_channels = 512,
                        padding=int((self.kernel_size - 1) // 2),
                        bias=False)
        
        out_Coh = self.conv8_Coh(out_Coh)
        print(out_Coh.shape)
        self.maxpool5_Coh = nn.MaxPool2d(2)
        out_Coh = self.maxpool5_S1(out_Coh)
        print(out_Coh.shape) #64,512,3,3 
        out_Coh = out_Coh.view(-1,4608)  #torch.Size([64, 4608])
        print(out_Coh.shape) 
        self.fc1_Coh = nn.Linear(in_features=4608, out_features=1024, bias=False) #4608
        out_Coh = self.fc1_Coh(out_Coh)
        print(out_Coh.shape)

        out = torch.cat((out_S2, out_S1, out_Coh), dim=1)
        print(out.shape)
        self.fc2 = nn.Linear(in_features=3072, out_features=256, bias=False)
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

        out_S2 = x[:,:5,:,:]
        out_S2 = F.relu(self.conv1_S2(out_S2))
        out_S2 = self.maxpool1_S2(out_S2)
        out_S2 = F.relu(self.conv2_S2(out_S2))
        out_S2 = F.dropout2d(out_S2, p=self.dropout_rate, training=self.training)
        out_S2 = self.maxpool2_S2(out_S2)
        out_S2 = F.relu(self.conv3_S2(out_S2))
        out_S2 = F.relu(self.conv4_S2(out_S2))
        out_S2 = self.maxpool3_S2(out_S2)
        out_S2 = F.relu(self.conv5_S2(out_S2))
        out_S2 = F.dropout2d(out_S2, p=self.dropout_rate, training=self.training)
        out_S2 = F.relu(self.conv6_S2(out_S2))
        out_S2 = self.maxpool4_S2(out_S2)
        out_S2 = F.relu(self.conv7_S2(out_S2))
        out_S2 = F.relu(self.conv8_S2(out_S2))
        out_S2 = F.dropout2d(out_S2, p=self.dropout_rate, training=self.training)
        out_S2 = self.maxpool5_S2(out_S2)
        out_S2 = out_S2.view(out_S2.shape[0], -1)
        out_S2 = F.relu(self.fc1_S2(out_S2))

        out_S1 = x[:,5:7,:,:]
        out_S1 = F.relu(self.conv1_S1(out_S1))
        out_S1 = self.maxpool1_S1(out_S1)
        out_S1 = F.relu(self.conv2_S1(out_S1))
        out_S1 = F.dropout2d(out_S1, p=self.dropout_rate, training=self.training)
        out_S1 = self.maxpool2_S1(out_S1)
        out_S1 = F.relu(self.conv3_S1(out_S1))
        out_S1 = F.relu(self.conv4_S1(out_S1))
        out_S1 = self.maxpool3_S1(out_S1)
        out_S1 = F.relu(self.conv5_S1(out_S1))
        out_S1 = F.dropout2d(out_S1, p=self.dropout_rate, training=self.training)
        out_S1 = F.relu(self.conv6_S1(out_S1))
        out_S1 = self.maxpool4_S1(out_S1)
        out_S1 = F.relu(self.conv7_S1(out_S1))
        out_S1 = F.relu(self.conv8_S1(out_S1))
        out_S1 = F.dropout2d(out_S1, p=self.dropout_rate, training=self.training)
        out_S1 = self.maxpool5_S1(out_S1)
        out_S1 = out_S1.view(out_S1.shape[0], -1)
        out_S1 = F.relu(self.fc1_S1(out_S1))


        out_Coh = x[:,7:8,:,:]
        out_Coh = F.relu(self.conv1_Coh(out_Coh))
        out_Coh = self.maxpool1_Coh(out_Coh)
        out_Coh = F.relu(self.conv2_Coh(out_Coh))
        out_Coh = F.dropout2d(out_Coh, p=self.dropout_rate, training=self.training)
        out_Coh = self.maxpool2_Coh(out_Coh)
        out_Coh = F.relu(self.conv3_Coh(out_Coh))
        out_Coh = F.relu(self.conv4_Coh(out_Coh))
        out_Coh = self.maxpool3_Coh(out_Coh)
        out_Coh = F.relu(self.conv5_Coh(out_Coh))
        out_Coh = F.dropout2d(out_Coh, p=self.dropout_rate, training=self.training)
        out_Coh = F.relu(self.conv6_Coh(out_Coh))
        out_Coh = self.maxpool4_Coh(out_Coh)
        out_Coh = F.relu(self.conv7_Coh(out_Coh))
        out_Coh = F.relu(self.conv8_Coh(out_Coh))
        out_Coh = F.dropout2d(out_Coh, p=self.dropout_rate, training=self.training)
        out_Coh = self.maxpool5_Coh(out_Coh)
        out_Coh = out_Coh.view(out_Coh.shape[0], -1)
        out_Coh = F.relu(self.fc1_Coh(out_Coh))

        out = torch.cat((out_S2, out_S1, out_Coh), dim=1)
        out = F.dropout2d(out, p=self.dropout_rate, training=self.training)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        out = out.view(-1)
        return out