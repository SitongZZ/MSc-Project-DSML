import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from datasets.ZSTdataset import HeatmapDataset
from network2 import VGG16_3D

import ptflops


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 10  
resume_epoch = 0  
resume_dir=r'./results/network2_raw_16/VGG3D_epoch-9.pth.tar'
bs=2 
useTest = True
lrwarmup_epoch=4
nTestInterval = 10
saveepochs = 5
lr = 1e-4
min_lr=1e-5
cls_num=2
data_roots=[r'Dataset/train',r'Dataset/validation',r'Dataset/test']
clip_len=16 



model = VGG16_3D.VGG163D(num_classes=cls_num)
if resume_epoch == 0:
        print("Kaiming Initialization...")
else:
        checkpoint = torch.load(resume_dir,map_location=lambda storage, loc: storage)
        print("Initializing weights from: {}...".format(resume_dir))
        model.load_state_dict(checkpoint['state_dict'])

print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000 )) #2**20

FLOPs , Params=ptflops.flops_counter.get_model_complexity_info(model, (1, 16, 256, 320), as_strings=True, print_per_layer_stat=True, verbose=True)
print(FLOPs)
print(Params)
