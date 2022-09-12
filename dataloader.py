# -*- coding: utf-8 -*-
import os
import cv2
import torch
import datetime
import numpy as np
from torch.utils.data import Dataset


# np.load()

class HeatmapDataset(Dataset):
    def __init__(self, dataset, clip_len=16, preprocess=False): #clip_len  has to be greater than the number of frames 
        self.root_dir= dataset
        self.clip_len = clip_len
        
        self.frame_block_dirs=[]  
        self.video_folders=os.listdir(self.root_dir)
        self.frames=[]
        frames_num=0 
        for f in self.video_folders: 
            folderdir=os.listdir(os.path.join(self.root_dir,f))
            folder_len=len(folderdir)
            for idx in range(folder_len-self.clip_len+1):
                frame_dir_block=[]
                for fd in folderdir[idx:idx+clip_len]:
                    frame_dir_block.append(os.path.join(self.root_dir,f,fd))
                self.frame_block_dirs.append(frame_dir_block)
            frames_num+= (folder_len-self.clip_len+1)
        self.frames_num=frames_num
        assert self.frames_num==len(self.frame_block_dirs)
    
    def __len__(self):
        return self.frames_num

    def __getitem__(self, index):

        np_frame_list=[np.expand_dims(np.load(frame_dir).astype(np.float32),2) for frame_dir in self.frame_block_dirs[index]]
        np_buffer=np.stack(np_frame_list,axis=0)
        
        label= os.path.basename(self.frame_block_dirs[index][0]).split('_')[12]
        # print('comfort value%s'%label)
        if int(label) == 0:
            label=0 #0 - comfort
        else:
            label=1 #1 -discomfort
        # print(np_buffer.shape)
        np_buffer = np_buffer.transpose((3, 0, 1, 2)) #Conv3D has input (N,C,D,H,W)，compare multiple [H,W,C] and form [D,H,W,C] using transpose
        return torch.from_numpy(np_buffer), label
