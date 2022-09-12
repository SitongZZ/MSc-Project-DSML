# -*- coding: utf-8 -*-

import os
import numpy
import cv2

mask_root='./seg_labels'
np_root='./P01THERMAL'
func=2 #func 1= dot product, func 2 = concate
factors=[0, 1.2, 1.4, 1.6 ,1.8,2.0]
0,1,2,3,4,5

for folder in os.listdir(mask_root):
    savef='mask1-'+folder
    savedir=os.path.join(np_root,savef)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    masks=os.path.join(mask_root, folder)
    nparrs=os.path.join(np_root, folder)
    for f in os.listdir(masks):
        mask=cv2.imread(masks+'/'+f,0) 
        nparr=numpy.load(nparrs+'/'+f.replace('.png', '.npy'))
        
        if func==1:
            factor=mask*0.2+1
            factor[factor==1] = 0
            # print(numpy.unique(factor))
            # print(factor.shape)
            # print(mask.shape)
            nparr=nparr*factor
            
        else:
            nparr=numpy.stack([mask,nparr],axis=-1)
            # print(nparr.shape)
            # print(numpy.unique(nparr))
        numpy.save(savedir+'/'+f.replace('.png', '.npy'), nparr)

