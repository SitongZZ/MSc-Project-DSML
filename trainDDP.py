# -*- coding: utf-8 -*-

import os
import timeit
from datetime import datetime
import socket
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from datasets.ZSTdataset import HeatmapDataset
from network2 import VGG16_3D #from model.py directory import model.py
import math


#Dis
import torch.utils.data.distributed
import torch.distributed
import argparse

#command: python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=12355 trainDDP.py



parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)  
args = parser.parse_args()
device = torch.device('cuda:{}'.format(args.local_rank))
torch.cuda.set_device(args.local_rank)


#parameter
nEpochs = 10  
resume_epoch = 0  #recovery
resume_dir=r'./run/run_7/models/VGG3D_epoch-4.pth.tar'
bs=6 
useTest = True
lrwarmup_epoch=4
nTestInterval = 5
saveepochs = 2
lr = 1e-4
min_lr=1e-5
cls_num=2
data_roots=[r'Dataset/train',r'Dataset/validation',r'Dataset/test']
clip_len=16 #time depth - number of frames in one input tensor

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'VGG3D'
saveName = modelName



def warmup_and_cosine_learning_rate(optimizer, epoch,use_cosine=False):
    """from 1/n, and get warmed up to certain lr and derease"""
    if epoch < lrwarmup_epoch:
        thelr = lr * (1+epoch) / lrwarmup_epoch 
        for param_group in optimizer.param_groups:
                param_group["lr"] = thelr
    else:
        if use_cosine:
            thelr = min_lr + (lr - min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - lrwarmup_epoch) / (nEpochs - lrwarmup_epoch)))
            for param_group in optimizer.param_groups:
                    param_group["lr"] = thelr



def train_model(dataset=data_roots, save_dir=save_dir, num_classes=cls_num, lr=lr,
                num_epochs=nEpochs, saveepochs=saveepochs, useTest=useTest, test_interval=nTestInterval):

    torch.distributed.init_process_group("nccl", init_method='env://')
    model = VGG16_3D.VGG163D(num_classes=num_classes)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    DDPmodel= torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],
                                                    output_device=args.local_rank)
    
    train_params = [{'params': VGG16_3D.get_1x_lr_params(model), 'lr': lr},
                        {'params': VGG16_3D.get_10x_lr_params(model), 'lr': lr * 10}]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4,
                                          gamma=0.1)

    if resume_epoch == 0:
        print("Training {} Kaiming Initialization...".format(modelName))
    else:
        checkpoint = torch.load(resume_dir,map_location=lambda storage, loc: storage)
        print("Initializing weights from: {}...".format(resume_dir))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 2.0**20))


    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    os.makedirs(log_dir, exist_ok=True) 
    global_rank=torch.distributed.get_rank()
    if global_rank==0:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(HeatmapDataset(dataset=dataset[0],clip_len=clip_len), batch_size=bs, shuffle=True, num_workers=8)
    val_dataloader   = DataLoader(HeatmapDataset(dataset=dataset[1],clip_len=clip_len), batch_size=bs, num_workers=8)
    test_dataloader  = DataLoader(HeatmapDataset(dataset=dataset[2],clip_len=clip_len), batch_size=bs, num_workers=8)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    
    
    for epoch in range(resume_epoch, num_epochs):

        for phase in ['train', 'val']:
            bg_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0
            if phase == 'train':
                if epoch>lrwarmup_epoch:
                    scheduler.step()
                # if global_rank==0:
                #     print(optimizer.param_groups[0]['lr'])
                warmup_and_cosine_learning_rate(optimizer,epoch)
                # if global_rank==0:
                #     print(optimizer.param_groups[0]['lr'])
                DDPmodel.train()
            else:
                DDPmodel.eval()
            
            for inputs, labels in tqdm(trainval_loaders[phase]):
                inputs = Variable(inputs, requires_grad=True).to(args.local_rank,non_blocking=True)
                labels = Variable(labels).to(args.local_rank,non_blocking=True)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = DDPmodel(inputs)

                else:
                    with torch.no_grad():
                        outputs = DDPmodel(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                labels=labels.long()
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                if writer is not None:
                    writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                    writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                if writer is not None:
                    writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                    writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)
            if global_rank==0:
                print("[{}] Epoch: {}/{} || Acc: {} || Loss: {} ".format(phase, epoch+1, nEpochs, epoch_acc, epoch_loss))
            st_time = timeit.default_timer()
            if global_rank==0:
                print(phase+"per epoch time: " + str(st_time - bg_time) + "\n")

        if epoch % saveepochs == (saveepochs - 1) and global_rank==0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            if global_rank==0:
                print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            bg_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(args.local_rank,non_blocking=True)
                labels = labels.to(args.local_rank,non_blocking=True)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size
            if writer is not None:
                writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)
            if global_rank==0:
                print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            st_time = timeit.default_timer()
            if global_rank==0:
                print("Execution time length:" + str(st_time - bg_time) + "\n")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    train_model()


