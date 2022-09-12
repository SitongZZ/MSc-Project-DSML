# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

def L2Normalize(x,power): 
    norm = x.pow(power).sum(1, keepdim=True).pow(1./power)
    out = x.div(norm)
    return out


class VGG163D(nn.Module):
    """
    comapre the front and back layer method for accuracy and comdel complexity
    """

    def __init__(self, num_classes): 
        super(VGG163D, self).__init__()
        
        
        self.conv1a = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1a=nn.BatchNorm3d(64)
        self.conv1b = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1b=nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) 
        self.conv2a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2a=nn.BatchNorm3d(128)
        self.conv2b = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2b=nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2))            

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3a=nn.BatchNorm3d(256)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3b=nn.BatchNorm3d(256)
        self.conv3c = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3c=nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4a=nn.BatchNorm3d(512)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4b=nn.BatchNorm3d(512)
        self.conv4c = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4c=nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5a=nn.BatchNorm3d(512)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5b=nn.BatchNorm3d(512)
        self.conv5c = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5c=nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        self.fc6 = nn.Linear(50688, 4096)  
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        
        self.dropout = nn.Dropout(p=0.5)
        #self.dropout3=nn.Dropout3d(p=0.5, inplace=False)
        self.relu = nn.ReLU()

        self.__init_weight()


    def forward(self, x):
        # print(x.shape)
        x = self.relu(self.conv1a(x))
        x= self.bn1a(x)
        # print(x.shape)torch.Size([bs, 64, 16, 256, 320])
        x = self.relu(self.conv1b(x))
        x= self.bn1b(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        # print('===================1=====================')
        x = self.relu(self.conv2a(x))
        x = self.bn2a(x)
        # print(x.shape)
        x = self.relu(self.conv2b(x))
        x = self.bn2b(x)
        x = self.pool2(x)
        # print(x.shape)
        # print('===================2=====================')
        x = self.relu(self.conv3a(x))
        x = self.bn3a(x)
        x = self.relu(self.conv3b(x))
        x = self.bn3b(x)
        x = self.relu(self.conv3c(x))
        x = self.bn3c(x)
        x = self.pool3(x)
        # print(x.shape)
        # print('===================3=====================')
        x = self.relu(self.conv4a(x))
        x = self.bn4a(x)
        x = self.relu(self.conv4b(x))
        x = self.bn4b(x)
        x = self.relu(self.conv4c(x))
        x = self.bn4c(x)
        x = self.pool4(x)
        # print(x.shape)
        # print('===================4=====================')
        x = self.relu(self.conv5a(x))
        x = self.bn5a(x)
        x = self.relu(self.conv5b(x))
        x = self.bn5b(x)
        x = self.relu(self.conv5c(x))
        x = self.bn5c(x)
        x = self.pool5(x)
        # print(x.shape)
        # print('===================5=====================')
        x = x.view(-1, 50688) #8192
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model): 
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1a,model.conv1b, model.conv2a,model.conv2b, model.conv3a, model.conv3b,model.conv3c, model.conv4a, model.conv4b,model.conv4c,
         model.conv5a, model.conv5b, model.conv5c, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k
