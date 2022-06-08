import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Mask import squareMask64,lineMask256,squareMask128,squareMask256
import numpy as np
from ButterFlyNet_INPAINT import ButterFlyNet_INPAINT
from Test_INPAINT_Types import *


epochs = 1
batch_size = 1
image_size = 256

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(root='../../CELEBA/',
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size))])),
    batch_size=batch_size, shuffle=False)

local_size = 32
Net = ButterFlyNet_INPAINT(local_size,5,4,False).cuda()



print('Loading parameters...')
Net.load_state_dict(torch.load('../../PTHS/54GRAYLineSeperateCelebainpainting.pth'))
print('Done.')


Rmask = eval('lineMask' + str(image_size))(torch.zeros(batch_size,3,image_size,image_size)).cuda()

overlap(train_loader,batch_size,Net,Rmask,image_size,local_size)
