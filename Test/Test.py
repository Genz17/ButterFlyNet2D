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
from ButterFlyNet_DENOISE import ButterFlyNet_DENOISE
from inpaint_test_func import *
from denoise_test_func import *

testTypeList = ['inpainting','denoising','deblurring']
testType = testTypeList[0]

batch_size = 256
local_size = 64
image_size = 256
data_path = '../../CELEBA/'
para_path = ''
net_layer = 6 # should be no more than log_2(local_size)
cheb_num = 4

if testType == 'inpainting':
    Net = ButterFlyNet_INPAINT(local_size, net_layer, cheb_num, False).cuda()
elif testType == 'denoising':
    Net = ButterFlyNet_DENOISE(local_size, net_layer, cheb_num, False).cuda()
elif testType == 'deblurring':
    pass
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(root=data_path,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size))])),
    batch_size=batch_size, shuffle=False)


print('Loading parameters...')
Net.load_state_dict(torch.load(para_path))
print('Done.')

if testType == 'inpainting':
    test_inpainting(test_loader,batch_size,Net,eval('squareMask'+str(image_size))(torch.zeros(batch_size,1,image_size,image_size)).cuda(),image_size,local_size)
elif testType == 'denoising':
    noise_mean = 0
    noise_std = 0.1
    test_denoising(test_loader, batch_size, Net, torch.normal(mean=noise_mean,std=noise_std,size=(batch_size,1,image_size,image_size),device='cuda:0'), image_size, local_size)
elif testType == 'deblurring':
    pass