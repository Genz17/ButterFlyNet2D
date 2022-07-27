import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from MaskTransform  import maskTransfrom
from BlurTransform  import blurTransfrom
from NoiseTransform import noiseTransfrom

import numpy as np
from ButterFlyNet_Identical import ButterFlyNet_Identical



from inpaint_test_func import *
from denoise_test_func import *
from deblur_test_func  import *

testTypeList = ['inpainting','denoising','deblurring']
testType = testTypeList[0]

batch_size = 1024
local_size = 64
image_size = 64
data_path = '../../data/CelebaTest/'
para_path = '../Training/64_64_Celeba_square_inpainting.pth'
net_layer = 6 # should be no more than log_2(local_size)
cheb_num = 4

print('Generating Net...')
Net = ButterFlyNet_Identical(image_size, layer, chebNum, False)
print('Done.')

if testType == 'inpainting':
    transformCompose = 
elif testType == 'denoising':

else:

test_loader = DataLoader(
    torchvision.datasets.ImageFolder(root=data_path,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    ])),
    batch_size=batch_size, shuffle=False)
print('Loading parameters...')
Net.load_state_dict(torch.load(para_path))
print('Done.')

if testType == 'inpainting':
    test_inpainting(test_loader,batch_size,Net,eval('squareMask'+str(image_size))(torch.zeros(batch_size,1,image_size,image_size)).cuda(),image_size,local_size)
elif testType == 'denoising':
    noise_mean = 0
    noise_std = 0.1
    test_denoising(test_loader, batch_size, Net, noise_mean, noise_std, image_size, local_size)
elif testType == 'deblurring':
    pass