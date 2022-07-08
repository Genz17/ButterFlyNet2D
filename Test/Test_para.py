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


local_size = 64
Net = ButterFlyNet_INPAINT(local_size,6,4,False)

num = 0
for para in Net.parameters():
    num+=torch.prod(torch.tensor(para.shape))
print(num)