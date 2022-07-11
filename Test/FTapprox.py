import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ButterFlyNet2D import ButterFlyNet2D

size = 64
layer_number = 64
cheb_num = 4

Net = ButterFlyNet2D(1, size, size, layer_number, cheb_num, 0, size, 0, size, 0, False)