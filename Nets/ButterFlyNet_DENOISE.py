import math
import torch.nn as nn
from ButterFlyNet2D import ButterFlyNet2D
from ButterFlyNet2D_IDFT import ButterFlyNet2D_IDFT
from DeblurMatNet import deblurNet

class ButterFlyNet_DENOISE(nn.Module):
    # Testing: suppose input to be 64
    def __init__(self, image_size, layer, chebNum, prefix):
        super(ButterFlyNet_DENOISE,self).__init__()
        self.encoderset = ButterFlyNet2D(1, image_size, image_size, layer, chebNum, 0, image_size, 0, image_size, prefix, True).cuda()
        self.decoderset = ButterFlyNet2D_IDFT(1, 0, image_size, 0, image_size, image_size, image_size, layer, chebNum, prefix, True).cuda()
    def forward(self, inputdata):

        res_before = self.encoderset(inputdata)
        res = self.decoderset(res_before)

        return res