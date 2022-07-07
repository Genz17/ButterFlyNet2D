import math
import torch.nn as nn
from ButterFlyNet2D import ButterFlyNet2D
from ButterFlyNet2D_IDFT import ButterFlyNet2D_IDFT
from DeblurMatNet import deblurNet

class ButterFlyNet_DEBLUR(nn.Module):
    # Testing: suppose input to be 64
    def __init__(self, image_size, chebNum):
        super(ButterFlyNet_DEBLUR,self).__init__()
        layer = int(math.log2(image_size))
        self.encoderset = ButterFlyNet2D(1, image_size, image_size, layer, chebNum, 0, image_size, 0, image_size, False,
                                          True).cuda()
        self.decoderset = ButterFlyNet2D_IDFT(1, 0, image_size, 0, image_size, image_size, image_size, layer, chebNum,
                                               False, True).cuda()
        self.linear = deblurNet(2.5,(5,5),image_size,image_size,(image_size,image_size))
    def forward(self, inputdata):
        batch_size = inputdata.shape[0]
        res_before = self.encoderset(inputdata)
        out1 = res_before.view(batch_size,-1)
        output = self.linear(out1)
        out2 = output.view(res_before.shape)
        res = self.decoderset(out2)

        return res