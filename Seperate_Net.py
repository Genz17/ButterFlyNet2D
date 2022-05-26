import torch.nn as nn
from ButterFlyNet2D import ButterFlyNet2D
from ButterFlyNet2D_IDFT import ButterFlyNet2D_IDFT

class Seperate_Net(nn.Module):
    # Testing: suppose input to be 64
    def __init__(self, image_size,layer,chebNum,Freq_Left,Freq_Right,prefix):
        super(Seperate_Net,self).__init__()
        self.image_size = image_size

        self.encoder = ButterFlyNet2D(1, image_size, image_size, layer, chebNum,
                                                                Freq_Left, Freq_Right,
                                                                Freq_Left, Freq_Right,
                                                                prefix, True).cuda()
        self.decoder = ButterFlyNet2D_IDFT(1, Freq_Left, Freq_Right,
                                            Freq_Left, Freq_Right,
                                            image_size, image_size, layer, chebNum,
                                            prefix, True).cuda()
    def forward(self, inputdata):

        res_before = self.encoder(inputdata)
        res = self.decoder(res_before)

        return res