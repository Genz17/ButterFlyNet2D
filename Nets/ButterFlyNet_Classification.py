import math
import torch
import torch.nn as nn
from ButterFlyNet2D import ButterFlyNet2D

class ButterFlyNet_Classification(nn.Module):
    # Testing: suppose input to be 64
    def __init__(self, image_size, chebNum,splitNum, out_feature):
        super(ButterFlyNet_Classification,self).__init__()
        self.image_size = image_size
        self.splitNum = splitNum
        layer = int(math.log2(image_size//splitNum))
        self.freq_split = image_size//splitNum
        self.encoderset = nn.ModuleDict({str(i):ButterFlyNet2D(1, image_size, image_size, layer, chebNum,
                                                               i*self.freq_split, (i+1)*self.freq_split, i*self.freq_split, (i+1)*self.freq_split,
                                                               True, True).cuda() for i in range(splitNum)})
        self.linear = nn.Linear((image_size**2)*4, out_feature)
    def forward(self, inputdata):

        batch_size = inputdata.shape[0]
        out = torch.zeros(batch_size,1,self.image_size, self.image_size, device='cuda:0', dtype=torch.complex64)
        for i in range(self.splitNum):
            out[:,:,i*self.freq_split:(i+1)*self.freq_split,i*self.freq_split:(i+1)*self.freq_split] = self.encoderset[str(i)](inputdata)
        out = self.split(out)
        out = out.view(batch_size,-1)
        res = self.linear(out)

        return res

    def pretrain(self, inputdata):

        batch_size = inputdata.shape[0]
        out = torch.fft.fft2(inputdata)
        out = self.split(out)
        out = out.view(batch_size, -1)
        res = self.linear(out)

        return res

    def split(self, input_data):
        data_split = torch.zeros((input_data.shape[0],
                     4*input_data.shape[1],
                     input_data.shape[2],
                     input_data.shape[3]), dtype=torch.float32, device='cuda:0')

        for channel in range(input_data.shape[1]):
            data_split[:, 4 * channel, :, :] = \
                input_data[:, channel, :, :].real
            data_split[:, 4 * channel + 1, :, :] = \
                input_data[:, channel, :, :].imag
            data_split[:, 4 * channel + 2, :, :] = \
                -input_data[:, channel, :, :].real
            data_split[:, 4 * channel + 3, :, :] = \
                -input_data[:, channel, :, :].imag
        data_split = nn.ReLU(inplace=True)(data_split)
        return data_split