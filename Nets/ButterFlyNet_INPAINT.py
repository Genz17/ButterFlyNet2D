import torch
import torch.nn as nn
from ButterFlyNet2D import ButterFlyNet2D
from ButterFlyNet2D_IDFT import ButterFlyNet2D_IDFT

class ButterFlyNet_INPAINT(nn.Module):
    # Testing: suppose input to be 64
    def __init__(self, image_size, layer, chebNum, prefix):
        super(ButterFlyNet_INPAINT,self).__init__()
        self.image_size = image_size
        self.encoderset = ButterFlyNet2D(1, image_size, image_size, layer, chebNum, 0, image_size, 0, image_size, prefix, True).cuda()
        self.decoderset = ButterFlyNet2D_IDFT(1, 0, image_size, 0, image_size, image_size, image_size, layer, chebNum, prefix, True).cuda()
    def forward(self, inputdata):

        res_before = self.encoderset(inputdata)
        res = self.decoderset(res_before)

        return res

    def preFT(self,iter):
        optimizer_encoder = torch.optim.Adam(self.encoderset.parameters(),1e-3)
        optimizer_decoder = torch.optim.Adam(self.decoderset.parameters(),1e-3)
        print('FT Approximation...')
        for i in range(iter):
            data = torch.rand(50,1,self.image_size,self.image_size, device='cuda:0')
            data_ft = torch.fft.fft2(data)
            out = self.encoderset(data)
            optimizer.zero_grad()
            loss = torch.norm(out-data_ft)/torch.norm(data_ft)
            loss.backward()
            optimizer_encoder.step()
            print('rel err: {}'.format(loss.item()))
        print('Done.')

        print('IFT Approximation...')
        for i in range(iter):
            data = torch.rand(50,1,self.image_size,self.image_size, device='cuda:0')
            data_ft = torch.fft.fft2(data)
            out = self.decoderset(data_ft)
            optimizer.zero_grad()
            loss = torch.norm(out-data)/torch.norm(data)
            loss.backward()
            optimizer_decoder.step()
            print('rel err: {}'.format(loss.item()))
        print('Done.')