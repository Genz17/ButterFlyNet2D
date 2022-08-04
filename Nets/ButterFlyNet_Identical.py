import torch
import torch.nn as nn
from ButterFlyNet2D import ButterFlyNet2D
from ButterFlyNet2D_IDFT import ButterFlyNet2D_IDFT

class ButterFlyNet_Identical(nn.Module):
    # Testing: suppose input to be 64
    def __init__(self, image_size, layer, chebNum, prefix=False):
        super(ButterFlyNet_Identical,self).__init__()
        self.image_size = image_size
        self.encoderset = ButterFlyNet2D(1, image_size, image_size, layer, chebNum, 0, image_size, 0, image_size, prefix, True).cuda()
        self.decoderset = ButterFlyNet2D_IDFT(1, 0, image_size, 0, image_size, image_size, image_size, layer, chebNum, prefix, True).cuda()
    def forward(self, inputdata):

        res = self.decoderset(self.encoderset(inputdata))

        return res

    def pretrain(self,iter):
        optimizer_encoder = torch.optim.Adam(self.encoderset.parameters(),1e-3)
        optimizer_decoder = torch.optim.Adam(self.decoderset.parameters(),1e-3)
        print('Fourier Transform Approximation...')
        for i in range(iter):
            data = torch.rand(20,1,self.image_size,self.image_size, device='cuda:0')
            data_ft = torch.fft.fft2(data)
            out = self.encoderset(data)
            optimizer_encoder.zero_grad()
            loss = torch.norm(out-data_ft)/torch.norm(data_ft)
            loss.backward()
            optimizer_encoder.step()
            print('{}/{},\trel err: {}'.format(i+1,iter,loss.item()))
        print('Done.')

        print('Inverse Fourier Transform Approximation...')
        for i in range(iter):
            data = torch.rand(20,1,self.image_size,self.image_size, device='cuda:0')
            data_ft = torch.fft.fft2(data)
            out = self.decoderset(data_ft)
            optimizer_decoder.zero_grad()
            loss = torch.norm(out-data)/torch.norm(data)
            loss.backward()
            optimizer_decoder.step()
            print('{}/{},\trel err: {}'.format(i+1,iter,loss.item()))
        print('Done.')