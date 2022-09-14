import torch
import torch.nn as nn
from ButterFlyNet2D import ButterFlyNet2D
from ButterFlyNet2D_IDFT import ButterFlyNet2D_IDFT

class ButterFlyNet_Identical(nn.Module):
    # Testing: suppose input to be 64
    def __init__(self, image_size, layer, chebNum, initMethod='kaimingU', part='All'):
        super(ButterFlyNet_Identical,self).__init__()
        self.part       = part
        self.image_size = image_size
        if self.part == 'All':
            self.encoderset = ButterFlyNet2D(1, image_size, image_size, layer, chebNum, 0, image_size, 0, image_size, initMethod, True).cuda()
            self.decoderset = ButterFlyNet2D_IDFT(1, 0, image_size, 0, image_size, image_size, image_size, layer, chebNum, initMethod, True).cuda()
        if self.part == 'f':
            self.encoderset = ButterFlyNet2D(1, image_size, image_size, layer, chebNum, 0, image_size, 0, image_size, initMethod, False).cuda()
            self.decoderset = lambda x:x
        if self.part == 'b':
            self.encoderset = lambda x:x
            self.decoderset = ButterFlyNet2D_IDFT(1, 0, image_size, 0, image_size, image_size, image_size, layer, chebNum, initMethod, False).cuda()
    def forward(self, inputdata):

        res = self.decoderset(self.encoderset(inputdata))

        return res

    def pretrain(self,iter):
        if self.part == 'All' or 'f':
            optimizer_encoder = torch.optim.Adam(self.encoderset.parameters(),1e-3)
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
        if self.part == 'All' or 'b':
            optimizer_decoder = torch.optim.Adam(self.decoderset.parameters(),1e-3)
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
