import torch
from Gaussian_Func import gauss
import numpy as np
import torch.nn as nn

class deblurNet(nn.Module):
    def __init__(self, std, kernel_size, in_height, in_width, original_shape):
        super(deblurNet, self).__init__()
        self.in_height = in_height
        self.in_width = in_width
        self.blurmat = self.generate_blurmat(std, kernel_size, original_shape)
        self.linear = nn.Linear(in_width*in_height, in_width*in_height,bias=False)
        self.linear.weight = self.generate_linear_weights()

    def forward(self, input_data):
        return self.linear(input_data)

    def generate_linear_weights(self):
        weight = np.zeros((self.in_height*self.in_width, self.in_height*self.in_width), dtype=complex)
        for i in range(self.in_height*self.in_width):
            weight[i][i] = 1/(self.blurmat[i//self.in_width][i%self.in_width])
        return nn.Parameter(torch.tensor(weight, dtype=torch.complex64, device='cuda:0'))

    def generate_blurmat(self,std, kernel_size, original_shape):
        blurmat = np.fft.fft2(gauss(0, 0, std, kernel_size), original_shape)
        return blurmat
