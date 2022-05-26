import torch
import torch.nn as nn

class identicleNet(nn.Module):
    def __init__(self, height, width):
        super(identicleNet, self).__init__()
        self.linear = nn.Linear(height*width, height*width, bias=False)
        self.linear.weight = nn.Parameter(torch.eye(self.linear.weight.shape[0],dtype=torch.complex64))

    def forward(self, input_data):
        return self.linear(input_data)

