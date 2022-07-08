import torch
import torch.nn as nn

class zeroNet(nn.Module):
    def __init__(self, height, width):
        super(zeroNet, self).__init__()
        self.linear = nn.Linear(height*width, height*width, bias=True)
        self.linear.weight = nn.Parameter(torch.zeros((height*width,height*width), dtype=torch.complex64))

    def forward(self, input_data):
        return self.linear(input_data)
