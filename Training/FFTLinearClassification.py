import math
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from Test import test
from Train import train

class FFTLinear_Classification(nn.Module):
    # Testing: suppose input to be 64
    def __init__(self, image_size, out_feature):
        super(FFTLinear_Classification,self).__init__()
        self.image_size = image_size
        self.linear = nn.Linear((image_size**2)*4, out_feature)
    def forward(self, inputdata):

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


epochs = 10
batch_size_train = 200
batch_size_test = 1000
learning_rate = 0.001

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
                                    torchvision.transforms.RandomRotation(10),
                                    torchvision.transforms.RandomHorizontalFlip(p=0.1),
                                    torchvision.transforms.Resize((32,32)),
                                    torchvision.transforms.ToTensor()])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((32,32))])),
    batch_size=batch_size_test, shuffle=True)

Net = FFTLinear_Classification(32,10).cuda()
optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
schedualer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=4, verbose=True,
                                                        threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
num=0
for para in Net.parameters():
    num+=torch.prod(torch.tensor(para.shape))
print(num)

loss_func = nn.CrossEntropyLoss().cuda()
train_losses = []
train_counter = []
train_test_acc = []
train_test_counter = []
test_acc = []
test_counter = []

for epoch in range(epochs):
    train(epoch,loss_func,optimizer,Net, train_loader, batch_size_train,
                train_counter,train_losses)
    with torch.no_grad():
        test_counter.append(epoch)
        test(loss_func, Net, test_loader,test_acc)
        train_test_counter.append(epoch)
        schedualer.step(test_acc[-1])

torch.save(Net.state_dict(), 'FFTLinearClassification.pth')