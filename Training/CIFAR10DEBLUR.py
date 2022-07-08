import torch
import torchvision
from torch.utils.data import DataLoader
from ButterFlyNet_DEBLUR import ButterFlyNet_DEBLUR
import matplotlib.pyplot as plt
from Gaussian_Func import gauss
import numpy as np

epochs = 12
batch_size_train = 100
learning_rate = 5e-4
log_interval = 1

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./data/',train=True,download=False,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.Resize((32,32))])),
    batch_size=batch_size_train, shuffle=True)

blurkernel = torch.zeros((batch_size_train,1,5,5)).cuda()
blurkernel[:,:,:,:] = torch.tensor(gauss(0,0,2.5,(5,5)))
Net = ButterFlyNet_DEBLUR(4,1).cuda()
optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.88, patience=30, verbose=True,
                                                        threshold=0.00005, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-16)

for epoch in range(epochs):
    for step, (image, label) in enumerate(train_loader):
        with torch.no_grad():
            image = image.cuda()
            fourierimage = torch.fft.fft2(image).cuda()
            maskedimage = (torch.fft.ifft2(fourierimage*torch.fft.fft2(blurkernel,(32,32)))).real
        optimizer.zero_grad()
        output = Net(maskedimage)
        loss = torch.norm(output - image) / torch.norm(image)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        print('CIFAR10DEBLUR: Train Epoch: {}, [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, step * len(image),
                                                                                len(train_loader.dataset),
                                                                                100 * step / len(train_loader),
                                                                                loss.item()))

torch.save(Net.state_dict(), '55GRAYCIFAR10deblurring.pth')
