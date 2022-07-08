import torch
import torchvision
from torch.utils.data import DataLoader
from ButterFlyNet2D_Flexible import ButterFlyNet2D_Flexible
import matplotlib.pyplot as plt
import torch.nn as nn
from Gaussian_Func import gauss
from ButterFlyNet2D_IDFT import ButterFlyNet2D_IDFT
from DeblurMatNet import deblurNet
import numpy as np

epochs = 1
batch_size_train = 1
learning_rate = 0.001
log_interval = 1
momentum = 0.5

train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=True, download=False,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((64,64))])),
    batch_size=batch_size_train, shuffle=False)

blurkernel = torch.zeros((batch_size_train,3,5,5)).cuda()
for channel in range(3):
    blurkernel[:,channel,:,:] = torch.tensor(gauss(0,0,2.5,(5,5)))
Netencoder = ButterFlyNet2D_Flexible(3, 64, 64, 5, 5, 0, 64, 0, 64, False, [(4,4),(2,2), (2,2),(2,2),(2,2)], ['Max', 'Max', 'Max','Max','Max'], False, True).cuda()
Netdecoder = ButterFlyNet2D_IDFT(3, 0, 64, 0, 64, 64, 64, 5, 5, False, False).cuda()
Netlineardeblur = deblurNet(2.5, (5,5), 64, 64, (64, 64)).cuda()

Netencoder.load_state_dict(torch.load('55encoderfordeblurSTL10.pth'))
Netdecoder.load_state_dict(torch.load('55decoderfordeblurSTL10.pth'))
Netlineardeblur.load_state_dict(torch.load('55linearfordeblurSTL10.pth'))

for epoch in range(epochs):
    for step, (image, label) in enumerate(train_loader):
        with torch.no_grad():
            # image = image.cuda()
            imagecolor = torch.zeros((1, 3, 64, 64))
            for channel in range(3):
                imagecolor[:, channel, :, :] = image
            image = imagecolor.cuda()
            fourierimage = torch.fft.fft2(image).cuda()
            maskedimage = (torch.fft.ifft2(fourierimage*torch.fft.fft2(blurkernel,(64,64)))).real

        output = Netencoder(maskedimage)
        # output = torch.fft.fft2(maskedimage)
        output_scooched = output.view((batch_size_train,3,1,-1))
        output_deblurred = Netlineardeblur(output_scooched)
        output_back = output_deblurred.view(output.shape)
        # output_done = torch.fft.ifft2(output_back)
        output_done = Netdecoder(output_back).real
        before = -10*np.log10((torch.norm(maskedimage - image, 'fro').item())**2/(3*64*64))
        after = -10*np.log10((torch.norm(output_done - image, 'fro').item())**2/(3*64*64))
        plt.imshow(np.transpose(image[0].cpu().detach().numpy(), (1, 2, 0)))
        plt.show()
        plt.imshow(np.transpose((maskedimage.cpu().detach().numpy()[0]).real, (1, 2, 0)))
        plt.show()
        plt.imshow((np.transpose((output_done.cpu().detach().numpy()[0]), (1, 2, 0))))
        plt.show()
        print('Before: {0} \t After: {1}'.format(before,after))