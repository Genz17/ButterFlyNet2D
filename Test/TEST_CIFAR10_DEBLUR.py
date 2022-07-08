import torch
import torchvision
from torch.utils.data import DataLoader
from ButterFlyNet_DEBLUR import ButterFlyNet_DEBLUR
from Gaussian_Func import gauss
import numpy as np
import matplotlib.pyplot as plt

epochs = 1
batch_size = 256

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./data/',train=False,download=False,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((32,32))])),
    batch_size=batch_size, shuffle=True)

blurkernel = torch.zeros((batch_size,1,5,5)).cuda()
blurkernel[:,:,:,:] = torch.tensor(gauss(0,0,2.5,(5,5)))
Net = ButterFlyNet_DEBLUR(32,5).cuda()
Net.load_state_dict(torch.load('55GRAYCIFAR10deblurring.pth'))

for epoch in range(epochs):
    for step, (image, label) in enumerate(train_loader):
        with torch.no_grad():
            image = image.cuda()
            fourierimage = torch.fft.fft2(image).cuda()
            maskedimage = (torch.fft.ifft2(fourierimage*torch.fft.fft2(blurkernel,(32,32)))).real
        output_done = torch.zeros(batch_size, 3, 32, 32).cuda()
        for i in range(3):
            output_done[:, i:i + 1, :, :] = Net(maskedimage[:, i:i + 1, :, :])
        before = sum([-10 * np.log10((torch.norm(maskedimage[i] - image[i], 'fro').item())
                                     ** 2 / (3 * 32 * 32)) for i in range(batch_size)]) / batch_size
        after = sum([-10 * np.log10((torch.norm(output_done[i] - image[i], 'fro').item())
                                    ** 2 / (3 * 32 * 32)) for i in range(batch_size)]) / batch_size

        plt.imshow(torch.permute(image[0].cpu(), (1, 2, 0)))
        plt.show()
        plt.imshow(torch.permute((maskedimage[0].cpu()), (1, 2, 0)))
        plt.show()
        plt.imshow((np.transpose((output_done.cpu().detach().numpy()[0]), (1, 2, 0))))
        plt.show()

        print('Before: {0} \t After: {1}'.format(before, after))


