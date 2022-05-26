import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from ButterFlyNet2D_Flexible import ButterFlyNet2D_Flexible
import matplotlib.pyplot as plt
from SquareMask import squareMask

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./data/', train=True, download=False,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor()])),
    batch_size=1, shuffle=True)

Net = ButterFlyNet2D_Flexible(3, 32, 32, 4, 4, 0, 32, 0, 32, False, [(4, 4), (2, 2), (2, 2),(2, 2),(2, 2)], ['Max', 'Max', 'Max','Max','Max'], False, True).cuda()
Net.load_state_dict(torch.load('net_params4.pth'))
print(Net)
for step, (image, label) in enumerate(train_loader):
    with torch.no_grad():
        maskedimage = (image * squareMask(image)).cuda()
        fourierimage = torch.tensor([[np.fft.fft2(image)]]).cuda()
    output = Net(maskedimage)
    loss = torch.norm(torch.fft.ifft2(output).cpu() - image) / torch.norm(image)
    print(loss)
    with torch.no_grad():
        plt.imshow(np.transpose(image[0].detach().numpy(), (1, 2, 0)))
        plt.show()
        plt.imshow(np.transpose(maskedimage.cpu().detach().numpy()[0], (1, 2, 0)))
        plt.show()
        plt.imshow((np.transpose(np.fft.ifft2(output.cpu().detach().numpy()[0]).real, (1, 2, 0))))
        plt.show()