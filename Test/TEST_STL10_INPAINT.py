import torch
import torchvision
from torch.utils.data import DataLoader
from ButterFlyNet2D import ButterFlyNet2D
import matplotlib.pyplot as plt
from ButterFlyNet2D_IDFT import ButterFlyNet2D_IDFT
from SquareMask64 import squareMask64
import numpy as np
from IdenticleNet import identicleNet

epochs = 1
batch_size = 256


train_loader = DataLoader(
    torchvision.datasets.STL10('./data/',
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((64,64))])),
    batch_size=batch_size, shuffle=False)

Netencoder = ButterFlyNet2D(3, 64, 64, 5, 5, 0, 64, 0, 64, False, True).cuda()
num = 0
for para in Netencoder.parameters():
    num+=torch.prod(torch.tensor(para.shape))
print(num)
Netdecoder = ButterFlyNet2D_IDFT(3, 0, 64, 0, 64, 64, 64, 5, 5, False, True).cuda()
Netlinear = identicleNet(64, 64).cuda()
num = 0
for para in Netencoder.parameters():
    num+=torch.prod(torch.tensor(para.shape))
print(num)

print('Loading parameters...')
Netencoder.load_state_dict(torch.load('55encoderforinpaintSTL10.pth'))
Netdecoder.load_state_dict(torch.load('55decoderforinpaintSTL10.pth'))
Netlinear.load_state_dict(torch.load('55linearforinpaintSTL10.pth'))
print('Done.')

for epoch in range(epochs):
    for step, (image, label) in enumerate(train_loader):
        with torch.no_grad():
            image = image.cuda()
            # imagecolor = torch.zeros((1,3,64,64))
            # for channel in range(3):
            #     imagecolor[:,channel,:,:] = image
            # image = imagecolor.cuda()
            maskedimage = (image * squareMask64(image))
            
        output = Netencoder(maskedimage)
        output_scooched = output.view((batch_size, 3, 1, -1))
        output_inpaint = Netlinear(output_scooched)
        output_back = output_inpaint.view(output.shape)
        output_done = Netdecoder(output)
        
        before = sum([-10 * np.log10((torch.norm(maskedimage[i] - image[i], 'fro').item())
                                     ** 2 / (3 * 64 * 64)) for i in range(batch_size)])/batch_size
        after = sum([-10 * np.log10((torch.norm(output_done[i] - image[i], 'fro').item())
                                    ** 2 / (3 * 64 * 64)) for i in range(batch_size)])/batch_size
        plt.imshow(np.transpose(image[0].cpu().detach().numpy(), (1, 2, 0)))
        plt.show()
        plt.imshow(np.transpose((maskedimage.cpu().detach().numpy()[0]).real, (1, 2, 0)))
        plt.show()
        plt.imshow((np.transpose((output_done.cpu().detach().numpy()[0]), (1, 2, 0))))
        plt.show()
        print('Before: {0} \t After: {1}'.format(before,after))