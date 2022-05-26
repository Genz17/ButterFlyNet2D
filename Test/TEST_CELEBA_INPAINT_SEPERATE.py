import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Mask import squareMask128
import numpy as np
from ButterFlyNet_INPAINT import ButterFlyNet_INPAINT
epochs = 1
batch_size = 1

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(root='F:/CELEBA/',
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((128,128))])),
    batch_size=batch_size, shuffle=False)

Net = ButterFlyNet_INPAINT(64,6,4,False).cuda()

print('Loading parameters...')
Net.load_state_dict(torch.load('64SeperateCelebainpainting.pth'))
print('Done.')


for step, (image, label) in enumerate(train_loader):
    with torch.no_grad():
        image = image.cuda()
        maskedimage = image*squareMask128(image)
    output_done = torch.zeros(batch_size,3,128,128).cuda()
    for i in range(3):
        for ii in range(4):
            output_done[:,i:i+1,(ii//2)*64:((ii//2)+1)*64,(ii%2)*64:((ii%2)+1)*64] = Net(maskedimage[:,i:i+1,(ii//2)*64:((ii//2)+1)*64,(ii%2)*64:((ii%2)+1)*64])
    before = sum([-10 * np.log10((torch.norm(maskedimage[i] - image[i], 'fro').item())
                                 ** 2 / (3 * 128 * 128)) for i in range(batch_size)])/batch_size
    after = sum([-10 * np.log10((torch.norm(output_done[i] - image[i], 'fro').item())
                                ** 2 / (3 * 128 * 128)) for i in range(batch_size)])/batch_size
# =============================================================================
    plt.imshow(torch.permute(image[0].cpu(), (1, 2, 0)))
    plt.show()
    plt.imshow(torch.permute((maskedimage[0].cpu()), (1, 2, 0)))
    plt.show()
    plt.imshow((np.transpose((output_done.cpu().detach().numpy()[0]), (1, 2, 0))))
    plt.show()
# =============================================================================
    print('Before: {0} \t After: {1}'.format(before,after))