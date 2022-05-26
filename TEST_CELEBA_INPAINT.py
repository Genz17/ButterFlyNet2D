import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Mask import squareMask64,lineMask256,squareMask128,squareMask256
import numpy as np
from ButterFlyNet_INPAINT import ButterFlyNet_INPAINT
epochs = 1
batch_size = 1
image_size = 256

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(root='F:/CELEBA/',
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size))])),
    batch_size=batch_size, shuffle=False)

local_size = 64
Net = ButterFlyNet_INPAINT(local_size,6,4,False).cuda()

print('Loading parameters...')
Net.load_state_dict(torch.load('64_GRAY_square256_Seperate_Celebainpainting.pth'))
print('Done.')

# General
# for step, (image, label) in enumerate(train_loader):
#     with torch.no_grad():
#         image = image.cuda()
#         maskedimage = image*squareMask64(image)
#     output_done = torch.zeros(batch_size,3,64,64).cuda()
#     for i in range(3):
#         output_done[:,i:i+1,:,:] = Net(maskedimage[:,i:i+1,:,:])
#     before = sum([-10 * np.log10((torch.norm(maskedimage[i] - image[i], 'fro').item())
#                                  ** 2 / (3 * 64 * 64)) for i in range(batch_size)])/batch_size
#     after = sum([-10 * np.log10((torch.norm(output_done[i] - image[i], 'fro').item())
#                                 ** 2 / (3 * 64 * 64)) for i in range(batch_size)])/batch_size



# Sperate

# squareMask
Rmask = squareMask256(torch.zeros(batch_size,1,image_size,image_size)).cuda()
for step, (image, label) in enumerate(train_loader):
    with torch.no_grad():
        lap_time = image_size // local_size
        overlap = torch.zeros((batch_size * (lap_time**2), 3, local_size, local_size)).cuda()
        overlapMasked = torch.zeros((batch_size * (lap_time**2), 3, local_size, local_size)).cuda()
        image = image.cuda()
        maskedimage = image * Rmask

        for ii in range(lap_time**2):
            overlap[ii * batch_size:(ii + 1) * batch_size, :, :, :] = image[:, :,
                                                                                  (ii // lap_time) * local_size:((ii // lap_time) + 1) * local_size,
                                                                                  (ii % lap_time) * local_size:((ii % lap_time) + 1) * local_size]
            overlapMasked[ii * batch_size:(ii + 1) * batch_size, :, :, :] = maskedimage[:, :,
                                                                                        (ii // lap_time) * local_size:((ii // lap_time) + 1) * local_size,
                                                                                        (ii % lap_time) * local_size:((ii % lap_time) + 1) * local_size]
    output_done = torch.zeros(batch_size,3,image_size,image_size).cuda()
    for i in range(3):
        done = Net(overlapMasked[:, i:i + 1, :, :])
        for ii in range(lap_time**2):
            output_done[:,i:i+1,(ii // lap_time) * local_size:((ii // lap_time) + 1) * local_size, (ii % lap_time) * local_size:((ii % lap_time) + 1) * local_size] = done[ii*batch_size:(ii+1)*batch_size,:,:,:]
    before = sum([-10 * np.log10((torch.norm(maskedimage[i] - image[i], 'fro').item())
                                 ** 2 / (3 * image_size * image_size)) for i in range(batch_size)])/batch_size
    after = sum([-10 * np.log10((torch.norm(output_done[i] - image[i], 'fro').item())
                                ** 2 / (3 * image_size * image_size)) for i in range(batch_size)])/batch_size


# Linemask
# Rmask = lineMask256(torch.zeros(batch_size,1,256,256)).cuda()
# for step, (image, label) in enumerate(train_loader):
#     with torch.no_grad():
#         overlap = torch.zeros((batch_size * 64, 3, 32, 32)).cuda()
#         overlapMasked = torch.zeros((batch_size * 64, 3, 32, 32)).cuda()
#         image = image.cuda()
#         maskedimage = image * Rmask
#         for ii in range(64):
#             overlap[ii * batch_size:(ii + 1) * batch_size, :, :, :] = image[:, :,
#                                                                                   (ii // 8) * 32:((ii // 8) + 1) * 32,
#                                                                                   (ii % 8) * 32:((ii % 8) + 1) * 32]
#             overlapMasked[ii * batch_size:(ii + 1) * batch_size, :, :, :] = maskedimage[:, :,
#                                                                                         (ii // 8) * 32:((
#                                                                                                                     ii // 8) + 1) * 32,
#                                                                                         (ii % 8) * 32:((
#                                                                                                                    ii % 8) + 1) * 32]
#     output_done = torch.zeros(batch_size,3,256,256).cuda()
#     for i in range(3):
#         done = Net(overlapMasked[:, i:i + 1, :, :])
#         for ii in range(64):
#             output_done[:,i:i+1,(ii // 8) * 32:((ii // 8) + 1) * 32, (ii % 8) * 32:((ii % 8) + 1) * 32] = done[ii*batch_size:(ii+1)*batch_size,:,:,:]
#     before = sum([-10 * np.log10((torch.norm(maskedimage[i] - image[i], 'fro').item())
#                                  ** 2 / (3 * 256 * 256)) for i in range(batch_size)])/batch_size
#     after = sum([-10 * np.log10((torch.norm(output_done[i] - image[i], 'fro').item())
#                                 ** 2 / (3 * 256 * 256)) for i in range(batch_size)])/batch_size



# =============================================================================
    plt.imshow(torch.permute(image[0].cpu(), (1, 2, 0)))
    plt.show()
    plt.imshow(torch.permute((maskedimage[0].cpu()), (1, 2, 0)))
    plt.show()
    plt.imshow((np.transpose((output_done.cpu().detach().numpy()[0]), (1, 2, 0))))
    plt.show()
# =============================================================================
    print('Before: {0} \t After: {1}'.format(before,after))