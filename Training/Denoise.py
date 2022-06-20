import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
import torch
import torchvision
from torch.utils.data import DataLoader
from ButterFlyNet_DENOISE import ButterFlyNet_DENOISE

#### Here are the settings to the training ###
epochs = 15
batch_size = 50
learning_rate = 0.001
data_path = '../../data/celebaselected/'
image_size = 64
local_size = 64
pile_time = image_size // local_size
net_layer = 6 # should be no more than log_2(local_size)
cheb_num = 4
noise = torch.normal(mean=0, std=0.0002, size=(image_size, image_size), device='cuda:0')
train_loader = DataLoader(
    torchvision.datasets.ImageFolder(data_path,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size))])),
    batch_size=batch_size, shuffle=True)

Net = ButterFlyNet_DENOISE(local_size,net_layer,cheb_num,True).cuda()
optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=50, verbose=True,
                                                         threshold=0.00005, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-16)
##############################################

num = 0
for para in Net.parameters():
    num+=torch.prod(torch.tensor(para.shape))
print('The number of paras in the network is {}'.format(num))

print('Training Begins.')
for epoch in range(epochs):
    for step, (image, label) in enumerate(train_loader):
        with torch.no_grad():
            pileImage = torch.zeros((batch_size * (pile_time ** 2), 1, local_size, local_size)).cuda()
            pileImageMasked = torch.zeros((batch_size * (pile_time ** 2), 1, local_size, local_size)).cuda()
            image = image.cuda()
            maskedimage = image + noise
            for ii in range(pile_time ** 2):
                pileImage[ii * batch_size:(ii + 1) * batch_size, :, :, :] = image[:, :,
                                                                          (ii // pile_time) * (local_size):(ii // pile_time) * (local_size) + local_size,
                                                                          (ii % pile_time) * (local_size):(ii % pile_time) * (local_size) + local_size]
                pileImageMasked[ii * batch_size:(ii + 1) * batch_size, :, :, :] = maskedimage[:, :,
                                                                          (ii // pile_time) * (local_size):(ii // pile_time) * (local_size) + local_size,
                                                                          (ii % pile_time) * (local_size):(ii % pile_time) * (local_size) + local_size]
        optimizer.zero_grad()
        output = Net(pileImageMasked)
        loss = torch.norm(output - pileImage) / torch.norm(pileImage)
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step(loss)
        print('Denoise: local size {}, image size {} Train Epoch: {}, [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(local_size,image_size,epoch, step * len(image),
                                                                        len(train_loader.dataset),
                                                                        100 * step / len(train_loader),
                                                                        loss.item()))
print('Training is Done.')
torch.save(Net.state_dict(),'{}_{}_Celeba_denoising'.format(local_size,image_size))