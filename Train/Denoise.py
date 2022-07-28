import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
import torch
import torchvision
from torch.utils.data import DataLoader
from ButterFlyNet_Identical import ButterFlyNet_Identical
from NoiseTransform import noiseTransfrom

#### Here are the settings to the training ###
print('Train Settings: \nepochs: {}, batchSize: {}; \nimageSize: {}, localSize: {}; \nnetLayer: {}, chebNum: {}; \nmean: {}, std: {}.'.format(sys.argv[1],
                                                                                                                        sys.argv[2],
                                                                                                                        sys.argv[3],
                                                                                                                        sys.argv[4],
                                                                                                                        sys.argv[5],
                                                                                                                        sys.argv[6],
                                                                                                                        sys.argv[7],
                                                                                                                        sys.argv[8]))
epochs              = int(sys.argv[1])
batch_size_train    = int(sys.argv[2])
image_size          = int(sys.argv[3]) # the image size
local_size          = int(sys.argv[4]) # size the network deals
net_layer           = int(sys.argv[5]) # should be no more than log_2(local_size)
cheb_num            = int(sys.argv[6])
noise_mean          = int(sys.argv[7])
noise_std           = float(sys.argv[8])

batch_size_test = 256
learning_rate = 0.002
data_path_train = '../../data/celebaselected/' # choose the path where your data is located
data_path_test = '../../data/CelebaTest/' # choose the path where your data is located
pile_time = image_size // local_size

distill = True

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(data_path_train,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    noiseTransfrom(noise_mean, noise_std)])),
    batch_size=batch_size_train, shuffle=True)

test_loader = DataLoader(
    torchvision.datasets.ImageFolder(data_path_test,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    noiseTransfrom(noise_mean, noise_std)])),
    batch_size=batch_size_test, shuffle=False)

print('Generating Net...')
Net = ButterFlyNet_Identical(local_size,net_layer,cheb_num,True).cuda()
if distill:
    Net.distill(200)
print('Done.')
optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=100, verbose=True,
                                                         threshold=0.00005, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-16)
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
            pileNImage = torch.zeros((batch_size * (pile_time ** 2), 1, local_size, local_size)).cuda()
            image = image[0].cuda()
            Nimage = image[1].cuda()
            for ii in range(pile_time ** 2):
                pileImage[ii * batch_size:(ii + 1) * batch_size, :, :, :] = image[:, :,
                                                                          (ii // pile_time) * (local_size):(ii // pile_time) * (local_size) + local_size,
                                                                          (ii % pile_time) * (local_size):(ii % pile_time) * (local_size) + local_size]
                pileNImage[ii * batch_size:(ii + 1) * batch_size, :, :, :] = Nimage[:, :,
                                                                          (ii // pile_time) * (local_size):(ii // pile_time) * (local_size) + local_size,
                                                                          (ii % pile_time) * (local_size):(ii % pile_time) * (local_size) + local_size]
        optimizer.zero_grad()
        output = Net(pileNImage)
        loss = torch.norm(output - pileImage) / torch.norm(pileImage)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        print('Denoise: image size {} Train Epoch: {}, [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(image_size,epoch, step * len(image),
                                                                        len(train_loader.dataset),
                                                                        100 * step / len(train_loader),
                                                                        loss.item()))
print('Training is Done.')
torch.save(Net.state_dict(),'{}_{}_{}_{}_Celeba_denoising'.format(image_size,local_size,noise_mean,noise_std))
