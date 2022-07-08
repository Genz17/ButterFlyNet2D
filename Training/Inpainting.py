'''
This is for the inpainting task.
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Test')))
import torch
import torchvision
from torch.utils.data import DataLoader
from Mask import squareMask32,squareMask64,squareMask128,squareMask256,squareMask1024,lineMask256,randomMask
from ButterFlyNet_INPAINT import ButterFlyNet_INPAINT
from inpaint_test_func import test_inpainting


#### Here are the settings to the training ###
epochs = 30
batch_size_train = 50
batch_size_test = 256
learning_rate = 0.001
data_path_train = '../../data/celebaselected/' # choose the path where your data is located
data_path_test = '../../data/CelebaTest/' # choose the path where your data is located
image_size = 64 # the image size
local_size = 64 # size the network deals
pile_time = image_size // local_size
net_layer = 6 # should be no more than log_2(local_size)
cheb_num = 4
mask = eval('squareMask'+str(image_size))(torch.zeros(batch_size,1,image_size,image_size)).cuda()

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(data_path_train,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size))])),
    batch_size=batch_size_train, shuffle=True)

test_loader = DataLoader(
    torchvision.datasets.ImageFolder(data_path_test,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size))])),
    batch_size=batch_size_test, shuffle=True)

Net = ButterFlyNet_INPAINT(local_size,net_layer,cheb_num,True).cuda()
optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=50, verbose=True,
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

            pileImage = torch.zeros((batch_size * (pile_time ** 2), 1, local_size, local_size), device='cuda:0')
            pileImageMasked = torch.zeros((batch_size * (pile_time ** 2), 1, local_size, local_size), device='cuda:0')
            image = image.cuda()
            maskedimage = image * mask
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
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        print('Inpaint: local size {}, image size {} Train Epoch: {}/{}, [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(local_size,image_size,epoch+1,epochs,step * len(image),
                                                                        len(train_loader.dataset),
                                                                        100 * step / len(train_loader),
                                                                        loss.item()))

    # Apply testing every epoch
    with torch.no_grad():
        test_inpainting(test_loader,batch_size_test,Net,mask,image_size,local_size)

print('Training is Done.')
torch.save(Net.state_dict(),'{}_{}_Celeba_square_inpainting.pth'.format(local_size,image_size))
