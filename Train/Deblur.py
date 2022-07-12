'''
This is for the deblurring task.
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Test')))
import torch
import torchvision
from torch.utils.data import DataLoader
from ButterFlyNet_DEBLUR import ButterFlyNet_DEBLUR
from Gaussian_Func import gauss
from deblur_test_func import test_deblurring

#### Here are the settings to the training ###
epochs = 30
batch_size_train = 30
batch_size_test = 256
learning_rate = 0.001
data_path_train = '../../data/celebaselected/' # choose the path where your data is located
data_path_test = '../../data/CelebaTest/' # choose the path where your data is located
image_size = 64 # the image size
net_layer = 6 # should be no more than log_2(local_size)
cheb_num = 4

blurkernel_train = torch.tensor(gauss(0,0,2.5,(5,5)), device='cuda:0').view(1,1,5,5).repeat(batch_size_train,1,1,1) # while training, we only focus on 1 channel
blurkernel_test = torch.tensor(gauss(0,0,2.5,(5,5)), device='cuda:0').view(1,1,5,5).repeat(batch_size_test,3,1,1) # 3 channels

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
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size))])),
    batch_size=batch_size_test, shuffle=False)

print('Generating Net...')
Net = ButterFlyNet_DEBLUR(image_size,net_layer,cheb_num,True).cuda()
print('Done.')
optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.98, patience=100, verbose=True,
                                                         threshold=0.00005, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-16)

num = 0
for para in Net.parameters():
    num+=torch.prod(torch.tensor(para.shape))
print('The number of paras in the network is {}.'.format(num))

##############################################

print('Traing Begins.')
for epoch in range(epochs):
    for step, (image, label) in enumerate(train_loader):
        with torch.no_grad():
            image = image.cuda()
            fourierimage = torch.fft.fft2(image).cuda()
            bluredimage = (torch.fft.ifft2(fourierimage*torch.fft.fft2(blurkernel_train,(image_size, image_size)))).real
        optimizer.zero_grad()
        output = Net(bluredimage)
        loss = torch.norm(output - image) / torch.norm(image)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        print('Deblur: image size {} Train Epoch: {}/{}, [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(image_size,epoch+1,epochs,step * len(image),
                                                                        len(train_loader.dataset),
                                                                        100 * step / len(train_loader),
                                                                        loss.item()))

    with torch.no_grad():
        # Apply testing every epoch
        test_deblurring(test_loader, batch_size, Net, blurkernel_test, image_size)
        print('Saving parameters...')
        torch.save(Net.state_dict(),'../../Pths/{}_Celeba_deblurring.pth'.format(image_size))
        print('Done.')
print('Training is Done.')