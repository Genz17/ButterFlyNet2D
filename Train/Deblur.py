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
from ButterFlyNet_Identical import ButterFlyNet_Identical
from deblur_test_func import test_deblurring
from BlurTransform import blurTransfrom
import matplotlib.pyplot as plt

#### Here are the settings to the training ###
print('Train Settings: \nepochs: {}, batchSize: {}; \nimageSize: {}; \nnetLayer: {}, chebNum: {}; \nkernelSize: {}, std: {}.'.format(sys.argv[1],
                                                                                                                        sys.argv[2],
                                                                                                                        sys.argv[3],
                                                                                                                        sys.argv[4],
                                                                                                                        sys.argv[5],
                                                                                                                        sys.argv[6],
                                                                                                                        sys.argv[7]))
epochs              = int(sys.argv[1])
batch_size_train    = int(sys.argv[2])
image_size          = int(sys.argv[3]) # the image size
net_layer           = int(sys.argv[5]) # should be no more than log_2(local_size)
cheb_num            = int(sys.argv[6])
kerNelSize          = int(sys.argv[7])
std                 = float(sys.argv[8])

batch_size_test = 256
learning_rate = 0.002
data_path_train = '../../data/celebaselected/' # choose the path where your data is located
data_path_test = '../../CelebaTest/' # choose the path where your data is located
distill = True

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(data_path_train,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    blurTransfrom(0, std, kerNelSize, 1)])),
    batch_size=batch_size_train, shuffle=True)

test_loader = DataLoader(
    torchvision.datasets.ImageFolder(data_path_test,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    blurTransfrom(0, std, kerNelSize, 3)])),
    batch_size=batch_size_test, shuffle=False)
for step, (x,y) in enumerate(test_loader):

    plt.imshow(torch.permute(x[0][0], (1, 2, 0)))
    plt.show()

    plt.imshow(torch.permute(x[1][0], (1, 2, 0)))
    plt.show()

print('Generating Net...')
Net = ButterFlyNet_Identical(local_size,net_layer,cheb_num,False).cuda()
if distill:
    Net.distill(200)
print('Done.')
optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=100, verbose=True,
                                                         threshold=0.00005, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-16)

num = 0
for para in Net.parameters():
    num+=torch.prod(torch.tensor(para.shape))
print('The number of paras in the network is {}.'.format(num))

##############################################

# Test before training
print('Test before training...')
with torch.no_grad():
    test_deblurring(test_loader, batch_size_test, Net, blurkernel_test, image_size)
print('Done.')

print('Traing Begins.')
for epoch in range(epochs):
    for step, (image, label) in enumerate(train_loader):
        with torch.no_grad():
            image = image[0].cuda()
            bluredimage = image[1].cuda()
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