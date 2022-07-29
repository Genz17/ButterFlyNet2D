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
from ButterFlyNet_Identical import ButterFlyNet_Identical
from inpaint_test_func import test_inpainting
from MaskTransform import maskTransfrom
from LossDraw import LossPlot
from SeedSetup import setup_seed

setup_seed(17)

#### Here are the settings to the training ###
print('\nTrain Settings: \nepochs: {}, batchSize: {}; \
\nimageSize: {}, localSize: {}; \nnetLayer: {}, chebNum: {};\nprefix: {}, pretrain: {}.\n'.format(sys.argv[1],
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
prefix              = eval(sys.argv[7])
pretrain            = eval(sys.argv[8])
datasetName         = sys.argv[9]

if prefix:
    p1 = 'prefix'
else:
    p1 = 'noprefix'
if pretrain:
    p2 = 'pretrain'
else:
    p2 = 'nopretrain'

batch_size_test = 256
learning_rate = 0.002
pile_time = image_size // local_size
lossList = []

if datasetName == 'Celeba':
    data_path_train = '../../data/celebaselected/' # choose the path where your data is located
    data_path_test = '../../data/CelebaTest/' # choose the path where your data is located

    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(data_path_train,
                                transform=torchvision.transforms.Compose(
                                    [torchvision.transforms.Grayscale(num_output_channels=1),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Resize((image_size,image_size)),
                                        maskTransfrom(image_size)])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(data_path_test,
                                transform=torchvision.transforms.Compose(
                                    [torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Resize((image_size,image_size)),
                                        maskTransfrom(image_size)])),
        batch_size=batch_size_test, shuffle=False)

    pthpath = '../../Pths/Inpaint/' + sys.argv[7] + '/' + sys.argv[8] + '/' + '{}_{}_{}_{}_Celeba_square_inpainting.pth'.format(local_size,image_size,net_layer,cheb_num)
    imgpath = '../../Images/Inpaint/' + sys.argv[7] + '/' + sys.argv[8] + '/' + '{}_{}_{}_{}_Celeba_square_inpainting.eps'.format(local_size,image_size,net_layer,cheb_num)

elif datasetName == 'CIFAR10':
    data_path_train = '../../data/'
    data_path_test = '../../data/'
    train_loader = DataLoader(
        torchvision.datasets.CIFAR10(data_path_train,
                                transform=torchvision.transforms.Compose(
                                    [torchvision.transforms.Grayscale(num_output_channels=1),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Resize((image_size,image_size)),
                                        maskTransfrom(image_size)])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = DataLoader(
        torchvision.datasets.CIFAR10(data_path_test,
                                transform=torchvision.transforms.Compose(
                                    [torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Resize((image_size,image_size)),
                                        maskTransfrom(image_size)])),
        batch_size=batch_size_test, shuffle=False)

    pthpath = '../../Pths/Inpaint/' + p1 + '/' + p2 + '/' + '{}_{}_{}_{}_CIFAR_square_inpainting.pth'.format(local_size,image_size,net_layer,cheb_num)
    imgpath = '../../Images/Inpaint/' + p1 + '/' + p2 + '/' + '{}_{}_{}_{}_CIFAR_square_inpainting.eps'.format(local_size,image_size,net_layer,cheb_num)

elif datasetName == 'STL10':
    data_path_train = '../../data/'
    data_path_test = '../../data/'
    train_loader = DataLoader(
        torchvision.datasets.STL10(data_path_train,
                                transform=torchvision.transforms.Compose(
                                    [torchvision.transforms.Grayscale(num_output_channels=1),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Resize((image_size,image_size)),
                                        maskTransfrom(image_size)])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = DataLoader(
        torchvision.datasets.STL10(data_path_test,
                                transform=torchvision.transforms.Compose(
                                    [torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Resize((image_size,image_size)),
                                        maskTransfrom(image_size)])),
        batch_size=batch_size_test, shuffle=False)
    
    pthpath = '../../Pths/Inpaint/' + sys.argv[7] + '/' + sys.argv[8] + '/' + '{}_{}_{}_{}_STL_square_inpainting.pth'.format(local_size,image_size,net_layer,cheb_num)
    imgpath = '../../Images/Inpaint/' + sys.argv[7] + '/' + sys.argv[8] + '/' + '{}_{}_{}_{}_STL_square_inpainting.eps'.format(local_size,image_size,net_layer,cheb_num)

print('Pth will be saved to: ' + pthpath)
print('\n')
print('Image will be saved to: ' + imgpath)


print('\nGenerating Net...')
Net = ButterFlyNet_Identical(local_size,net_layer,cheb_num,prefix).cuda()
if pretrain:
    Net.pretrain(200)
print('Done.')

num = 0
for para in Net.parameters():
    num+=torch.prod(torch.tensor(para.shape))
print('The number of paras in the network is {}.'.format(num))

optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=100, verbose=True,
                                                         threshold=0.00005, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-16)
##############################################

print('Test before training...')
# Apply one test before training
with torch.no_grad():
    test_inpainting(test_loader,batch_size_test,Net,image_size,local_size)
print('Done.')

print('Training Begins.')
for epoch in range(epochs):
    for step, (Totalimage, label) in enumerate(train_loader):
        with torch.no_grad():

            pileImage = torch.zeros((batch_size_train * (pile_time ** 2), 1, local_size, local_size)).cuda()
            pileImageMasked = torch.zeros((batch_size_train * (pile_time ** 2), 1, local_size, local_size)).cuda()
            image = Totalimage[0].cuda()
            maskedimage = Totalimage[1].cuda()
            for ii in range(pile_time ** 2):
                pileImage[ii * batch_size_train:(ii + 1) * batch_size_train, :, :, :] = image[:, :,
                                                                          (ii // pile_time) * (local_size):(ii // pile_time) * (local_size) + local_size,
                                                                          (ii % pile_time) * (local_size):(ii % pile_time) * (local_size) + local_size]
                pileImageMasked[ii * batch_size_train:(ii + 1) * batch_size_train, :, :, :] = maskedimage[:, :,
                                                                          (ii // pile_time) * (local_size):(ii // pile_time) * (local_size) + local_size,
                                                                          (ii % pile_time) * (local_size):(ii % pile_time) * (local_size) + local_size]
        optimizer.zero_grad()
        output = Net(pileImageMasked)
        loss = torch.norm(output - pileImage) / torch.norm(pileImage)
        if step % 50 == 0:
            lossList.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        print('prefix: ' + sys.argv[7] + ' pretrain: ' + sys.argv[8] + '. Inpaint: local size {}, image size {} Train Epoch: {}/{}, [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(local_size,image_size,epoch+1,epochs,step * len(image),
                                                                        len(train_loader.dataset),
                                                                        100 * step / len(train_loader),
                                                                        loss.item()))

    # Apply testing every epoch
    with torch.no_grad():
        test_inpainting(test_loader,batch_size_test,Net,image_size,local_size)
        print('Saving parameters and image...')
        torch.save(Net.state_dict(),pthpath)
        LossPlot([i*50 for i in range(len(lossList))], lossList, epoch+1, imgpath)
        print('Done.')

print('Training is Done.')