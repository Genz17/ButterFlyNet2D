import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Test')))
import torch
from LossDraw               import LossPlot
from SeedSetup              import setup_seed
from Loader                 import load_dataset
from Netinit                import Netinit
from Trainer                import trainModel
from Test                   import test

setup_seed(17)

#### Here are the settings to the training ###

datasetName         = sys.argv[1]
task                = sys.argv[2]
epoches             = int(sys.argv[3])
batch_size_train    = int(sys.argv[4])
image_size          = int(sys.argv[5]) # the image size
local_size          = int(sys.argv[6]) # size the network deals
prefix              = eval(sys.argv[7])
pretrain            = eval(sys.argv[8])
Resume              = eval(sys.argv[9])

if prefix:
    p1 = 'prefix'
else:
    p1 = 'noprefix'
if pretrain:
    p2 = 'pretrain'
else:
    p2 = 'nopretrain'

print('Task: {};\ndataset: {}.\n'.format(task,datasetName))

batch_size_test = 256
net_layer       = 6
cheb_num        = 2
pile_time = image_size // local_size
lossList = []

pthpath = '../../Pths/' + task + '/' + p1 + '/' + p2 + '/' + datasetName + '_{}_{}_{}_{}.pth'.format(local_size,image_size,net_layer,cheb_num)
imgpath = '../../Images/' + task + '/' + p1 + '/' + p2 + '/' + datasetName + '_{}_{}_{}_{}.eps'.format(local_size,image_size,net_layer,cheb_num)
print('Pth will be saved to: ' + pthpath)
print('\n')
print('Image will be saved to: ' + imgpath)

train_loader, test_loader                   = load_dataset(task, datasetName, batch_size_train, batch_size_test, image_size, local_size, p1, p2)
Net, optimizer, scheduler, startEpoch       = Netinit(local_size, net_layer, cheb_num, Resume, prefix, pretrain, pthpath)

##############################################

print('Test before training...')
# Apply one test before training
with torch.no_grad():
    test(task, test_loader, batch_size_test, Net, image_size, local_size)
print('Done.')

print('Training Begins.')
for epoch in range(startEpoch, epoches):
    print('Now at epoch [{}/{}].'.format(epoch+1,epoches))
    trainModel(task, train_loader, epoch, epoches, Net, optimizer, scheduler, lossList, local_size, image_size)
    # Apply testing every epoch
    with torch.no_grad():
        test(task, test_loader, batch_size_test, Net, image_size, local_size)
        print('Saving parameters and image...')
        checkPoint = {
            'Net':Net.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch+1,
            'scheduler':scheduler.state_dict()
        }
        torch.save(checkPoint,pthpath)
        LossPlot([i*50 for i in range(len(lossList))], lossList, epoch+1, imgpath)
        print('Done.')

print('Training is Done. Now at epoch {}.'.format(epoch+1))