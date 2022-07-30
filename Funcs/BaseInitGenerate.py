import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
import torch
import torchvision
from torch.utils.data import DataLoader
from ButterFlyNet_Identical import ButterFlyNet_Identical
from SeedSetup import setup_seed

setup_seed(17)

def initgen():

    print('Generating Net...')
    Net = ButterFlyNet_Identical(image_size, layer, chebNum, prefix)
    print('Done.\n')
    if pretrain:
        print('PreTrain...')
        Net.pretrain(200)
    path = '../../Pths/Base' + '/{}_{}.pth'.format(sys.argv[4],sys.argv[5])
    torch.save(Net.state_dict(),path)

if __name__ == "__main__":
    image_size  = int(sys.argv[1])
    layer       = int(sys.argv[2])
    chebNum     = int(sys.argv[3])
    prefix      = eval(sys.argv[4])
    pretrain    = eval(sys.argv[5])
    initgen()