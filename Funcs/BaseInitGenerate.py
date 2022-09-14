import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
import torch
from ButterFlyNet_Identical import ButterFlyNet_Identical
from SeedSetup import setup_seed

setup_seed(17)

def initgen():

    print('Generating Net...')
    Net = ButterFlyNet_Identical(image_size, layer, chebNum, initMethod, part)
    print('Done.\n')
    if pretrain:
        print('PreTrain...')
        Net.pretrain(200)
    if part == 'All':
        path = '../../Pths/Base' + '/{}_{}_{}_{}_{}.pth'.format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
    else:
        path = '../../Pths/Part' + '/{}_{}_{}_{}_{}_{}.pth'.format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])

    torch.save(Net.state_dict(),path)

if __name__ == "__main__":
    image_size  = int(sys.argv[1])
    layer       = int(sys.argv[2])
    chebNum     = int(sys.argv[3])
    initMethod  = sys.argv[4]
    pretrain    = eval(sys.argv[5])
    part        = sys.argv[6]
    initgen()
