import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Test')))
import torch
from ButterFlyNet2D import ButterFlyNet2D
from SeedSetup import setup_seed
import math
import matplotlib.pyplot as plt
import numpy as np

setup_seed(17)

testType        = sys.argv[1]
if testType == 'layer':
    chebNum     = int(sys.argv[2])
if testType == 'cheb':
    layerNum    = int(sys.argv[2])

inputSize       = int(sys.argv[3])
prefix          = eval(sys.argv[4])
data            = torch.rand(1,1,inputSize,inputSize, dtype=torch.complex64) # choose someting you like

if testType == 'layer':
    lossList1 = []
    lossList2 = []
    lossListinf = []
    for layerNum in range(1, int(math.log2(inputSize))+1):
        print(layerNum,int(math.log2(inputSize)))
        Net = ButterFlyNet2D(1, inputSize, inputSize, layerNum, chebNum, 0, inputSize, 0, inputSize, True, False)
        out = Net(data)
        out_ft = torch.fft.fft2(data).cuda()
        err1 = torch.norm(out-out_ft,1)/torch.norm(out_ft,1)
        err2 = torch.norm(out-out_ft,2)/torch.norm(out_ft,2)
        errinf = torch.norm(out-out_ft,np.inf)/torch.norm(out_ft,np.inf)
        print('1-norm err: {}\n2-norm err: {}\ninf-norm err: {}\n'.format(err1.item(),err2.item(),errinf.item()))
        lossList1.append(err1.item())
        lossList2.append(err2.item())
        lossListinf.append(errinf.item())
        del Net

    fig = plt.figure()
    plt.yscale('log')
    plt.ylabel('rel err in log') 
    plt.xlabel('layerNum')
    plt.plot(range(1, int(math.log2(inputSize))+1), lossList1)
    plt.plot(range(1, int(math.log2(inputSize))+1), lossList2)
    plt.plot(range(1, int(math.log2(inputSize))+1), lossListinf)
    plt.legend(['1-norm','2-norm','inf-norm'])
    plt.title('err-layerNum')
    plt.savefig('./layertest_{}.eps'.format(inputSize))
    plt.show()
    plt.show()
    

if testType == 'cheb':
    lossList1 = []
    lossList2 = []
    lossListinf = []
    for chebNum in range(1, 7):
        print(chebNum,6)
        Net = ButterFlyNet2D(1, inputSize, inputSize, layerNum, chebNum, 0, inputSize, 0, inputSize, True, False)
        out = Net(data)
        out_ft = torch.fft.fft2(data).cuda()
        err1 = torch.norm(out-out_ft,1)/torch.norm(out_ft,1)
        err2 = torch.norm(out-out_ft,2)/torch.norm(out_ft,2)
        errinf = torch.norm(out-out_ft,np.inf)/torch.norm(out_ft,np.inf)
        print('1-norm err: {}\n2-norm err: {}\ninf-norm err: {}\n'.format(err1.item(),err2.item(),errinf.item()))
        lossList1.append(err1.item())
        lossList2.append(err2.item())
        lossListinf.append(errinf.item())
        del Net


    fig = plt.figure()
    plt.yscale('log')
    plt.ylabel('rel err in log') 
    plt.xlabel('chebNum')
    plt.plot(range(1, 7), lossList1)
    plt.plot(range(1, 7), lossList2)
    plt.plot(range(1, 7), lossListinf)
    plt.legend(['1-norm','2-norm','inf-norm'])
    plt.title('err-chebNum')
    plt.savefig('./chebtest_{}.eps'.format(inputSize))
    plt.show()