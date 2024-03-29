import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Test')))
import torch
from ButterFlyNet2D import ButterFlyNet2D
from ButterFlyNet2D_IDFT import ButterFlyNet2D_IDFT
from ButterFlyNet_Identical import ButterFlyNet_Identical
from SeedSetup import setup_seed
import math
import matplotlib.pyplot as plt
import numpy as np

setup_seed(17)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def main(testType,Num,inputSize,initMethod,netType,pretrain):
    dataTest            = torch.rand(1,1,inputSize,inputSize, dtype=torch.complex64, device=device) # choose something you like
    outTest_ft          = torch.fft.fft2(dataTest)
    if testType == 'layer':
        chebNum = Num
        lossList1 = []
        lossList2 = []
        lossListinf = []
        for layerNum in range(1, int(math.log2(inputSize))+1):
            print(layerNum,int(math.log2(inputSize)))
            Net = ButterFlyNet_Identical(inputSize, layerNum, chebNum, initMethod, netType)
            if pretrain:
                path = '../Pths/Part' + '/{}_{}_{}_{}_{}_{}.pth'.format(inputSize,layerNum,chebNum,initMethod,pretrain,netType)
                try:
                    Net.load_state_dict(torch.load(path))
                except BaseException:
                    Net.pretrain(200)
                    torch.save(Net.state_dict(),path)
            if netType == 'f':
                out = Net(dataTest)
                err1 = torch.norm(out-outTest_ft,1)/torch.norm(outTest_ft,1)
                err2 = torch.norm(out-outTest_ft,2)/torch.norm(outTest_ft,2)
                errinf = torch.norm(out-outTest_ft,np.inf)/torch.norm(outTest_ft,np.inf)

            if netType == 'b':
                out = Net(outTest_ft)
                err1 = torch.norm(out-dataTest,1)/torch.norm(dataTest,1)
                err2 = torch.norm(out-dataTest,2)/torch.norm(dataTest,2)
                errinf = torch.norm(out-dataTest,np.inf)/torch.norm(dataTest,np.inf)
            print('1-norm err: {}\n2-norm err: {}\ninf-norm err: {}\n'.format(err1.item(),err2.item(),errinf.item()))
            lossList1.append(err1.item())
            lossList2.append(err2.item())
            lossListinf.append(errinf.item())
            del Net

        fig = plt.figure()
        plt.yscale('log')
        plt.ylabel('rel err in log')
        plt.xlabel('layerNum')
        plt.semilogy(range(1, int(math.log2(inputSize))+1), lossList1)
        plt.semilogy(range(1, int(math.log2(inputSize))+1), lossList2)
        plt.semilogy(range(1, int(math.log2(inputSize))+1), lossListinf)
        plt.legend(['1-norm','2-norm','inf-norm'])
        plt.title('err-layerNum')
        plt.xticks(range(1, int(math.log2(inputSize))+1))
        plt.savefig('./layertest_{}_{}_{}.eps'.format(inputSize, netType, pretrain))
        plt.show()

    if testType == 'cheb':
        layerNum = Num
        lossList1 = []
        lossList2 = []
        lossListinf = []
        for chebNum in range(1, 7):
            print(chebNum)
            Net = ButterFlyNet_Identical(inputSize, layerNum, chebNum, initMethod, netType)
            if pretrain:
                path = '../Pths/Part' + '/{}_{}_{}_{}_{}_{}.pth'.format(inputSize,layerNum,chebNum,initMethod,pretrain,netType)
                try:
                    Net.load_state_dict(torch.load(path))
                except BaseException:
                    Net.pretrain(200)
                    torch.save(Net.state_dict(),path)
            if netType == 'f':
                out = Net(dataTest)
                err1 = torch.norm(out-outTest_ft,1)/torch.norm(outTest_ft,1)
                err2 = torch.norm(out-outTest_ft,2)/torch.norm(outTest_ft,2)
                errinf = torch.norm(out-outTest_ft,np.inf)/torch.norm(outTest_ft,np.inf)

            if netType == 'b':
                out = Net(outTest_ft)
                err1 = torch.norm(out-dataTest,1)/torch.norm(dataTest,1)
                err2 = torch.norm(out-dataTest,2)/torch.norm(dataTest,2)
                errinf = torch.norm(out-dataTest,np.inf)/torch.norm(dataTest,np.inf)
            print('1-norm err: {}\n2-norm err: {}\ninf-norm err: {}\n'.format(err1.item(),err2.item(),errinf.item()))
            lossList1.append(err1.item())
            lossList2.append(err2.item())
            lossListinf.append(errinf.item())
            del Net

        fig = plt.figure()
        plt.yscale('log')
        plt.ylabel('rel err in log')
        plt.xlabel('chebNum')
        plt.semilogy(range(1, 7), lossList1)
        plt.semilogy(range(1, 7), lossList2)
        plt.semilogy(range(1, 7), lossListinf)
        plt.legend(['1-norm','2-norm','inf-norm'])
        plt.title('err-chebNum')
        plt.xticks(range(1, 7))
        plt.savefig('./chebtest_{}_{}_{}.eps'.format(inputSize, netType, pretrain))
        plt.show()

if __name__ == '__main__':
    testType        = sys.argv[1]
    if testType == 'layer':
        Num     = int(sys.argv[2])
    if testType == 'cheb':
        Num    = int(sys.argv[2])
    inputSize       = int(sys.argv[3])
    initMethod      = sys.argv[4]
    netType         = sys.argv[5]
    pretrain        = eval(sys.argv[6])
    main(testType,Num,inputSize,initMethod,netType,pretrain)
