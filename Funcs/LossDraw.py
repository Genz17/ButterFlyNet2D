import matplotlib.pyplot as plt
import sys
import torch
from pylab import xticks


def LossPlot(Xiter, Yiter, epochNum, path):
    plt.rcParams.update({'font.size':20})
    fig, ax1 = plt.subplots(figsize=(15,6))
    ax1.set_ylabel('loss')
    ax1.set_xlabel('iter')
    plt.yscale('log')
    ax1.plot(Xiter, Yiter)
    ax1.tick_params(axis ='x', labelcolor = 'tab:blue')
    ax2 = ax1.twiny()
    ax2.set_xlabel('epoch', color = 'tab:red')
    xticks(range(0,epochNum+max(epochNum//12,1),max(epochNum//12,1)))
    plt.savefig(path)

if __name__ == '__main__':
    task        = sys.argv[1]
    datasetName = sys.argv[2]
    image_size  = int(sys.argv[3])
    local_size  = int(sys.argv[4])
    net_layer   = int(sys.argv[5])
    cheb_num    = int(sys.argv[6])
    initMethod  = sys.argv[7]
    pretrain    = eval(sys.argv[8])

    p1 = initMethod
    if pretrain:
        p2 = 'pretrain'
    else:
        p2 = 'nopretrain'

    pthpath = '../Pths/' + task + '/' + p1 + '/' + p2 + '/' + datasetName + '_{}_{}_{}_{}.pth'.format(local_size,image_size,net_layer,cheb_num)
    imgpath = '../Images/' + task + '/' + p1 + '/' + p2 + '/' + datasetName + '_{}_{}_{}_{}.eps'.format(local_size,image_size,net_layer,cheb_num)
    checkPoint = torch.load(pthpath)
    lossList = checkPoint['lossList']
    epoch = checkPoint['epoch']
    LossPlot([i*50 for i in range(len(lossList))], lossList, epoch+1, imgpath)
