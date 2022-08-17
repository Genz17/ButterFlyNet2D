import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
import torch
from ButterFlyNet2D import ButterFlyNet2D
import torchviz

def SPlot():
    data = torch.ones(1,1,image_size,image_size)
    Net = ButterFlyNet2D(1, image_size, image_size, layer, chebNum, 0, image_size, 0, image_size, initMethod, True)
    model_file =  '../../Pths/Base' + '/{}_{}_{}_{}_Graph'.format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
    out = Net(data)
    graph = torchviz.make_dot(out)
    graph.render(model_file)
    print('Done.')

if __name__ == "__main__":
    image_size  = int(sys.argv[1])
    layer       = int(sys.argv[2])
    chebNum     = int(sys.argv[3])
    initMethod  = sys.argv[4]
    SPlot()
