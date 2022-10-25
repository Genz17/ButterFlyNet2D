import sys
import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))
from Loader import load_dataset
from ButterFlyNet_Identical import ButterFlyNet_Identical

def test(test_loader,batch_size,Net,image_size,local_size,pic=False):
    for step, (Totalimage, label) in enumerate(test_loader):
        try:
            with torch.no_grad():
                lap_time = image_size // local_size
                overlapMasked = torch.zeros((batch_size * (lap_time ** 2), 3, local_size, local_size)).cuda()
                image = Totalimage[0].cuda()
                operatedimage = Totalimage[1].cuda()

                for ii in range(lap_time ** 2):
                    overlapMasked[ii * batch_size:(ii + 1) * batch_size, :, :, :] = operatedimage[:, :,
                                                                                    (ii // lap_time) * local_size:((
                                                                                    ii // lap_time) + 1) * local_size,
                                                                                    (ii % lap_time) * local_size:((
                                                                                    ii % lap_time) + 1) * local_size]
                output_done = torch.zeros(batch_size, 3, image_size, image_size).cuda()
                for i in range(3):
                    done = Net(overlapMasked[:, i:i + 1, :, :])
                    for ii in range(lap_time ** 2):
                        output_done[:, i:i + 1,
                                    (ii // lap_time) * local_size:((ii // lap_time) + 1) * local_size,
                                    (ii % lap_time) * local_size:((ii % lap_time) + 1) * local_size] = \
                                    done[ii * batch_size:(ii + 1) * batch_size, :, :, :]

                before = sum([-10 * np.log10((torch.norm(operatedimage[i:i+1,:,:,:] - image[i:i+1,:,:,:], 'fro').item())
                                             ** 2 / (3 * image_size * image_size)) for i in range(batch_size)]) / batch_size
                after = sum([-10 * np.log10((torch.norm(output_done[i:i+1,:,:,:] - image[i:i+1,:,:,:], 'fro').item())
                                            ** 2 / (3 * image_size * image_size)) for i in range(batch_size)]) / batch_size
            print('Before PSNR: {}, \t After PSNR: {}'.format(before, after))


        except Exception:
            pass

        if pic:
            fig = plt.figure()
            plt.axis('off')
            plt.imshow(torch.permute(image[0].cpu(),(1,2,0)))
            plt.savefig(imgpath_origin,bbox_inches='tight',pad_inches=0)

            fig = plt.figure()
            plt.axis('off')
            plt.imshow(torch.permute(operatedimage[0].cpu(),(1,2,0)))
            plt.savefig(imgpath_operated,bbox_inches='tight',pad_inches=0)

            fig = plt.figure()
            plt.axis('off')
            plt.imshow(torch.permute(output_done[0].cpu(),(1,2,0)))
            plt.savefig(imgpath_recover,bbox_inches='tight',pad_inches=0)
            break

if __name__ == '__main__':
    task        = sys.argv[1]
    datasetName = sys.argv[2]
    image_size  = int(sys.argv[3])
    local_size  = int(sys.argv[4])
    net_layer   = int(sys.argv[5])
    cheb_num    = int(sys.argv[6])
    initMethod  = sys.argv[7]
    pretrain    = eval(sys.argv[8])
    pic         = eval(sys.argv[9])

    p1 = initMethod
    if pretrain:
        p2 = 'pretrain'
    else:
        p2 = 'nopretrain'

    batch_size_test = 256
    pthpath = '../Pths/' + task + '/' + p1 + '/' + p2 + '/' + datasetName + '_{}_{}_{}_{}.pth'.format(local_size,image_size,net_layer,cheb_num)
    imgpath_origin = '../Images/pics/' + task + '/' + p1 + '/' + p2 + '/' + datasetName + '_{}_{}_{}_{}_or.eps'.format(local_size,image_size,net_layer,cheb_num)
    imgpath_operated = '../Images/pics/' + task + '/' + p1 + '/' + p2 + '/' + datasetName + '_{}_{}_{}_{}_op.eps'.format(local_size,image_size,net_layer,cheb_num)
    imgpath_recover = '../Images/pics/' + task + '/' + p1 + '/' + p2 + '/' + datasetName + '_{}_{}_{}_{}_re.eps'.format(local_size,image_size,net_layer,cheb_num)

    checkPoint = torch.load(pthpath)
    Net = ButterFlyNet_Identical(local_size,net_layer,cheb_num).cuda()
    Net.load_state_dict(checkPoint['Net'])
    train_loader,test_loader = load_dataset(task, datasetName, 20, batch_size_test, image_size, local_size, p1, p2)
    test(test_loader,batch_size_test,Net,image_size,local_size,pic)



