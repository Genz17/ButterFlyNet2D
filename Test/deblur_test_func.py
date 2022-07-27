import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))

def test_deblurring(test_loader,batch_size,Net,image_size):
    for step, (image, label) in enumerate(test_loader):
        with torch.no_grad():
            image = image[0].cuda()
            bluredimage = image[1].cuda()
            output = torch.zeros((batch_size,3,image_size,image_size), device='cuda:0')
        for i in range(3):
            output[:,i:i+1,:,:] = Net(bluredimage[:,i:i+1,:,:])
        
        before = sum([-10 * np.log10((torch.norm(bluredimage[i:i+1,:,:,:] - image[i:i+1,:,:,:], 'fro').item())
                                        ** 2 / (3 * image_size * image_size)) for i in range(batch_size)]) / batch_size
        after = sum([-10 * np.log10((torch.norm(output[i:i+1,:,:,:] - image[i:i+1,:,:,:], 'fro').item())
                                    ** 2 / (3 * image_size * image_size)) for i in range(batch_size)]) / batch_size
                                        
        print('Before PSNR: {}, \t After PSNR: {}'.format(before, after))

        # fig = plt.figure()
        # plt.imshow(torch.permute(image[0].cpu(), (1, 2, 0)))
        # plt.savefig('origin.png')

        # fig = plt.figure()
        # plt.imshow(torch.permute((bluredimage[0].cpu()), (1, 2, 0)))
        # plt.savefig('operated.png')

        # fig = plt.figure()
        # plt.imshow((np.transpose((output.cpu().detach().numpy()[0]), (1, 2, 0))))
        # plt.savefig('test.png')