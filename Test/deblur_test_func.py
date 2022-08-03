import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))

def test_deblurring(test_loader,batch_size,Net,image_size):
    for step, (Totalimage, label) in enumerate(test_loader):
        try:
            with torch.no_grad():
                image = Totalimage[0].cuda()
                bluredimage = Totalimage[1].cuda()
                output = torch.zeros((batch_size,3,image_size,image_size), device='cuda:0')
            for i in range(3):
                output[:,i:i+1,:,:] = Net(bluredimage[:,i:i+1,:,:])
            
            before = sum([-10 * np.log10((torch.norm(bluredimage[i:i+1,:,:,:] - image[i:i+1,:,:,:], 'fro').item())
                                            ** 2 / (3 * image_size * image_size)) for i in range(batch_size)]) / batch_size
            after = sum([-10 * np.log10((torch.norm(output[i:i+1,:,:,:] - image[i:i+1,:,:,:], 'fro').item())
                                        ** 2 / (3 * image_size * image_size)) for i in range(batch_size)]) / batch_size
                                            
            print('Before PSNR: {}, \t After PSNR: {}'.format(before, after))
        except Exception:
            pass
