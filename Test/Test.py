import sys
import os
import torch
import numpy as np
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))

def test(test_loader,batch_size,Net,image_size,local_size):
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

