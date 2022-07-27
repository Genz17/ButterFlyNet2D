import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Funcs')))
sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','Nets')))

def test_denoising(test_loader,batch_size,Net,noise_mean,noise_std,image_size,local_size):
    for step, (image, label) in enumerate(test_loader):
        with torch.no_grad():
            lap_time = image_size // local_size
            pileNImage = torch.zeros((batch_size * (lap_time ** 2), 3, local_size, local_size)).cuda()
            image = image[0].cuda()
            Nimage = image[1].cuda()

            for ii in range(lap_time ** 2):
                pileNImage[ii * batch_size:(ii + 1) * batch_size, :, :, :] = Nimage[:, :,
                                                                                (ii // lap_time) * local_size:((
                                                                                                                           ii // lap_time) + 1) * local_size,
                                                                                (ii % lap_time) * local_size:((
                                                                                                                          ii % lap_time) + 1) * local_size]
            output_done = torch.zeros(batch_size, 3, image_size, image_size).cuda()
            for i in range(3):
                done = Net(pileNImage[:, i:i + 1, :, :])
                for ii in range(lap_time ** 2):
                    output_done[:, i:i + 1, (ii // lap_time) * local_size:((ii // lap_time) + 1) * local_size,
                    (ii % lap_time) * local_size:((ii % lap_time) + 1) * local_size] = done[ii * batch_size:(ii + 1) * batch_size, :, :, :]
                    
            before = sum([-10 * np.log10(((torch.norm(maskedimage[i] - image[i], 'fro').item())
                                         ** 2) / (3 * image_size * image_size)) for i in range(batch_size)]) / batch_size
            after = sum([-10 * np.log10(((torch.norm(output_done[i] - image[i], 'fro').item())
                                        ** 2) / (3 * image_size * image_size)) for i in range(batch_size)]) / batch_size

        print('Before PSNR: {}, \t After PSNR: {}'.format(before, after))

        fig = plt.figure()
        plt.imshow(torch.permute(image[0].cpu(), (1, 2, 0)))
        plt.savefig('origin.png')

        fig = plt.figure()
        plt.imshow(torch.permute((Nimage[0].cpu()), (1, 2, 0)))
        plt.savefig('operated.png')

        fig = plt.figure()
        plt.imshow((np.transpose((output_done.cpu().detach().numpy()[0]), (1, 2, 0))))
        plt.savefig('recovered.png')