import torch
from Gaussian_Func import gauss

class blurTransform(object):
    def __init__(self, mean, std, kernelSize, colorChannel):
        self.blurkernel = torch.tensor(gauss(mean,mean,std,(kernelSize,kernelSize))).view(1,kernelSize,kernelSize).repeat(colorChannel,1,1)

    def __call__(self, img):

        return (img, torch.fft.ifft2(torch.fft.fft2(img)*torch.fft.fft2(self.blurkernel, (img.shape[1],img.shape[2]))).real)
