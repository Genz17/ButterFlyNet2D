import torch
from Gaussian_Func import gauss

class blurTransfrom(object):
    def __init__(self, mean, std, kernelSize, colorChannel):
        self.blurkernel = torch.tensor(gauss(mean,mean,std,(kernelSize,kernelSize))).view(1,1,kernelSize,kernelSize).repeat(batch_size_train,colorChannel,1,1)

    def __call__(self, img):

        return (img, torch.fft.ifft(torch.fft.fft(img)*torch.fft.fft(self.blurkernel, img.shape)).real)