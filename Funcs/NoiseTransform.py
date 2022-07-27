import torch

class noiseTransfrom(object):
    def __init__(self, mean, std):
        self.mean   = mean
        self.std    = std

    def __call__(self, img):

        return (img, img + torch.normal(self.mean, self.std, size=(img.shape)))