import torch
import matplotlib.pyplot as plt

class watermaskTransfrom(object):
    def __init__(self, imgShape):
        self.imgShape   = imgShape

    def __call__(self, img):

        waterMark = torch.tensor(plt.imread('./watermarks/watermark'+str(self.imgShape)+'.png'))
        return (img,img+waterMark)
