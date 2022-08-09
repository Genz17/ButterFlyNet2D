import torch
from Mask import *

class maskTransfrom(object):
    def __init__(self, imgShape,TP):
        self.imgShape   = imgShape
        self.TP         = TP

    def __call__(self, img):

        return (img,img*eval(self.TP+'Mask'+str(self.imgShape))(img))
