import torch
from Mask import *

class maskTransfrom(object):
    def __init__(self, imgShape):
        self.imgShape = imgShape

    def __call__(self, img):

        return (img,img*eval('squareMask'+str(self.imgShape))(img))