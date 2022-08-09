import torch
from Mask import *

class maskTransfrom(object):
    def __init__(self, imgShape):
        self.imgShape = imgShape

    def __call__(self, img,TP):
        # TP should be 'square' or 'line'        
        return (img,img*eval(TP+'Mask'+str(self.imgShape))(img))
