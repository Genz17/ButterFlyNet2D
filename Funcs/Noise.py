import torch
def noise(height,width,mean,std):

    return torch.normal(mean,std,size=(height,width))

