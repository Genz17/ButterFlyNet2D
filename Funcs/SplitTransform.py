import torch

class splitTransform(object):
    def __init__(self, imgSize, localSize, colorChannel):
        self.imgSize        = imgSize
        self.localSize      = localSize
        self.colorChannel   = colorChannel

    def __call__(self, Totalimage):
        pile_time = self.imgSize//self.localSize
        local_size = self.localSize

        pileImage = torch.zeros((self.colorChannel * (pile_time ** 2), local_size, local_size))
        pileImageMasked = torch.zeros((self.colorChannel * (pile_time ** 2), local_size, local_size))
        image = Totalimage[0]
        maskedimage = Totalimage[1]
        for ii in range(pile_time ** 2):
            pileImage[ii * self.colorChannel:(ii + 1) * self.colorChannel, :, :] = image[:,
                                                        (ii // pile_time) * (local_size):(ii // pile_time) * (local_size) + local_size,
                                                        (ii % pile_time) * (local_size):(ii % pile_time) * (local_size) + local_size]
            pileImageMasked[ii * self.colorChannel:(ii + 1) * self.colorChannel, :, :] = maskedimage[:,
                                                        (ii // pile_time) * (local_size):(ii // pile_time) * (local_size) + local_size,
                                                        (ii % pile_time) * (local_size):(ii % pile_time) * (local_size) + local_size]
        return (pileImage,pileImageMasked)
