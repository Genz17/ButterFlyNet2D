import torch
import random

def squareMask32(inputData):
    # 4 dimension
    mat = torch.ones_like(inputData)
    mat[:,:,12:22,12:22]  =\
        torch.zeros_like(mat[:,:,12:22,12:22])
    return mat

def squareMask64(inputData):
    # 4 dimension
    mat = torch.ones_like(inputData)
    mat[:,:,23:43,23:43]  =\
        torch.zeros_like(mat[:,:,23:43,23:43])
    return mat

def squareMask128(inputData):
    # 4 dimension
    mat = torch.ones_like(inputData)
    mat[:,:,45:85,45:85] = 0
    return mat

def squareMask256(inputData):
    # 4 dimension
    mat = torch.ones_like(inputData)
    mat[:,:,87:167,87:167]  =\
        torch.zeros_like(mat[:,:,87:167,87:167])
    return mat

def squareMask1024(inputData):
    # 4 dimension
    mat = torch.ones_like(inputData)
    mat[:,:,353:673,353:673]  =\
        torch.zeros_like(mat[:,:,353:673,353:673])
    return mat

def lineMask256(inputData):
    mat = torch.ones_like(inputData)
    for col in range(8):
        mat[:,:,(col*32)+14:(col*32)+17,:] = 0
    for row in range(8):
        mat[:,:,:,(row*32)+14:(row*32)+17] = 0
    return mat

def randomMask(inputData, num):
    mat = torch.ones_like(inputData)
    for ii in range(num):
        x = random.randint(0,inputData.shape[2]-1)
        y = random.randint(0,inputData.shape[3]-1)
        mat[:, :, x, y] = 0
    return mat