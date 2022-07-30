import torch
import random

def squareMask32(inputData, colorChannel=3):
    # 4 dimension
    mat = torch.ones_like(inputData)
    mat[:,12:22,12:22]  = 0
    return mat

def squareMask64(inputData):
    # 4 dimension
    mat = torch.ones_like(inputData)
    mat[:,23:43,23:43]  = 0
    return mat

def squareMask128(inputData):
    # 4 dimension
    mat = torch.ones_like(inputData)
    mat[:,45:85,45:85] = 0
    return mat

def squareMask256(inputData):
    # 4 dimension
    mat = torch.ones_like(inputData)
    mat[:,87:167,87:167]  = 0
    return mat

def lineMask256(inputData):
    mat = torch.ones_like(inputData)
    for col in range(8):
        mat[:,(col*32)+14:(col*32)+17,:] = 0
    for row in range(8):
        mat[:,(row*32)+14:(row*32)+17] = 0
    return mat