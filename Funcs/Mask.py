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

def lineMask32(inputData):
    mat = torch.ones_like(inputData)
    for col in range(8):
        mat[:,(col*4)+2:(col*4)+3,:] = 0
    for row in range(8):
        mat[:,:,(row*4)+2:(row*4)+3] = 0
    return mat

def lineMask64(inputData):
    mat = torch.ones_like(inputData)
    for col in range(8):
        mat[:,(col*8)+4:(col*8)+5,:] = 0
    for row in range(8):
        mat[:,:,(row*8)+4:(row*8)+5] = 0
    return mat

def lineMask128(inputData):
    mat = torch.ones_like(inputData)
    for col in range(8):
        mat[:,(col*16)+8:(col*16)+10,:] = 0
    for row in range(8):
        mat[:,:,(row*16)+8:(row*16)+10] = 0
    return mat

def lineMask256(inputData):
    mat = torch.ones_like(inputData)
    for col in range(8):
        mat[:,(col*32)+14:(col*32)+17,:] = 0
    for row in range(8):
        mat[:,:(row*32)+14:(row*32)+17] = 0
    return mat
