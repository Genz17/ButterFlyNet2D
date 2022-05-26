import numpy as np
import torch
import matplotlib.pyplot as plt
from ButterFlyNet2D import ButterFlyNet2D

MAT = torch.randn((1,1,4,4), dtype=torch.complex64).cuda()
MAT = MAT + 1j*MAT
MAT_F = torch.fft.fft2(MAT)
Net = ButterFlyNet2D(1,4,4,1,6,0,2,2,4,True,False).cuda()
num = 0
for para in Net.parameters():
    num+=torch.prod(torch.tensor(para.shape))
print(num)
res = Net(MAT)
print(MAT_F)
print(res)
# print(torch.norm(res-MAT_F)/torch.norm(MAT_F))
# plot_x,plot_y=np.meshgrid(range(0,8), range(0,8))
# ax = plt.axes(projection='3d')
# ax.scatter(plot_x, plot_y, (res.cpu())[0][0].real.detach().numpy()-MAT_F[0][0].cpu().real.detach().numpy())
# ax.scatter(plot_x, plot_y, (res.cpu())[0][0].imag.detach().numpy()-MAT_F[0][0].cpu().imag.detach().numpy())
# plt.show()