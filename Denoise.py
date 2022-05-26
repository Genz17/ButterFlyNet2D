import torch
from Signal_Mat import signal
from ButterFlyNet2D import ButterFlyNet2D
from ButterFlyNet2D_IDFT import ButterFlyNet2D_IDFT
import matplotlib.pyplot as plt
import numpy as np

learning_rate = 1e-5
steps = 2000
loss_log = [0 for i in range(steps)]
plot_log = [i for i in range(steps)]


######################
## Net preparation ##
######################
Net_encode = ButterFlyNet2D(1, 32, 32, 4, 4, -8, 8, -8, 8, True)
Net_decode = ButterFlyNet2D_IDFT(1, -8, 8, -8, 8, 32, 32, 5, 4, True)
optimizer_encode = torch.optim.Adam(Net_encode.parameters(), lr=learning_rate)
optimizer_decode = torch.optim.Adam(Net_decode.parameters(), lr=learning_rate)
schedualer_encode = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_encode, mode='min', factor=0.5, patience=10, verbose=True,
                                                        threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
schedualer_decode = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_decode, mode='min', factor=0.5, patience=10, verbose=True,
                                                        threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
######################
##      Process     ##
######################
for step in range(steps):
    real_signal = signal(4,4,-5000,5000,-5000,5000,'low',32,32)
    real_signal_tensor = torch.tensor([[real_signal]], dtype=torch.complex64, device='cuda:0')
    noise_signal = real_signal+np.random.normal(0,0.1,real_signal.shape)
    noise_signal_tensor = torch.tensor([[noise_signal]], dtype=torch.complex64, device='cuda:0')
    # print(torch.norm(real_signal_tensor, 'fro'))
    # print(torch.norm(noise_signal_tensor, 'fro'))
    # print(torch.norm(real_signal_tensor-noise_signal_tensor, 'fro'))
    signal_transformed = Net_encode(noise_signal_tensor)
    denoise_signal_tensor = Net_decode(signal_transformed)
    optimizer_encode.zero_grad()
    optimizer_decode.zero_grad()
    loss = torch.norm(denoise_signal_tensor-real_signal_tensor, 'fro')
    loss.backward()
    optimizer_encode.step()
    optimizer_decode.step()
    if step % 10 == 0:
        schedualer_encode.step(loss)
        schedualer_decode.step(loss)
    with torch.no_grad():
        print(loss)
        loss_log[step] = loss.item()
torch.save(Net_encode.state_dict(), '.\ButterFlyNet2D_prefix_denoise_para')
torch.save(Net_decode.state_dict(), '.\ButterFlyNet2D_IDFT_prefix_denoise_para')
denoise_signal = denoise_signal_tensor.cpu().detach().numpy()[0][0]
print(torch.norm(real_signal_tensor-noise_signal_tensor,'fro'))
print(torch.norm(real_signal_tensor-denoise_signal_tensor,'fro'))
plot_x, plot_y = np.meshgrid(range(32),range(32))
ax = plt.axes(projection='3d')
ax.plot_surface(plot_x, plot_y, real_signal, cmap=plt.get_cmap('magma'))
ax.plot_surface(plot_x, plot_y, noise_signal, cmap=plt.get_cmap('rainbow'))
ax.plot_surface(plot_x, plot_y, denoise_signal.real, cmap=plt.get_cmap('cool'))
plt.show()

fig = plt.figure()
plt.plot(plot_log, loss_log)
plt.show()