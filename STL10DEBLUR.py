import torch
import torchvision
from torch.utils.data import DataLoader
from ButterFlyNet2D_Flexible import ButterFlyNet2D_Flexible
import matplotlib.pyplot as plt
from Gaussian_Func import gauss
from ButterFlyNet2D_IDFT import ButterFlyNet2D_IDFT
from DeblurMatNet import deblurNet
import numpy as np

epochs = 5
batch_size_train = 250
learning_rate = 0.0008
log_interval = 1
momentum = 0.5

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.STL10('./data/', split='train', download=False,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((64,64))])),
    batch_size=batch_size_train, shuffle=True)

blurkernel = torch.zeros((batch_size_train,3,5,5)).cuda()
for channel in range(3):
    blurkernel[:,channel,:,:] = torch.tensor(gauss(0,0,2.5,(5,5)))
Netencoder = ButterFlyNet2D_Flexible(3, 64, 64, 5, 3, 0, 64, 0, 64, False, [(4,4),(2,2),(2,2),(2,2),(2,2)], ['Max', 'Max', 'Max','Max','Max'], False, True).cuda()
Netdecoder = ButterFlyNet2D_IDFT(3, 0, 64, 0, 64, 64, 64, 5, 3, False, False).cuda()
Netlineardeblur = deblurNet(2.5, (5,5), 64, 64, (64, 64)).cuda()
num = 0
for para in Netencoder.parameters():
    num+=torch.prod(torch.tensor(para.shape))
print(num)
num = 0
for para in Netdecoder.parameters():
    num += torch.prod(torch.tensor(para.shape))
print(num)
num = 0
for para in Netlineardeblur.parameters():
    num+=torch.prod(torch.tensor(para.shape))
print(num)
optimizerencoder = torch.optim.Adam(Netencoder.parameters(), lr=learning_rate)
optimizerdecoder = torch.optim.Adam(Netencoder.parameters(), lr=learning_rate)
optimizerlinear = torch.optim.Adam(Netencoder.parameters(), lr=learning_rate)
scheduler_encode = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerencoder, mode='min', factor=0.8, patience=10, verbose=True,
                                                        threshold=0.00005, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-16)
scheduler_deblur_linear = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerlinear, mode='min', factor=0.8, patience=10, verbose=True,
                                                        threshold=0.00005, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-16)
scheduler_decode = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerdecoder, mode='min', factor=0.8, patience=10, verbose=True,
                                                        threshold=0.00005, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-16)
train_losses = []
train_counter = []
for epoch in range(epochs):
    for step, (image, label) in enumerate(train_loader):
        with torch.no_grad():
            image = image.cuda()
            fourierimage = torch.fft.fft2(image).cuda()
            maskedimage = (torch.fft.ifft2(fourierimage*torch.fft.fft2(blurkernel,(64,64)))).real
        optimizerencoder.zero_grad()
        optimizerdecoder.zero_grad()
        optimizerlinear.zero_grad()
        output = Netencoder(maskedimage)
        # output = torch.fft.fft2(maskedimage)
        output_scooched = output.view((batch_size_train,3,1,-1))
        output_deblurred = Netlineardeblur(output_scooched)
        output_back = output_deblurred.view(output.shape)
        # output_done = torch.fft.ifft2(output_back)
        output_done = Netdecoder(output_back)
        loss = torch.norm(output_done - image) / torch.norm(image)
        loss.backward()
        optimizerencoder.step()
        optimizerdecoder.step()
        optimizerlinear.step()
        if step % 5 == 0:
            scheduler_encode.step(loss)
            scheduler_deblur_linear.step(loss)
            scheduler_decode.step(loss)
        with torch.no_grad():
            train_losses.append(loss.item())
            train_counter.append(step * batch_size_train + epoch * len(train_loader.dataset))
            # plt.imshow(np.transpose(image[0].cpu().detach().numpy(), (1, 2, 0)))
            # plt.show()
            # plt.imshow(np.transpose((maskedimage.cpu().detach().numpy()[0]).real, (1, 2, 0)))
            # plt.show()
            # plt.imshow((np.transpose((output_done.cpu().detach().numpy()[0]).real, (1, 2, 0))))
            # plt.show()
            print('Train Epoch: {}, [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, step * len(image),
                                                                            len(train_loader.dataset),
                                                                            100 * step / len(train_loader),
                                                                            loss.item()))

torch.save(Netencoder.state_dict(), '55encoderfordeblurSTL10.pth')
torch.save(Netdecoder.state_dict(), '55decoderfordeblurSTL10.pth')
torch.save(Netlineardeblur.state_dict(), '55linearfordeblurSTL10.pth')

plt.figure()
plt.plot(train_counter,train_losses)
plt.savefig('DEBLURONSTL10')
plt.show()
