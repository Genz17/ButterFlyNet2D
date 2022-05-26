import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from ButterFlyNet2D_Flexible import ButterFlyNet2D_Flexible
from ButterFlyNet2D_IDFT import ButterFlyNet2D_IDFT
import matplotlib.pyplot as plt
from randomMask64 import randomMask64

epochs = 10
batch_size_train = 250
learning_rate = 0.001
log_interval = 1

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.STL10('./data/', split='train', download=False,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((64,64))])),
    batch_size=batch_size_train, shuffle=True)



train_losses = []
train_counter = []
for epoch in range(epochs):
    for step, (image, label) in enumerate(train_loader):
        with torch.no_grad():
            image = image.cuda()
            maskedimage = (image * randomMask64(image))
        optimizerencoder.zero_grad()
        optimizerdecoder.zero_grad()
        output = Netencoder(maskedimage)
        output_done = Netdecoder(output).real
        loss = torch.norm(output_done - image) / torch.norm(image)
        loss.backward()
        optimizerencoder.step()
        optimizerdecoder.step()
        scheduler_encode.step(loss)
        scheduler_decode.step(loss)
        with torch.no_grad():
            train_losses.append(loss.item())
            train_counter.append(step * batch_size_train + epoch * len(train_loader.dataset))
            if step % log_interval == 0:
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

torch.save(Netencoder.state_dict(), '55encoderforrandominpaintSTL10.pth')
torch.save(Netdecoder.state_dict(), '55decoderforrandominpaintSTL10.pth')

plt.figure()
plt.plot(train_counter,train_losses)
plt.savefig('INPAINTONSTL10')
plt.show()