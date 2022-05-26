import torch
import torchvision
from torch.utils.data import DataLoader
from ButterFlyNet2D import ButterFlyNet2D
from ButterFlyNet2D_IDFT import ButterFlyNet2D_IDFT
from SquareMask64 import squareMask64
from IdenticleNet import identicleNet

epochs = 6
batch_size_train = 100
learning_rate = 0.001

train_loader = DataLoader(
    torchvision.datasets.STL10('./data/', split='train', download=False,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((64,64))])),
    batch_size=batch_size_train, shuffle=True)

Netencoder = ButterFlyNet2D(3, 64, 64, 5, 5, 0, 64, 0, 64, True, True).cuda()
Netdecoder = ButterFlyNet2D_IDFT(3, 0, 64, 0, 64, 64, 64, 5, 5, True, True).cuda()
Netlinear = identicleNet(64, 64).cuda()

optimizerencoder = torch.optim.Adam(Netencoder.parameters(), lr=learning_rate)
optimizerdecoder = torch.optim.Adam(Netencoder.parameters(), lr=learning_rate)
optimizerlinear = torch.optim.Adam(Netlinear.parameters(), lr=learning_rate)

scheduler_encode = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerencoder, mode='min', factor=0.5, patience=10, verbose=True,
                                                        threshold=0.00005, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-16)
scheduler_decode = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerdecoder, mode='min', factor=0.5, patience=10, verbose=True,
                                                        threshold=0.00005, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-16)
scheduler_linear = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerlinear, mode='min', factor=0.8, patience=10, verbose=True,
                                                        threshold=0.00005, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-16)


for epoch in range(epochs):
    for step, (image, label) in enumerate(train_loader):
        with torch.no_grad():
            image = image.cuda()
            maskedimage = (image * squareMask64(image))
        optimizerencoder.zero_grad()
        optimizerdecoder.zero_grad()
        optimizerlinear.zero_grad()
        
        output = Netencoder(maskedimage)
        output_scooched = output.view((batch_size_train, 3, 1, -1))
        output_inpaint = Netlinear(output_scooched)
        output_back = output_inpaint.view(output.shape)
        output_done = Netdecoder(output)
        
        loss = torch.norm(output_done - image) / torch.norm(image)
        loss.backward()
        optimizerencoder.step()
        optimizerdecoder.step()
        optimizerlinear.step()
        
        scheduler_encode.step(loss)
        scheduler_decode.step(loss)
        scheduler_linear.step(loss)
        
        print('Inpainting For STL10, Train Epoch: {}, [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, step * len(image),
                                                                        len(train_loader.dataset),
                                                                        100 * step / len(train_loader),
                                                                        loss.item()))

torch.save(Netencoder.state_dict(), '55encoderforinpaintSTL10.pth')
torch.save(Netdecoder.state_dict(), '55decoderforinpaintSTL10.pth')
torch.save(Netlinear.state_dict(), '55linearforinpaintSTL10.pth')
