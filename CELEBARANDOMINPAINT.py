import torch
import torchvision
from torch.utils.data import DataLoader
from Mask import squareMask256
from ButterFlyNet_INPAINT import ButterFlyNet_INPAINT

epochs = 3
batch_size_train = 1
learning_rate = 0.001

train_loader = DataLoader(
    torchvision.datasets.ImageFolder('./data/celebaselected/',
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((256,256))])),
    batch_size=batch_size_train, shuffle=True)

Net = ButterFlyNet_INPAINT(64,6,4,True).cuda()
optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=30, verbose=True,
                                                         threshold=0.00005, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-16)

num = 0
for para in Net.parameters():
    num += torch.prod(torch.tensor(para.shape))
print(num)

Rmask = squareMask256(torch.zeros(batch_size_train,1,256,256)).cuda()

print('Training Begins.')
for epoch in range(epochs):
    for step, (image, label) in enumerate(train_loader):
        overlap = torch.zeros((batch_size_train*16,1,64,64)).cuda()
        overlapMasked = torch.zeros((batch_size_train * 16, 1, 64, 64)).cuda()
        optimizer.zero_grad()
        image = image.cuda()
        maskedimage = image * Rmask

        for ii in range(16):
            overlap[ii*batch_size_train:(ii+1)*batch_size_train,:,:,:] = image[:,:,(ii//4)*64:((ii//4)+1)*64,(ii%4)*64:((ii%4)+1)*64]
            overlapMasked[ii*batch_size_train:(ii+1)*batch_size_train,:,:,:] = maskedimage[:,:,(ii//4)*64:((ii//4)+1)*64,(ii%4)*64:((ii%4)+1)*64]
        output_done = Net(overlapMasked)
        loss = torch.norm(output_done - overlap) / torch.norm(overlap)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        print(
            'CELEBAINPAINT: Train Epoch: {}, [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                                            step * len(
                                                                                                image),
                                                                                            len(train_loader.dataset),
                                                                                            100 * step / len(
                                                                                                train_loader),
                                                                                            loss.item()))


torch.save(Net.state_dict(),'64_GRAY_square256_Seperate_Celebainpainting.pth')
