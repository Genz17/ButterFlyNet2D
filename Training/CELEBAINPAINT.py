import torch
import torchvision
from torch.utils.data import DataLoader
from SquareMask import squareMask64
from ButterFlyNet_INPAINT import ButterFlyNet_INPAINT

epochs = 12
batch_size_train = 100
learning_rate = 0.001
log_interval = 1

train_loader = DataLoader(
    torchvision.datasets.ImageFolder('./data/celebaselected',
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((64,64))])),
    batch_size=batch_size_train, shuffle=True)

Net = ButterFlyNet_INPAINT(64,6,4,True).cuda()
optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=30, verbose=True,
                                                         threshold=0.00005, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-16)

num = 0
for para in Net.parameters():
    num+=torch.prod(torch.tensor(para.shape))
print(num)

print('Training Begins.')
for epoch in range(epochs):
    for step, (image, label) in enumerate(train_loader):

        image = image.cuda()
        maskedimage = (image*squareMask64(image))

        optimizer.zero_grad()
        output_done = Net(maskedimage)
        loss = torch.norm(output_done - image) / torch.norm(image)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        with torch.no_grad():
            print('Gray 64 CELEBAINPAINT: Train Epoch: {}, [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, step * len(image),
                                                                            len(train_loader.dataset),
                                                                            100 * step / len(train_loader),
                                                                            loss.item()))

torch.save(Net.state_dict(),'64NoLiGRAYCelebainpainting.pth')