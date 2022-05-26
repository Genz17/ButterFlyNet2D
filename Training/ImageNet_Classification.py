import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from ButterFlyNet_Classification import ButterFlyNet_Classification
from Test import test
from Train import train

pretrainepoch = 3
epochs = 50
batch_size_train = 200
batch_size_test = 1000
learning_rate = 0.001

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
                                    torchvision.transforms.RandomRotation(10),
                                    torchvision.transforms.RandomHorizontalFlip(p=0.1),
                                    torchvision.transforms.Resize((32,32)),
                                    torchvision.transforms.ToTensor()])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((32,32))])),
    batch_size=batch_size_test, shuffle=True)

Net = ButterFlyNet_Classification(32,5,1,10).cuda()
preoptimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
schedualer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=4, verbose=True,
                                                        threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
num=0
for para in Net.parameters():
    num+=torch.prod(torch.tensor(para.shape))
print(num)

loss_func = nn.CrossEntropyLoss().cuda()
train_losses = []
train_counter = []
train_test_acc = []
train_test_counter = []
test_acc = []
test_counter = []


for epoch in range(pretrainepoch):
    train(epoch,loss_func,preoptimizer,Net, train_loader, batch_size_train,
                train_counter,train_losses, True)



for epoch in range(epochs):
    train(epoch,loss_func,optimizer,Net, train_loader, batch_size_train,
                train_counter,train_losses, False)
    with torch.no_grad():
        test_counter.append(epoch)
        test(loss_func, Net, test_loader,test_acc)
        train_test_counter.append(epoch)
        schedualer.step(test_acc[-1])

fig_loss = plt.figure()
plt.scatter(train_counter, train_losses)
plt.savefig('Loss')
plt.show()

fig_train_acc = plt.figure()
plt.scatter(train_test_counter, train_test_acc)
plt.savefig('Trainacc')
plt.show()

fig_test_acc = plt.figure()
plt.scatter(test_counter, test_acc)
plt.savefig('Testacc')
plt.show()
