import torch

def train(epoch, loss_func ,optimizer, Net,train_loader, batch_size_train,
          train_counter, train_losses, pretrain):
    with torch.no_grad():
        total_loss = 0
    for step, (image, label) in enumerate(train_loader):
        image = image.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        if pretrain:
            output = Net.pretrain(image)
        else:
            output = Net(image)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_loss = total_loss + loss
            print('Train Epoch: {}, [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, step*len(image),
                                                                            len(train_loader.dataset),
                                                                            100*step/len(train_loader),
                                                                            loss.item()))
            train_losses.append(loss.item())
            train_counter.append(step*batch_size_train+epoch*len(train_loader.dataset))
    print('Avg loss: {}'.format(total_loss/len(train_loader.dataset)))