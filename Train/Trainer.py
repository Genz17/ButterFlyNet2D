

def trainModel(task, train_loader, Net, optimizer, scheduler, lossList):
    for step, (Totalimage, label) in enumerate(train_loader):
        pileImage       = Totalimage[0].cuda().view(-1,1,local_size,local_size)
        pileImageMasked = Totalimage[1].cuda().view(-1,1,local_size,local_size)

        optimizer.zero_grad()
        output = Net(pileImageMasked)
        loss = torch.norm(output - pileImage) / torch.norm(pileImage)
        if step % 50 == 0:
            lossList.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        print(task + ' local size {}, image size {} Train Epoch: {}/{}, [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                                                                        local_size,image_size,epoch+1,epochs,step * len(pileImage),
                                                                        len(train_loader.dataset),
                                                                        100 * step / len(train_loader),
                                                                        loss.item()))
