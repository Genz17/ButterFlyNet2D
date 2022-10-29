import torch

def trainModel(task, train_loader, epoch, epoches, Net, optimizer, scheduler, lossList, local_size, image_size):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    for step, (Totalimage, label) in enumerate(train_loader):
        pileImage       = Totalimage[0].view(-1,1,local_size,local_size).to(device)
        pileImageMasked = Totalimage[1].view(-1,1,local_size,local_size).to(device)

        optimizer.zero_grad()
        output = Net(pileImageMasked)
        loss = torch.norm(output - pileImage) / torch.norm(pileImage)
        if step % 50 == 0:
            lossList.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        print(task + ' local size {}, image size {}. Process: [{}/{} ({:.2f}%)] [{}/{}]\tLoss: {:.6f}'.format(
                                                                        local_size,image_size,int(step*len(pileImage)/((image_size//local_size)**2)),
                                                                        len(train_loader.dataset),
                                                                        100 * step / len(train_loader),
                                                                        epoch+1,
                                                                        epoches,
                                                                        loss.item()))
