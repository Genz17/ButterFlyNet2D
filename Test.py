import torch

def test(loss_func, Net, test_loader,test_acc):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for step, (image, label) in enumerate(test_loader):
            image = image.cuda()
            label = label.cuda()
            output = Net(image)
            test_loss = test_loss + loss_func(output, label).item()
            pred = torch.max(output.detach(), 1)[1]
            correct = correct + pred.eq(label.detach().view_as(pred)).sum()
        test_loss = test_loss/len(test_loader.dataset)
        test_acc.append((correct/len(test_loader.dataset)).item())
        print('\nTest set: Avg. loss:\t{:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(test_loss,
                                                                                   correct,
                                                                                   len(test_loader.dataset),
                                                                                   100*correct/len(test_loader.dataset)))