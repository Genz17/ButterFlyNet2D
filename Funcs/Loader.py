import torchvision
from MaskTransform  import maskTransfrom
from SplitTransform import splitTransfrom
from NoiseTransform import noiseTransfrom
from BlurTransform  import blurTransfrom


def load_dataset(task, datasetName, image_size, local_size, p1, p2):
    if task == 'Inpaint':
        trainTransfrom = [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    maskTransfrom(image_size),
                                    splitTransfrom(image_size, local_size, 1)]
        testTransfrom = [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    maskTransfrom(image_size)]

    if task == 'Deblur':
        trainTransfrom = [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    blurTransfrom(0, 5, 5, 1),
                                    splitTransfrom(image_size, local_size, 1)]
        testTransfrom = [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    blurTransfrom(0, std, kerNelSize, 3)]

    if task == 'Denoise':
        trainTransfrom = [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    noiseTransfrom(0, 0.1),
                                    splitTransfrom(image_size, local_size, 1)]
        testTransfrom = [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    noiseTransfrom(0, 0.1)]

    if datasetName == 'Celeba':
        data_path_train = '../../data/celebaselected/' # choose the path where your data is located
        data_path_test = '../../data/CelebaTest/' # choose the path where your data is located

        train_loader = DataLoader(
            torchvision.datasets.ImageFolder(data_path_train,
                                    transform=torchvision.transforms.Compose(trainTransfrom)),
            batch_size=batch_size_train, shuffle=True)

        test_loader = DataLoader(
            torchvision.datasets.ImageFolder(data_path_test,
                                    transform=torchvision.transforms.Compose(testTransfrom)),
            batch_size=batch_size_test, shuffle=False)

    
    elif datasetName == 'CIFAR10':
        data_path_train = '../../data/'
        data_path_test = '../../data/'
        train_loader = DataLoader(
            torchvision.datasets.CIFAR10(data_path_train,
                                    transform=torchvision.transforms.Compose(trainTransfrom)),
            batch_size=batch_size_train, shuffle=True)

        test_loader = DataLoader(
            torchvision.datasets.CIFAR10(data_path_test,
                                    transform=torchvision.transforms.Compose(testTransfrom)),
            batch_size=batch_size_test, shuffle=False)

    elif datasetName == 'STL10':
        data_path_train = '../../data/'
        data_path_test = '../../data/'
        train_loader = DataLoader(
            torchvision.datasets.STL10(data_path_train,
                                    transform=torchvision.transforms.Compose(trainTransfrom)),
            batch_size=batch_size_train, shuffle=True)

        test_loader = DataLoader(
            torchvision.datasets.STL10(data_path_test,
                                    transform=torchvision.transforms.Compose(testTransfrom)),
            batch_size=batch_size_test, shuffle=False)
        

    pthpath = '../../Pths/' + task + '/' + p1 + '/' + p2 + '/' + '{}_{}_{}_{}.pth'.format(local_size,image_size,net_layer,cheb_num)
    imgpath = '../../Images/' + task + '/' + p1 + '/' + p2 + '/' + '{}_{}_{}_{}.eps'.format(local_size,image_size,net_layer,cheb_num)
    print('Pth will be saved to: ' + pthpath)
    print('\n')
    print('Image will be saved to: ' + imgpath)

    return train_loader, test_loader, pthpath, imgpath
