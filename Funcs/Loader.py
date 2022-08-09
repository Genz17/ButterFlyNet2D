import torchvision
from torch.utils.data   import DataLoader
from MaskTransform      import maskTransform
from SplitTransform     import splitTransform
from NoiseTransform     import noiseTransform
from BlurTransform      import blurTransform


def load_dataset(task, datasetName, batch_size_train, batch_size_test, image_size, local_size, p1, p2):
    if task == 'Inpaint':
        trainTransform = [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    maskTransform(image_size,'square'),
                                    splitTransform(image_size, local_size, 1)]
        testTransform = [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    maskTransform(image_size,'square')]

    if task == 'Linewatermark':
        trainTransform = [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    maskTransform(image_size,'line'),
                                    splitTransform(image_size, local_size, 1)]
        testTransform = [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    maskTransform(image_size,'line')]

    if task == 'Deblur':
        trainTransform = [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    blurTransform(0, 2.5, 5, 1),
                                    splitTransform(image_size, local_size, 1)]
        testTransform = [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    blurTransform(0, 2.5, 5, 3)]

    if task == 'Denoise':
        trainTransform = [torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    noiseTransform(0, 0.1),
                                    splitTransform(image_size, local_size, 1)]
        testTransform = [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Resize((image_size,image_size)),
                                    noiseTransform(0, 0.1)]

    if datasetName == 'Celeba':
        data_path_train = '../../data/celebaselected/' # choose the path where your data is located
        data_path_test = '../../data/CelebaTest/' # choose the path where your data is located

        train_loader = DataLoader(
            torchvision.datasets.ImageFolder(data_path_train,
                                    transform=torchvision.transforms.Compose(trainTransform)),
            batch_size=batch_size_train, shuffle=True)

        test_loader = DataLoader(
            torchvision.datasets.ImageFolder(data_path_test,
                                    transform=torchvision.transforms.Compose(testTransform)),
            batch_size=batch_size_test, shuffle=False)

    elif datasetName == 'CIFAR10':
        data_path_train = '../../data/'
        data_path_test = '../../data/'
        train_loader = DataLoader(
            torchvision.datasets.CIFAR10(data_path_train,train=True,
                                    transform=torchvision.transforms.Compose(trainTransform)),
            batch_size=batch_size_train, shuffle=True)

        test_loader = DataLoader(
            torchvision.datasets.CIFAR10(data_path_test,train=False,
                                    transform=torchvision.transforms.Compose(testTransform)),
            batch_size=batch_size_test, shuffle=False)

    elif datasetName == 'STL10':
        data_path_train = '../../data/'
        data_path_test = '../../data/'
        train_loader = DataLoader(
            torchvision.datasets.STL10(data_path_train,split='train',
                                    transform=torchvision.transforms.Compose(trainTransform)),
            batch_size=batch_size_train, shuffle=True)

        test_loader = DataLoader(
            torchvision.datasets.STL10(data_path_test,split='test',
                                    transform=torchvision.transforms.Compose(testTransform)),
            batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader
