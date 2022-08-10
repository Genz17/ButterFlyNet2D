import torch
from ButterFlyNet_Identical import ButterFlyNet_Identical

def Netinit(local_size,net_layer,cheb_num,Resume,initMethod,pretrain,pthpath):
    if Resume:
        print('Resume. Loading...')
        Net = ButterFlyNet_Identical(local_size,net_layer,cheb_num).cuda()
        optimizer = torch.optim.Adam(Net.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.98, patience=100, verbose=True,
                                                                threshold=0.00005, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-16)

        checkPoint = torch.load(pthpath)
        lossList = checkPoint['lossList']
        Net.load_state_dict(checkPoint['Net'])
        optimizer.load_state_dict(checkPoint['optimizer'])
        scheduler.load_state_dict(checkPoint['scheduler'])
        startEpoch = checkPoint['epoch']
        print('Done.\n')

        print('Starts from epoch {}.'.format(startEpoch))
        for group in optimizer.param_groups:
            print('Learning Rate: {}.'.format(group['lr']))

    else:
        print('\nGenerating Net...')
        Net = ButterFlyNet_Identical(local_size,net_layer,cheb_num).cuda()
        try:
            path = '../../Pths/Base' + '/{}_{}_{}_{}_{}.pth'.format(local_size,net_layer,cheb_num,initMethod,pretrain)
            Net.load_state_dict(torch.load(path))
            print('Paras have been created. Loaded.')
        except Exception:
            print('Need to initialize from the bottom.')
            print('\nGenerating Net...')
            Net = ButterFlyNet_Identical(local_size,net_layer,cheb_num,initMethod).cuda()
            if pretrain:
                Net.pretrain(200)
            print('Done.')
        optimizer = torch.optim.Adam(Net.parameters(), lr=0.002)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.98, patience=100, verbose=True,
                                                                threshold=0.00005, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-16)
        startEpoch = 0
        lossList = []


    num = 0
    for para in Net.parameters():
        num+=torch.prod(torch.tensor(para.shape))
    print('The number of paras in the network is {}.'.format(num))
    return Net,optimizer,scheduler,startEpoch,lossList
