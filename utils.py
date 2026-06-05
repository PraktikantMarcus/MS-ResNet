import sys
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler


def get_network(args, cfg=None):
    """Return the requested network, forwarding dataset config parameters.

    args.net selects the architecture family.  cfg is an optional dataset
    config dict (from DATASET_CONFIGS in train_hpc.py) that supplies
    num_classes, in_channels, T, and dvs where applicable.
    """
    cfg = cfg or {}
    num_classes = cfg.get('num_classes', 1000)
    in_channels = cfg.get('in_channels', 3)
    T           = cfg.get('T', 5)
    dvs         = cfg.get('dvs', False)
    sequential  = cfg.get('sequential', False)

    if args.net == 'resnet18':
        from models.MS_ResNet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.MS_ResNet import resnet34
        net = resnet34()
    elif args.net == 'resnet104':
        from models.MS_ResNet import resnet104
        net = resnet104()
    elif args.net == 'resnet18_fast':
        from models.MS_ResNet_fast import resnet18
        net = resnet18(num_classes=num_classes, T=T)
    elif args.net == 'resnet34_fast':
        from models.MS_ResNet_fast import resnet34
        net = resnet34(num_classes=num_classes, T=T)
    elif args.net == 'resnet104_fast':
        from models.MS_ResNet_fast import resnet104
        net = resnet104(num_classes=num_classes, T=T)
    elif args.net == 'resnet20_cifar':
        from models.MS_ResNet_fast import resnet20_cifar
        net = resnet20_cifar(num_classes=num_classes, in_channels=in_channels, T=T, dvs=dvs,
                             sequential=sequential)
    elif args.net == 'resnet110_cifar':
        from models.MS_ResNet_fast import resnet110_cifar
        net = resnet110_cifar(num_classes=num_classes, in_channels=in_channels, T=T, dvs=dvs,
                              sequential=sequential)
    elif args.net == 'resnet20_cifar_fullres':
        from models.MS_ResNet_fast import resnet20_cifar_fullres
        net = resnet20_cifar_fullres(num_classes=num_classes, in_channels=in_channels, T=T,
                                     sequential=sequential)
    elif args.net == 'resnet110_cifar_fullres':
        from models.MS_ResNet_fast import resnet110_cifar_fullres
        net = resnet110_cifar_fullres(num_classes=num_classes, in_channels=in_channels, T=T,
                                      sequential=sequential)
    elif args.net == 'resnet20_dvs128':
        from models.MS_ResNet_fast import resnet20_dvs128
        net = resnet20_dvs128(num_classes=num_classes, in_channels=in_channels, T=T,
                              sequential=sequential)
    elif args.net == 'resnet110_dvs128':
        from models.MS_ResNet_fast import resnet110_dvs128
        net = resnet110_dvs128(num_classes=num_classes, in_channels=in_channels, T=T,
                               sequential=sequential)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net


def get_training_dataloader(traindir,
                            sampler=None,
                            batch_size=16,
                            num_workers=2,
                            shuffle=True,
                            persistent_workers=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ImageNet_training = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            normalize,
        ]))
    if sampler is not None:
        ImageNet_training_loader = DataLoader(
            ImageNet_training,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=True,
            persistent_workers=persistent_workers,
            sampler=DistributedSampler(ImageNet_training))
    else:
        ImageNet_training_loader = DataLoader(ImageNet_training,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              persistent_workers=persistent_workers)

    return ImageNet_training_loader


def get_test_dataloader(valdir,
                        sampler=None,
                        batch_size=16,
                        num_workers=2,
                        shuffle=False,
                        persistent_workers=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ImageNet_test = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),  # 320
            transforms.CenterCrop(224),  # 288
            transforms.ToTensor(),
            normalize,
        ]))
    if sampler is not None:
        ImageNet_test_loader = DataLoader(
            ImageNet_test,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=True,
            persistent_workers=persistent_workers,
            sampler=DistributedSampler(ImageNet_test))
    else:
        ImageNet_test_loader = DataLoader(ImageNet_test,
                                          shuffle=shuffle,
                                          num_workers=num_workers,
                                          batch_size=batch_size,
                                          persistent_workers=persistent_workers)

    return ImageNet_test_loader


def get_training_dataloader_cifar10(data_path,
                                    batch_size=128,
                                    num_workers=4,
                                    shuffle=True):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True)


def get_test_dataloader_cifar10(data_path,
                                batch_size=128,
                                num_workers=4,
                                shuffle=False):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True)
