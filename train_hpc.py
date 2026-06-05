import csv
import os
import math
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import get_network

# ---------------------------------------------------------------------------
# Dataset configurations (ADR 0001, 0002, 0004, 0005, 0006)
# Each entry contains every parameter that differs between datasets.
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    'imagenet': dict(
        model='resnet104_fast',
        T=5, in_channels=3, num_classes=1000, dvs=False,
        epochs=125, batch_size=256, lr=0.1,
        optimizer='sgd', weight_decay=1e-5, momentum=0.9,
        distributed=True, workers=8,
        loss='label_smooth',
    ),
    'cifar10': dict(
        model='resnet110_cifar',
        T=5, in_channels=3, num_classes=10, dvs=False,
        epochs=200, batch_size=128, lr=0.1,
        optimizer='sgd', weight_decay=5e-4, momentum=0.9,
        distributed=False, workers=4,
        loss='ce',
    ),
    'cifar100': dict(
        model='resnet110_cifar',
        T=5, in_channels=3, num_classes=100, dvs=False,
        epochs=200, batch_size=128, lr=0.1,
        optimizer='sgd', weight_decay=5e-4, momentum=0.9,
        distributed=False, workers=4,
        loss='ce',
    ),
    'cifar10dvs': dict(
        model='resnet20_cifar_fullres',
        T=20, in_channels=2, num_classes=10, dvs=True,
        spatial=None,  # native 128×128; no conv1 downsampling — stage1 at 128×128 (ADR-0010)
        epochs=125, batch_size=128, lr=1e-3,
        sequential=True,  # sequential conv matches supervisor's setup; enables B=128 at fullres
        optimizer='adamw', weight_decay=0.06,
        distributed=False, workers=4,
        loss='ce',
    ),
    'dvs128': dict(
        model='resnet20_dvs128',
        T=16, in_channels=2, num_classes=11, dvs=True,
        spatial=None,  # keep native 128×128; stride-2 stem handles downsampling (ADR 0007)
        epochs=100, batch_size=16, lr=1e-3,
        optimizer='adamw', weight_decay=0.06,
        distributed=False, workers=4,
        loss='ce',
    ),
}

SEED = 445


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets_one_hot = (1 - self.epsilon) * targets_one_hot + \
                          self.epsilon / self.num_classes
        return (-targets_one_hot * log_probs).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def _split_to_train_test(origin_dataset, num_classes, train_ratio=0.9):
    """Random 90/10 split per class, seeded for reproducibility."""
    label_idx = [[] for _ in range(num_classes)]
    for i, (_, y) in enumerate(origin_dataset):
        label_idx[y].append(i)
    rng = np.random.default_rng(SEED)
    train_idx, test_idx = [], []
    for indices in label_idx:
        indices = list(indices)
        rng.shuffle(indices)
        pos = math.ceil(len(indices) * train_ratio)
        train_idx.extend(indices[:pos])
        test_idx.extend(indices[pos:])
    from torch.utils.data import Subset
    return Subset(origin_dataset, train_idx), Subset(origin_dataset, test_idx)


def build_dataloaders(dataset_name, cfg, data_path):
    workers   = cfg['workers']
    batch     = cfg['batch_size']
    T         = cfg['T']

    if dataset_name == 'imagenet':
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(
            os.path.join(data_path, 'train'),
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.AutoAugment(),
                transforms.ToTensor(), normalize,
            ]))
        val_set = datasets.ImageFolder(
            os.path.join(data_path, 'val'),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(), normalize,
            ]))
        train_sampler = DistributedSampler(train_set)
        val_sampler   = DistributedSampler(val_set, shuffle=False)
        train_loader = DataLoader(train_set, batch_size=batch,
                                  sampler=train_sampler, num_workers=workers,
                                  pin_memory=True, persistent_workers=True)
        val_loader   = DataLoader(val_set, batch_size=batch,
                                  sampler=val_sampler,   num_workers=workers,
                                  pin_memory=True, persistent_workers=True)
        return train_loader, val_loader, train_sampler

    if dataset_name in ('cifar10', 'cifar100'):
        cls = datasets.CIFAR10 if dataset_name == 'cifar10' else datasets.CIFAR100
        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean, std)
        train_set = cls(root=data_path, train=True, download=False,
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(), normalize,
                        ]))
        val_set = cls(root=data_path, train=False, download=False,
                      transform=transforms.Compose([
                          transforms.ToTensor(), normalize,
                      ]))
        train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                                  num_workers=workers, pin_memory=True)
        val_loader   = DataLoader(val_set,   batch_size=batch, shuffle=False,
                                  num_workers=workers, pin_memory=True)
        return train_loader, val_loader, None

    if dataset_name == 'cifar10dvs':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        from autoaugment import SNNAugmentWide
        origin_set = CIFAR10DVS(root=data_path, data_type='frame',
                                frames_number=T, split_by='number')
        train_set, val_set = _split_to_train_test(origin_set, num_classes=10)
        spatial = cfg.get('spatial')
        flip = transforms.RandomHorizontalFlip(p=0.5)
        snn_aug = SNNAugmentWide()

        def dvs_collate_train(batch):
            imgs, labels = zip(*batch)
            imgs = torch.stack([torch.tensor(img, dtype=torch.float32)
                                for img in imgs])          # (B, T, C, H, W)
            if spatial is not None:
                B_, T_, C_, H_, W_ = imgs.shape
                imgs = torch.nn.functional.interpolate(
                    imgs.view(B_ * T_, C_, H_, W_),
                    size=(spatial, spatial), mode='bilinear', align_corners=False)
                imgs = imgs.view(B_, T_, C_, spatial, spatial)
            imgs = torch.stack([snn_aug(flip(imgs[i])) for i in range(len(imgs))])
            return imgs, torch.tensor(labels)

        def dvs_collate_val(batch):
            imgs, labels = zip(*batch)
            imgs = torch.stack([torch.tensor(img, dtype=torch.float32)
                                for img in imgs])
            if spatial is not None:
                B_, T_, C_, H_, W_ = imgs.shape
                imgs = torch.nn.functional.interpolate(
                    imgs.view(B_ * T_, C_, H_, W_),
                    size=(spatial, spatial), mode='bilinear', align_corners=False)
                imgs = imgs.view(B_, T_, C_, spatial, spatial)
            return imgs, torch.tensor(labels)

        train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                                  num_workers=workers, pin_memory=True,
                                  collate_fn=dvs_collate_train)
        val_loader   = DataLoader(val_set,   batch_size=batch, shuffle=False,
                                  num_workers=workers, pin_memory=True,
                                  collate_fn=dvs_collate_val)
        return train_loader, val_loader, None

    if dataset_name == 'dvs128':
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        from autoaugment import SNNAugmentWide
        train_set = DVS128Gesture(root=data_path, train=True,
                                  data_type='frame', frames_number=T,
                                  split_by='number')
        val_set   = DVS128Gesture(root=data_path, train=False,
                                  data_type='frame', frames_number=T,
                                  split_by='number')
        flip = transforms.RandomHorizontalFlip(p=0.5)
        snn_aug = SNNAugmentWide()

        def dvs_collate(batch):
            imgs, labels = zip(*batch)
            imgs = torch.stack([torch.tensor(img, dtype=torch.float32)
                                for img in imgs])
            imgs = torch.stack([snn_aug(flip(imgs[i])) for i in range(len(imgs))])
            return imgs, torch.tensor(labels)

        train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                                  num_workers=workers, pin_memory=True,
                                  collate_fn=dvs_collate)
        val_loader   = DataLoader(val_set,   batch_size=batch, shuffle=False,
                                  num_workers=workers, pin_memory=True)
        return train_loader, val_loader, None

    raise ValueError(f'Unknown dataset: {dataset_name}')


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(net, loader, optimizer, loss_fn, scaler, writer,
                    epoch, distributed, train_sampler):
    net.train()
    if distributed and train_sampler is not None:
        train_sampler.set_epoch(epoch)
    running_loss = 0.0
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        total += images.size(0)

        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = net(images)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()

        if batch_idx % 50 == 49:
            avg_loss = running_loss / 50
            print('Epoch {} [{}/{}]  Loss: {:.4f}  LR: {:.6f}  Time: {:.1f}s'.format(
                epoch, total, len(loader.dataset), avg_loss,
                optimizer.param_groups[0]['lr'], time.time() - start))
            writer.add_scalar('Train/loss', avg_loss,
                               (epoch - 1) * len(loader) + batch_idx)
            running_loss = 0.0

    acc = 100.0 * correct / total
    epoch_avg_loss = total_loss / len(loader)
    print('Epoch {} train acc: {:.2f}%  time: {:.1f}s'.format(
        epoch, acc, time.time() - start))
    writer.add_scalar('Train/acc', acc, epoch)
    return epoch_avg_loss, acc


@torch.no_grad()
def evaluate(net, loader, loss_fn, writer, epoch):
    net.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    start = time.time()

    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        total += images.size(0)
        outputs = net(images)
        test_loss += loss_fn(outputs, labels).item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()

    acc = 100.0 * correct / total
    avg_loss = test_loss / len(loader)
    print('Test  Loss: {:.4f}  Acc: {:.2f}%  Time: {:.1f}s'.format(
        avg_loss, acc, time.time() - start))
    writer.add_scalar('Test/acc', acc, epoch)
    writer.add_scalar('Test/loss', avg_loss, epoch)
    return avg_loss, acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='dataset name — selects model, T, optimizer, etc.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='directory for checkpoints and tensorboard logs')
    parser.add_argument('--workers', type=int, default=None,
                        help='override dataloader worker count from config')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='override batch size from config')
    parser.add_argument('--resume', type=str, default='',
                        help='path to checkpoint-latest.pth to resume from')
    # Distributed
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    if args.workers is not None:
        cfg = {**cfg, 'workers': args.workers}
    if args.batch_size is not None:
        cfg = {**cfg, 'batch_size': args.batch_size}

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------
    distributed = cfg['distributed']
    if distributed:
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    args.net = cfg['model']
    net = get_network(args, cfg)
    net.cuda()

    if distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[local_rank])

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, train_sampler = build_dataloaders(
        args.dataset, cfg, args.data_path)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    if cfg['loss'] == 'label_smooth':
        loss_fn = CrossEntropyLabelSmooth(cfg['num_classes']).cuda()
    else:
        loss_fn = nn.CrossEntropyLoss().cuda()

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    EPOCHS = cfg['epochs']
    if cfg['optimizer'] == 'adamw':
        optimizer = optim.AdamW(net.parameters(),
                                lr=cfg['lr'],
                                weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.SGD(net.parameters(),
                              lr=cfg['lr'],
                              momentum=cfg.get('momentum', 0.9),
                              weight_decay=cfg['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=0)
    scaler = GradScaler('cuda')

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 1
    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    csv_path = os.path.join(args.output_dir, 'training_log.csv')
    csv_fields = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'lr']

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cuda')
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc    = ckpt['best_acc']
        print('Resumed from epoch {}, best_acc={:.2f}%'.format(
            ckpt['epoch'], best_acc))

    # Write CSV header only when starting fresh (not resuming)
    if not args.resume:
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print('Dataset: {}  Model: {}  T: {}  Epochs: {}  Optimizer: {}'.format(
        args.dataset, cfg['model'], cfg['T'], EPOCHS, cfg['optimizer']))
    print('Start: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            net, train_loader, optimizer, loss_fn, scaler,
            writer, epoch, distributed, train_sampler)
        scheduler.step()
        test_loss, acc = evaluate(net, val_loader, loss_fn, writer, epoch)

        with open(csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow({
                'epoch':      epoch,
                'train_loss': f'{train_loss:.6f}',
                'train_acc':  f'{train_acc:.2f}',
                'test_loss':  f'{test_loss:.6f}',
                'test_acc':   f'{acc:.2f}',
                'lr':         f'{optimizer.param_groups[0]["lr"]:.8f}',
            })

        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict() if not distributed else net.module.state_dict(),
                       os.path.join(args.output_dir, 'best_model.pth'))

        torch.save({
            'model':     net.state_dict() if not distributed else net.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler':    scaler.state_dict(),
            'epoch':     epoch,
            'best_acc':  best_acc,
            'dataset':   args.dataset,
            'cfg':       cfg,
        }, os.path.join(args.output_dir, 'checkpoint-latest.pth'))

    print('Training complete. Best acc: {:.2f}%'.format(best_acc))
    writer.close()
