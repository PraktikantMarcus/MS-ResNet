import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from utils import get_network, get_training_dataloader_cifar10, get_test_dataloader_cifar10

EPOCHS = 200


def train(epoch):
    running_loss = 0.0
    start = time.time()
    net.train()
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        total += images.size(0)
        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = net(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if batch_idx % 50 == 49:
            print('Epoch {} [{}/{}]  Loss: {:.4f}  LR: {:.6f}  Time: {:.1f}s'.format(
                epoch, total, len(train_loader.dataset),
                running_loss / 50, optimizer.param_groups[0]['lr'],
                time.time() - start))
            writer.add_scalar('Train/loss', running_loss / 50,
                               (epoch - 1) * len(train_loader) + batch_idx)
            running_loss = 0.0
    acc = 100.0 * correct / total
    print('Epoch {} train acc: {:.2f}%  time: {:.1f}s'.format(
        epoch, acc, time.time() - start))
    writer.add_scalar('Train/acc', acc, epoch)


@torch.no_grad()
def eval_training(epoch):
    start = time.time()
    net.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        total += images.size(0)
        outputs = net(images)
        test_loss += loss_function(outputs, labels).item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
    acc = 100.0 * correct / total
    print('Test  Loss: {:.4f}  Acc: {:.2f}%  Time: {:.1f}s'.format(
        test_loss / len(test_loader), acc, time.time() - start))
    writer.add_scalar('Test/acc', acc, epoch)
    writer.add_scalar('Test/loss', test_loss / len(test_loader), epoch)
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet110_cifar10')
    parser.add_argument('-b', type=int, default=128,
                        help='batch size')
    parser.add_argument('-lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('-data_path', type=str, default='/data/cifar10',
                        help='directory that contains the CIFAR-10 dataset')
    parser.add_argument('-output_dir', type=str, default='output_cifar10',
                        help='directory for checkpoints and tensorboard logs')
    parser.add_argument('-workers', type=int, default=4,
                        help='dataloader workers')
    parser.add_argument('-resume', type=str, default='',
                        help='path to checkpoint-latest.pth to resume from')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    SEED = 445
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    net = get_network(args)
    net.cuda()

    train_loader = get_training_dataloader_cifar10(
        data_path=args.data_path,
        batch_size=args.b,
        num_workers=args.workers,
    )
    test_loader = get_test_dataloader_cifar10(
        data_path=args.data_path,
        batch_size=args.b,
        num_workers=args.workers,
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=0, last_epoch=0)
    scaler = GradScaler('cuda')

    start_epoch = 1
    best_acc = 0.0

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda')
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print('Resumed from epoch {}, best_acc={:.2f}%'.format(
            checkpoint['epoch'], best_acc))

    print('Start time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))

    for epoch in range(start_epoch, EPOCHS + 1):
        train(epoch)
        scheduler.step()
        acc = eval_training(epoch)

        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(),
                       os.path.join(args.output_dir, 'best_model.pth'))

        torch.save({
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
        }, os.path.join(args.output_dir, 'checkpoint-latest.pth'))

    print('Training complete. Best acc: {:.2f}%'.format(best_acc))
    writer.close()
