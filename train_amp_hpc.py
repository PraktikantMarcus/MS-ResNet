import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader


def train(epoch, args):
    running_loss = 0
    start = time.time()
    net.train()
    correct = 0.0
    num_sample = 0
    for batch_index, (images, labels) in enumerate(ImageNet_training_loader):
        if args.gpu:
            labels = labels.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
        num_sample += images.size()[0]
        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = net(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n_iter = (epoch - 1) * len(ImageNet_training_loader) + batch_index + 1
        if batch_index % 10 == 9:
            if args.local_rank == 0:
                print(
                    'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'
                    .format(running_loss / 10,
                            optimizer.param_groups[0]['lr'],
                            epoch=epoch,
                            trained_samples=batch_index * args.b + len(images),
                            total_samples=len(ImageNet_training_loader.dataset)))
                print('training time consumed: {:.2f}s'.format(time.time() -
                                                               start))
                writer.add_scalar('Train/avg_loss', running_loss / 10, n_iter)
                writer.add_scalar('Train/avg_loss_numpic', running_loss / 10,
                                  n_iter * args.b)
            running_loss = 0
    finish = time.time()
    if args.local_rank == 0:
        writer.add_scalar('Train/acc', correct / num_sample * 100, epoch)
    print("Training accuracy: {:.2f} of epoch {}".format(
        correct / num_sample * 100, epoch))
    print('epoch {} training time consumed: {:.2f}s'.format(
        epoch, finish - start))


@torch.no_grad()
def eval_training(epoch, args):

    start = time.time()
    net.eval()

    test_loss = 0.0
    correct = 0.0
    real_batch = 0
    for (images, labels) in ImageNet_test_loader:
        real_batch += images.size()[0]
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Evaluating Network.....')
    print(
        'Test set: Average loss: {:.4f}, Accuracy: {:.4f}%, Time consumed:{:.2f}s'
        .format(test_loss * args.b / len(ImageNet_test_loader.dataset),
                correct.float() / real_batch * 100, finish - start))

    if args.local_rank == 0:
        writer.add_scalar(
            'Test/Average loss',
            test_loss * args.b / len(ImageNet_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy',
                          correct.float() / real_batch * 100, epoch)

    return correct.float() / len(ImageNet_test_loader.dataset)


# for resnet-104
class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes=1000, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1),
                                                       1)
        targets = (1 -
                   self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu',
                        action='store_true',
                        default=True,
                        help='use gpu or not')
    parser.add_argument('-b',
                        type=int,
                        default=256,
                        help='batch size for dataloader')
    parser.add_argument('-lr',
                        type=float,
                        default=0.1,
                        help='initial learning rate')
    parser.add_argument('-data_path',
                        type=str,
                        default='/data/imagenet',
                        help='path to imagenet dataset root (expects train/ and val/ subdirs)')
    parser.add_argument('-output_dir',
                        type=str,
                        default='output',
                        help='directory for checkpoints, logs, and tensorboard')
    parser.add_argument('-workers',
                        type=int,
                        default=8,
                        help='dataloader workers per GPU process')
    parser.add_argument('-resume',
                        type=str,
                        default='',
                        help='path to checkpoint-latest.pth to resume training')
    parser.add_argument('--local_rank',
                        default=-1,
                        type=int,
                        help='node rank for distributed training (legacy; torchrun uses LOCAL_RANK env var)')
    args = parser.parse_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    print(args.local_rank)
    # device_id handles clusters that restrict CUDA_VISIBLE_DEVICES per-process
    # (each process sees only 1 GPU as device 0) as well as the shared-visibility case
    device_id = args.local_rank % max(torch.cuda.device_count(), 1)
    torch.cuda.set_device(device_id)
    torch.distributed.init_process_group(backend='nccl')

    SEED = 445
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    torch.backends.cudnn.benchmark = True

    net = get_network(args)
    net.cuda()
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[device_id])
    torch._dynamo.config.optimize_ddp = False
    net = torch.compile(net)

    world_size = torch.distributed.get_world_size()
    if world_size > 1 and args.local_rank == 0:
        print("Let's use", world_size, "GPUs!")

    ImageNet_training_loader = get_training_dataloader(
        traindir=os.path.join(args.data_path, 'train'),
        num_workers=args.workers,
        batch_size=args.b // world_size,
        shuffle=False,
        sampler=1,
        persistent_workers=True,
    )

    ImageNet_test_loader = get_test_dataloader(
        valdir=os.path.join(args.data_path, 'val'),
        num_workers=args.workers,
        batch_size=args.b // world_size,
        shuffle=False,
        sampler=1,
        persistent_workers=True,
    )

    b_lr = args.lr
    loss_function = CrossEntropyLabelSmooth()
    optimizer = optim.SGD([{
        'params': net.parameters(),
        'initial_lr': b_lr
    }],
                          momentum=0.9,
                          lr=b_lr,
                          weight_decay=1e-5)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.EPOCH, eta_min=0, last_epoch=0)
    scaler = GradScaler('cuda')

    start_epoch = 1
    best_acc = 0.0

    if args.resume:
        map_location = lambda storage, loc: storage.cuda(device_id)
        checkpoint = torch.load(args.resume, map_location=map_location)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        if args.local_rank == 0:
            print(f"Resumed from epoch {checkpoint['epoch']}, best_acc={best_acc:.4f}")

    if args.local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(
            log_dir=os.path.join(args.output_dir, 'tensorboard'))

    for epoch in range(start_epoch, settings.EPOCH + 1):
        ImageNet_training_loader.sampler.set_epoch(epoch)
        train(epoch, args)
        train_scheduler.step()
        acc = eval_training(epoch, args).item()

        if acc > best_acc:
            best_acc = acc
            if args.local_rank == 0:
                torch.save(
                    net.state_dict(),
                    os.path.join(args.output_dir, 'best_model.pth'))

        # Save full training state every 5 epochs for preemption recovery
        if args.local_rank == 0:
            torch.save({
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': train_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
            }, os.path.join(args.output_dir, 'checkpoint-latest.pth'))

    if args.local_rank == 0:
        writer.close()
