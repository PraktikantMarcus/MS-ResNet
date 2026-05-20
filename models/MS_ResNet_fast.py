import torch
import torch.nn as nn
import torch.nn.functional as F

# Optimised forward pass for MS-ResNet:
#   Snn_Conv2d merges the T and B dimensions so all timesteps go through
#   a single conv2d call instead of T sequential calls.  This eliminates
#   T-1 redundant CUDA kernel launches per layer and gives the GPU a
#   larger effective batch to amortise launch overhead.
#
#   mem_update removes the spurious .to(device) re-allocations (zeros_like
#   already inherits the device) and reads the loop bound from the tensor
#   instead of the global time_window.

thresh = 0.5
lens = 0.5
decay = 0.25
num_classes = 1000
time_window = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        temp = temp / (2 * lens)
        return grad_input * temp.float()


act_fun = ActFun.apply


class mem_update(nn.Module):

    def __init__(self):
        super(mem_update, self).__init__()

    def forward(self, x):
        # Reads output[i-1] instead of cloning mem each step, eliminating
        # T tensor allocations per call (515 per forward pass across all layers).
        T = x.size(0)
        output = torch.zeros_like(x)
        mem = x[0]
        output[0] = act_fun(mem)
        for i in range(1, T):
            mem = mem * decay * (1 - output[i - 1].detach()) + x[i]
            output[i] = act_fun(mem)
        return output


class batch_norm_2d(nn.Module):
    """TDBN"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()
        self.bn = BatchNorm3d1(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class batch_norm_2d1(nn.Module):
    """TDBN-Zero init"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm3d2(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d1(torch.nn.BatchNorm3d):

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)
            nn.init.zeros_(self.bias)


class BatchNorm3d2(torch.nn.BatchNorm3d):

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0)
            nn.init.zeros_(self.bias)


class Snn_Conv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 marker='b'):
        super(Snn_Conv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        self.marker = marker

    def forward(self, input):
        # input: [T, B, C_in, H, W]
        # Merge T and B so all timesteps go through one conv2d call,
        # then split back. Same FLOPs, far fewer CUDA kernel launches.
        # reshape (not view) because mem_update output is non-contiguous.
        T, B = input.size(0), input.size(1)
        x = input.reshape(T * B, input.size(2), input.size(3), input.size(4))
        out = F.conv2d(x, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out.reshape(T, B, self.out_channels, out.size(2), out.size(3))


######################################################################################################################
class BasicBlock_104(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            mem_update(),
            Snn_Conv2d(in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=stride,
                       padding=1,
                       bias=False),
            batch_norm_2d(out_channels),
            mem_update(),
            Snn_Conv2d(out_channels,
                       out_channels * BasicBlock_104.expansion,
                       kernel_size=3,
                       padding=1,
                       bias=False),
            batch_norm_2d1(out_channels * BasicBlock_104.expansion),
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock_104.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2)),
                Snn_Conv2d(in_channels,
                           out_channels * BasicBlock_104.expansion,
                           kernel_size=1,
                           stride=1,
                           bias=False),
                batch_norm_2d(out_channels * BasicBlock_104.expansion),
            )

    def forward(self, x):
        return (self.residual_function(x) + self.shortcut(x))


class ResNet_104(nn.Module):
    def __init__(self, block, num_block, num_classes=1000):
        super().__init__()
        k = 1
        self.in_channels = 64 * k
        self.conv1 = nn.Sequential(
            Snn_Conv2d(3, 64 * k, kernel_size=3, padding=1, stride=2),
            Snn_Conv2d(64 * k, 64 * k, kernel_size=3, padding=1, stride=1),
            Snn_Conv2d(64 * k, 64 * k, kernel_size=3, padding=1, stride=1),
            batch_norm_2d(64 * k),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.mem_update = mem_update()
        self.conv2_x = self._make_layer(block, 64 * k, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128 * k, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256 * k, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512 * k, num_block[3], 2)
        self.fc = nn.Linear(512 * block.expansion * k, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Replicate input across the time dimension; use x.device for safety
        input = x.unsqueeze(0).expand(time_window, -1, -1, -1, -1).contiguous()
        output = self.conv1(input)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.mem_update(output)
        output = F.adaptive_avg_pool3d(output, (None, 1, 1))
        output = output.view(output.size()[0], output.size()[1], -1)
        output = output.sum(dim=0) / output.size()[0]
        output = self.dropout(output)
        output = self.fc(output)
        return output


def resnet104():
    return ResNet_104(BasicBlock_104, [3, 8, 32, 8])


class BasicBlock_18(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            mem_update(),
            Snn_Conv2d(in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=stride,
                       padding=1,
                       bias=False),
            batch_norm_2d(out_channels),
            mem_update(),
            Snn_Conv2d(out_channels,
                       out_channels * BasicBlock_18.expansion,
                       kernel_size=3,
                       padding=1,
                       bias=False),
            batch_norm_2d1(out_channels * BasicBlock_18.expansion),
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock_18.expansion * out_channels:
            self.shortcut = nn.Sequential(
                Snn_Conv2d(in_channels,
                           out_channels * BasicBlock_18.expansion,
                           kernel_size=1,
                           stride=stride,
                           bias=False),
                batch_norm_2d(out_channels * BasicBlock_18.expansion),
            )

    def forward(self, x):
        return (self.residual_function(x) + self.shortcut(x))


class ResNet_origin_18(nn.Module):
    def __init__(self, block, num_block, num_classes=1000):
        super().__init__()
        k = 1
        self.in_channels = 64 * k
        self.conv1 = nn.Sequential(
            Snn_Conv2d(3,
                       64 * k,
                       kernel_size=7,
                       padding=3,
                       bias=False,
                       stride=2),
            batch_norm_2d(64 * k),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.mem_update = mem_update()
        self.conv2_x = self._make_layer(block, 64 * k, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128 * k, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256 * k, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512 * k, num_block[3], 2)
        self.fc = nn.Linear(512 * block.expansion * k, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        input = x.unsqueeze(0).expand(time_window, -1, -1, -1, -1).contiguous()
        output = self.conv1(input)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.mem_update(output)
        output = F.adaptive_avg_pool3d(output, (None, 1, 1))
        output = output.view(output.size()[0], output.size()[1], -1)
        output = output.sum(dim=0) / output.size()[0]
        output = self.fc(output)
        return output


def resnet18():
    return ResNet_origin_18(BasicBlock_18, [2, 2, 2, 2])


def resnet34():
    return ResNet_origin_18(BasicBlock_18, [3, 4, 6, 3])


class ResNet_CIFAR18(nn.Module):
    """ResNet-18 adapted for CIFAR (32×32): 3×3 stem at stride=1, no MaxPool,
    first stage at stride=1 so spatial dims reach 4×4 before global pool."""

    def __init__(self, block, num_block, num_classes=10):
        super().__init__()
        k = 1
        self.in_channels = 64 * k
        self.conv1 = nn.Sequential(
            Snn_Conv2d(3, 64 * k, kernel_size=3, padding=1, bias=False),
            batch_norm_2d(64 * k),
        )
        self.mem_update = mem_update()
        self.conv2_x = self._make_layer(block, 64 * k,  num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128 * k, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256 * k, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512 * k, num_block[3], 2)
        self.fc = nn.Linear(512 * block.expansion * k, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        input = x.unsqueeze(0).expand(time_window, -1, -1, -1, -1).contiguous()
        output = self.conv1(input)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.mem_update(output)
        output = F.adaptive_avg_pool3d(output, (None, 1, 1))
        output = output.view(output.size()[0], output.size()[1], -1)
        output = output.sum(dim=0) / output.size()[0]
        output = self.fc(output)
        return output


def resnet18_cifar10():
    return ResNet_CIFAR18(BasicBlock_18, [2, 2, 2, 2], num_classes=10)
