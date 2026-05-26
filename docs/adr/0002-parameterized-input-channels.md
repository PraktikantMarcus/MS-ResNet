# ADR 0002: Parameterized input channels in model constructor

## Status
Accepted

## Context
Static datasets (CIFAR-10, CIFAR-100, ImageNet) have 3 RGB input channels. DVS datasets (CIFAR-10-DVS, DVS128 Gesture) have 2 polarity channels (ON/OFF). The first convolutional layer of the current architecture is hardcoded to 3 input channels. Extending to DVS datasets requires adapting this layer.

## Decision
Make `in_channels` a constructor argument on the model class (e.g. `ResNet(in_channels=3)` for static, `ResNet(in_channels=2)` for DVS). All residual blocks downstream are unchanged. Each dataset configuration specifies its own `in_channels`.

## Alternatives considered
**1×1 projection adapter:** Add a trainable `Conv2d(2, 3, kernel_size=1)` before the backbone so the backbone always sees 3 channels. Useful if transferring pretrained static-image weights to a DVS task, but adds an unnecessary layer when training from scratch.

**Zero-padding to 3 channels:** Append a zero channel to DVS data. Trivial but wastes filter capacity.

Both alternatives were rejected because all experiments train from scratch, making the parameterized constructor the cleanest option with no redundant modules.

## Consequences
Dataset configurations must specify `in_channels`. Pretrained checkpoints from static and DVS datasets are not weight-compatible at the first layer.
