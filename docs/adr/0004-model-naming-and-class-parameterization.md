# ADR 0004: Model naming convention and parameterized output classes

## Status
Accepted

## Context
The original codebase named the CIFAR model `resnet110_cifar10`, embedding the dataset name and implicitly the class count (10) in the model identifier. Extending to CIFAR-100 and CIFAR-10-DVS breaks this convention: both share the same architecture but differ in class count and input channels. A consistent naming scheme is needed before adding new variants.

The ImageNet models (`resnet18`, `resnet34`, `resnet104`) encode only architecture depth, not dataset. The CIFAR naming should follow the same principle.

## Decision
Model names encode **architecture depth and structural family only**. Dataset-specific values (number of output classes, number of input channels) are constructor parameters.

Naming:
- `resnet110_cifar` — CIFAR-style stem (3×3 conv, no stride, 3 residual stages). Used for CIFAR-10, CIFAR-100, and CIFAR-10-DVS (48×48).
- `resnet110_dvs128` — Same backbone with an additional initial downsampling stem. Used for DVS128 Gesture (128×128 input).

The previous `resnet110_cifar10` name is retired.

## Alternatives considered
Keep separate named models per dataset (`resnet110_cifar10`, `resnet110_cifar100`, `resnet110_cifar10dvs`). Rejected because it multiplies nearly-identical model definitions and obscures the shared structure.

## Consequences
`get_network()` in `utils.py` must be updated to dispatch on `resnet110_cifar` and `resnet110_dvs128`, passing `in_channels` and `num_classes` from the dataset config. Existing CIFAR-10 checkpoint filenames that encode the old model name will not match the new identifier.
