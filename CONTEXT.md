# MS-ResNet Glossary

## MS-ResNet
**Membrane-based Spiking ResNet.** A spiking neural network architecture that combines residual learning with leaky integrate-and-fire neuron dynamics. "MS" stands for Membrane-based Spiking — not "multi-scale." Proposed in Hu et al. (IEEE TNNLS 2024).

## Spiking Neuron / mem_update
The core neuron model. Each neuron maintains a membrane potential that decays exponentially and fires a binary spike when it exceeds a threshold. Implemented in `mem_update` inside `MS_ResNet_fast.py`. The firing function uses a surrogate gradient (`ActFun`) for backpropagation.

## time_window (T)
The number of discrete timesteps the network processes per sample. For static datasets (CIFAR-10, ImageNet), the same input frame is replicated T times — T carries no new information and exists solely so membrane dynamics can propagate. For DVS datasets, T is the number of event bins and carries real temporal structure. T is per-dataset configurable (not a single global constant).

## Static Dataset
A dataset of RGB images (e.g. CIFAR-10, CIFAR-100, ImageNet). Input tensor shape: `(T, B, 3, H, W)`, where the same frame is replicated across the T dimension.

## DVS Dataset
A dataset captured by a Dynamic Vision Sensor, producing a stream of `(x, y, polarity, timestamp)` events. Examples: CIFAR-10-DVS, DVS128 Gesture. Requires event-to-frame encoding before the network can process the data. Input tensor shape after encoding: `(T, B, 2, H, W)`, where 2 channels correspond to ON/OFF polarity and each of the T frames accumulates events from one time bin.

## Event-to-Frame Encoding
The conversion of raw DVS events into a fixed-size tensor. Events are binned into T groups, and per-pixel event counts per polarity channel are accumulated. Handled by SpikingJelly's dataset classes (`data_type='frame'`, `frames_number=T`). The binning strategy (fixed count vs. fixed time) is chosen per dataset following the SNN literature.

## Spike Sequence (RTL context)
The binarized input tensor produced from a DVS frame encoding — the sequence of binary spatial spike maps across T timesteps that will be replayed by an RTL hardware testbench. Distinct from the raw event stream and from the floating-point frame tensor used during training. Saving spike sequences is a post-training export step, not part of the training loop.

## Surrogate Gradient
A differentiable approximation of the spike-generation step used during backpropagation. The true spike function is a step function (non-differentiable). `ActFun` approximates its gradient with a piecewise-linear function (lens=0.5). Used in both the original and fast model variants.

## Fast Variant
The performance-optimised model (`MS_ResNet_fast.py`). Merges the time (T) and batch (B) dimensions before the convolution call, reducing CUDA kernel launches by T−1 per layer. Functionally equivalent to the original; preferred for all training runs.

## Model Naming Convention
Model names encode architecture depth and structural family, not dataset. The class count and input channels are constructor parameters, not part of the name.

| Name | Architecture family | Typical use |
|---|---|---|
| `resnet18`, `resnet34`, `resnet104` | ImageNet stem (7×7 stride-2 + maxpool, 4 stages) | ImageNet |
| `resnet110_cifar` | CIFAR stem (3×3 no-stride, 3 stages) | CIFAR-10, CIFAR-100, CIFAR-10-DVS |
| `resnet110_dvs128` | CIFAR stem + one stride-2 spiking conv (128→64) | DVS128 Gesture (128×128 input) |

All variants accept `in_channels` and `num_classes` as constructor arguments.

## Unified Training Script
A single `train_hpc.py` entry point for all datasets, selected via `--dataset`. Dataset-specific parameters are resolved from an inline `DATASET_CONFIGS` dict keyed by dataset name. The training loop, checkpointing, and logging are shared. The previous per-dataset scripts (`train_amp_hpc.py`, `train_cifar10_hpc.py`) are kept for reference only.

The config dict captures all values that differ between datasets:

| Key | imagenet | cifar10 | cifar100 | cifar10dvs | dvs128 |
|---|---|---|---|---|---|
| `model` | `resnet104_fast` | `resnet110_cifar` | `resnet110_cifar` | `resnet110_cifar` | `resnet110_dvs128` |
| `T` | 5 | 5 | 5 | 10 | 16 |
| `in_channels` | 3 | 3 | 3 | 2 | 2 |
| `num_classes` | 1000 | 10 | 100 | 10 | 11 |
| `epochs` | 125 | 200 | 200 | 100 | 100 |
| `optimizer` | `sgd` | `sgd` | `sgd` | `adamw` | `adamw` |
| `batch_size` | 256 | 128 | 128 | TBD | TBD |
| `distributed` | True | False | False | False | False |

T differs between neuromorphic datasets because CIFAR-10-DVS has short, simple temporal structure (slow camera rotation of static images) while DVS128 Gesture has richer, longer temporal dynamics (full hand gesture trajectories). T=10 and T=16 are the respective community standards.

## Augmentation Strategy
Static datasets use the existing augmentation pipeline (random crop + flip for CIFAR; AutoAugment for ImageNet). Neuromorphic datasets (CIFAR-10-DVS, DVS128 Gesture) use random horizontal flip plus `SNNAugmentWide` — a geometric-only augmentation (shear, translate, rotate, cutout) that avoids colour transforms which are meaningless for polarity-channel event data. `SNNAugmentWide` is sourced from the QKFormer codebase and lives in `autoaugment.py` in this repository. Using the same augmentation as QKFormer makes accuracy comparisons against that baseline valid.

## Optimizer Convention
Static-image datasets (CIFAR-10, CIFAR-100, ImageNet) use **SGD with momentum**, consistent with the original paper. Neuromorphic datasets (CIFAR-10-DVS, DVS128 Gesture) use **AdamW**, consistent with the SNN literature those experiments are compared against. The optimizer is part of the dataset config.
