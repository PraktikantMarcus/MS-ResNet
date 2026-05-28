# ADR 0009: Native-resolution input with stride-2 conv1 for CIFAR-10-DVS

## Status
Accepted

## Context
CIFAR-10-DVS has a native sensor resolution of 128×128. Our earlier approach resized frames to 48, 64, or 96 pixels in the dataloader collate function (bilinear interpolation on CPU before the network sees the data). This is a fixed, non-learned operation that discards spatial information uniformly.

Ablation results across three collate-resize settings:

| spatial | Best test acc | Train-test gap |
|---------|--------------|----------------|
| 48      | 71.7%        | ~10.3%         |
| 64      | 73.1%        | ~8.5%          |
| 96      | 69.4%        | ~5.4% (underfitting) |

The 96px result revealed that the model capacity (ResNet-20) is the binding constraint at high resolution, not information quantity. The 64px result is the collate-resize optimum for ResNet-20.

Hu et al. (2024) state that for CIFAR-10-DVS their MSResNet-20 uses "additional downsampling placed at the first CONV stage due to the larger input size" and that they "adopt the same data preprocessing as Fang et al. (arXiv:2007.05785)". Fang et al. feed native 128×128 frames and handle spatial reduction inside the network.

## Decision
Remove spatial resize from the CIFAR-10-DVS collate functions (`spatial=None`). Add `stride=2` to `conv1` in `ResNet_CIFAR` when `dvs=True`, reducing 128×128 → 64×64 before the residual stages. The residual stages then run at 64×64, 32×32, and 16×16 — consistent with the paper.

This makes the downsampling a **learned** operation trained end-to-end, rather than a fixed bilinear filter applied outside the model. It also matches the exact architectural description given by the paper.

## Alternatives considered

**Bilinear resize in collate at spatial=64:** Our previous best (73.1%). Achieves the same 64×64 feature map size in stage1, but via a fixed non-learned operation. Rejected as misaligned with the paper's description.

**Separate ResNet_CIFAR_DVS class (like ResNet_DVS128):** Would duplicate the CIFAR architecture class. Rejected in favour of a conditional `stride=2` inside the existing `ResNet_CIFAR.__init__` — one line of difference, no class duplication.

**Two stride-2 convs (128 → 32):** Would match the original 32×32 CIFAR feature map size throughout. Rejected because the paper describes a single "additional downsampling" step, and the 96px ablation already showed ResNet-20 underfits at stage1 resolutions above 64×64.

## Consequences
- `ResNet_CIFAR` with `dvs=True` and `dvs=False` now differ in `conv1` stride; the factory functions `resnet20_cifar` and `resnet110_cifar` pass `dvs` through correctly.
- The CIFAR-10-DVS collate functions are simplified (no interpolation call).
- The SpikingJelly `frames_number_10_split_by_number` cache (already built from the T=10 runs) is reused directly — no re-encoding needed.
- Prior checkpoint files trained with the collate-resize approach are incompatible with this model (conv1 weight shape is unchanged, but the effective receptive field differs). Do not resume old checkpoints.
