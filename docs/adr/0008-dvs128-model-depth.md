# ADR 0008: ResNet-20 depth for DVS128 Gesture

## Status
Accepted

## Context
Hu et al. (2024) report MS-ResNet results for CIFAR-10-DVS but **do not report results for DVS128 Gesture**. Our DVS128 experiment is therefore original work beyond the paper's scope, requiring an independent depth choice.

The paper uses MSResNet-20 (n=3, 20 weight layers) for CIFAR-10-DVS without explicitly motivating the depth choice. Our own CIFAR-10-DVS run with ResNet-110 (n=18) achieved 74.60% against the paper's 75.56%, despite using a model with 6× more layers — suggesting the dataset size is the bottleneck, not model capacity. CIFAR-10-DVS has approximately 9,000 training samples, and a 110-layer network offers little benefit over a 20-layer one at that scale.

DVS128 Gesture is significantly smaller:

| Dataset        | Train samples | Test samples |
|----------------|--------------|-------------|
| CIFAR-10-DVS   | ~9,000       | ~1,000      |
| DVS128 Gesture | ~1,176       | ~288        |

DVS128 Gesture has roughly 8× fewer training samples than CIFAR-10-DVS.

## Decision
Use `resnet20_dvs128` (n=3) for DVS128 Gesture experiments. If a 110-layer network adds negligible value on CIFAR-10-DVS (~9k samples), it is unlikely to help on DVS128 Gesture (~1.2k samples) and is likely to overfit.

## Alternatives considered
**ResNet-110 (n=18):** ~660k parameters vs ~74k for ResNet-20. With only ~1,176 training samples across 11 classes (~107 per class), a 110-layer network is at high risk of overfitting. Rejected.

**Follow the paper exactly:** The paper provides no DVS128 baseline for MS-ResNet, so there is no depth to follow. The closest analogous choice in the paper is the CIFAR-10-DVS / ResNet-20 pairing.

## Consequences
- DVS128 results cannot be directly compared to any published MS-ResNet number (none exist).
- The ResNet-20 choice is a principled extrapolation from the paper's own architecture selection and our empirical observation on CIFAR-10-DVS, appropriate for a thesis discussion.
- If a future baseline with ResNet-110 on DVS128 is needed for ablation, `resnet110_dvs128` remains available in the codebase and can be selected by editing the `model` key in `DATASET_CONFIGS`.
