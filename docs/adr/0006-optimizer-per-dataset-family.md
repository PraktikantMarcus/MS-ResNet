# ADR 0006: Optimizer selection per dataset family

## Status
Accepted

## Context
The original MS-ResNet uses SGD with momentum for all training. Extending to neuromorphic datasets (CIFAR-10-DVS, DVS128 Gesture) raises the question of whether to keep SGD for consistency or switch to AdamW, which is the established optimizer for DVS datasets in recent SNN literature (including the QKFormer baseline this work compares against).

## Decision
Use **SGD with momentum** for static-image datasets (CIFAR-10, CIFAR-100, ImageNet) and **AdamW** for neuromorphic datasets (CIFAR-10-DVS, DVS128 Gesture). The optimizer is resolved from the dataset config dict in the unified training script, not hardcoded.

## Alternatives considered
SGD uniformly across all datasets: preserves consistency with the original paper but risks underperforming on DVS datasets relative to published baselines that use AdamW.

AdamW uniformly: aligns with neuromorphic literature but makes ImageNet and CIFAR results incomparable with the original paper's numbers.

## Consequences
The unified training script must instantiate either SGD or AdamW based on the dataset config. Weight decay values differ between optimizers (1e-5 for SGD on ImageNet, 5e-4 for SGD on CIFAR; to be set from literature for AdamW on DVS datasets). Optimizer state is saved in checkpoints, so checkpoints are not cross-compatible between static and DVS training runs.
