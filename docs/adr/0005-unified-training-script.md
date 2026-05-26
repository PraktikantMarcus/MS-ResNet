# ADR 0005: Unified training script

## Status
Accepted

## Context
The original codebase has one training script per use case: `train_amp.py` and `train_amp_hpc.py` for ImageNet, `train_cifar10_hpc.py` for CIFAR-10. Adding three new datasets (CIFAR-100, CIFAR-10-DVS, DVS128 Gesture) would produce five or more scripts that duplicate the training loop, checkpointing logic, and logging. Keeping them separate also makes it harder to argue experimental consistency across datasets.

## Decision
Retire the per-dataset training scripts and replace them with a single `train_hpc.py` that selects the model, dataloader, T, augmentation, and optimizer settings via a `--dataset` argument. Dataset-specific parameters (T, in_channels, num_classes, spatial resolution, augmentation) are resolved inside the script from the dataset name. The training loop itself is shared across all datasets.

The old scripts (`train_amp.py`, `train_amp_hpc.py`, `train_cifar10_hpc.py`) are kept for reference but are no longer the primary entry point.

## Alternatives considered
One script per dataset. Rejected because it duplicates the training loop and checkpointing logic, and forces manual synchronisation of any improvement (e.g. a fix to checkpoint saving) across all scripts.

## Consequences
The unified script must handle both distributed (ImageNet) and single-GPU (CIFAR, DVS) modes. The SLURM submission scripts must be updated to call `train_hpc.py --dataset <name>`. All dataset-specific configuration lives in one place, making it easy to audit what differs between experiments.
