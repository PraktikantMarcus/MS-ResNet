# ADR 0001: Per-dataset configurable time window (T)

## Status
Accepted

## Context
The original codebase defines `time_window = 5` as a global constant in `MS_ResNet_fast.py`. For static datasets (CIFAR-10, ImageNet) this value is arbitrary — T only exists so membrane dynamics can propagate, and 5 is sufficient. For DVS datasets (CIFAR-10-DVS, DVS128 Gesture), T is the number of event bins and directly controls temporal resolution. The optimal T differs between datasets and is determined by the SNN literature, not by architectural constraints.

## Decision
Make T a per-dataset parameter passed at training time (e.g. via a command-line argument), rather than a hardcoded global constant. Each dataset configuration specifies its own T. The model constructor accepts T as an argument.

## Alternatives considered
Keep T as a global constant and set it to the highest value needed across all datasets. Rejected because this would inflate memory and compute for datasets that need a lower T, and because it obscures the fact that T has different semantic meaning for static vs. DVS inputs.

## Consequences
Training scripts must pass T explicitly. The model cannot assume a fixed T at definition time. Checkpoints from different datasets are not directly comparable unless T is also recorded.
