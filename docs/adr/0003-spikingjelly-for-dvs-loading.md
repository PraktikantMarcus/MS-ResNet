# ADR 0003: SpikingJelly for DVS dataset loading

## Status
Accepted

## Context
The original MS-ResNet codebase contains no external SNN library dependency — neuron dynamics, surrogate gradients, and data loading are all custom-implemented. Extending to CIFAR-10-DVS and DVS128 Gesture requires parsing AEDAT event files, which is non-trivial to reimplement correctly (file format handling, timestamp normalisation, split logic).

A downstream requirement also exists: the event-to-frame conversion pipeline must eventually produce saveable binary spike sequences for RTL hardware testbench replay. This means the conversion logic must be transparent and controllable.

## Decision
Use SpikingJelly's built-in dataset classes (`spikingjelly.datasets.cifar10_dvs`, `spikingjelly.datasets.dvs128_gesture`) for raw event file parsing and frame binning. Write the binarisation and spike-sequence saving steps ourselves on top of SpikingJelly's output. The neuron dynamics inside the network remain the original custom implementation — SpikingJelly is used only at the data loading boundary.

## Alternatives considered
**Full custom implementation:** Reimplement AEDAT parsing and event binning without SpikingJelly. Gives complete control but duplicates well-tested library code and would need to be argued and maintained separately. Rejected because the RTL requirement is about the binarised input format, not about the file parsing step.

## Consequences
SpikingJelly becomes a new project dependency. The custom neuron implementation inside `MS_ResNet_fast.py` is not replaced. The binarisation and saving layer sits between SpikingJelly's output and the network input, and is owned by this project.
