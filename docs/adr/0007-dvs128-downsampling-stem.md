# ADR 0007: Single stride-2 conv stem for DVS128 Gesture

## Status
Accepted

## Context
The `resnet110_cifar` backbone was designed for 32×32 inputs with no initial downsampling. DVS128 Gesture has a 128×128 sensor resolution. Feeding 128×128 directly into the backbone would cause the first residual stage to run at 128×128 — 16× more feature map area than intended, making training prohibitively expensive.

## Decision
Add a single spiking conv layer (3×3, stride 2, padding 1, followed by BatchNorm and membrane update) as a stem before the CIFAR backbone in `resnet110_dvs128`. This reduces 128×128 → 64×64 before the first residual stage.

## Alternatives considered
**Two stride-2 convs (128 → 64 → 32):** Matches the 32×32 resolution the backbone was designed for. Cheaper through residual stages but discards more spatial information upfront. Rejected because DVS128 Gesture gesture patterns are spatially distributed and the literature shows 64×64 is competitive.

**Resize to 64×64 on load (no stem):** Simplest change — no architectural modification needed. Rejected because it happens outside the model and cannot be easily argued as a principled architectural choice; also ties spatial resolution to the dataloader rather than the model.

## Consequences
`resnet110_dvs128` has one additional spiking conv layer compared to `resnet110_cifar`. The first residual stage operates at 64×64 instead of 32×32, meaning it is ~4× more expensive per layer than the CIFAR variant. This cost is accepted as inherent to the larger sensor resolution.

## Superseded — architectural convergence with ADR-0009

When CIFAR-10-DVS was later given the same stride-2 treatment (ADR-0009), `ResNet_CIFAR` gained a `dvs` flag that sets `conv1` stride to 2, producing an identical forward graph to the former `ResNet_DVS128` class. The only difference was the module name (`conv1` vs `stem`), which affected only checkpoint key names.

`ResNet_DVS128` has therefore been removed. The factory functions `resnet20_dvs128` and `resnet110_dvs128` now delegate directly to `ResNet_CIFAR(dvs=True)`, unifying both DVS datasets under a single class. No DVS128 checkpoints existed at the time of this change so there is no migration concern.
