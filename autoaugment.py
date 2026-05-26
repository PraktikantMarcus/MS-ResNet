import math
import torch
from torch import Tensor
from typing import List, Tuple, Optional, Dict
from torchvision.transforms import functional as F, InterpolationMode
from torchvision.transforms.transforms import RandomErasing


def _apply_op(img: Tensor, op_name: str, magnitude: float,
              interpolation: InterpolationMode, fill: Optional[List[float]]):
    if op_name == "ShearX":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                       interpolation=interpolation, fill=fill)
    elif op_name == "TranslateX":
        img = F.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "TranslateY":
        img = F.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img


class SNNAugmentWide(torch.nn.Module):
    """Geometric-only TrivialAugment variant for event-based (DVS) data.

    Applies one randomly chosen geometric transform per sample. Colour
    operations are intentionally excluded because DVS frames encode
    polarity (ON/OFF), not colour — brightness or contrast transforms
    would corrupt the spike representation.

    Source: QKFormer codebase (adapted for MS-ResNet).
    """

    def __init__(self,
                 num_magnitude_bins: int = 31,
                 interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: Optional[List[float]] = None) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.cutout = RandomErasing(p=1, scale=(0.001, 0.11), ratio=(1, 1))

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity":   (torch.tensor(0.0), False),
            "ShearX":     (torch.linspace(-0.3, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(-5.0, 5.0, num_bins), True),
            "TranslateY": (torch.linspace(-5.0, 5.0, num_bins), True),
            "Rotate":     (torch.linspace(-30.0, 30.0, num_bins), True),
            "Cutout":     (torch.linspace(1.0, 30.0, num_bins), True),
        }

    def forward(self, img: Tensor) -> Tensor:
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = float(
            magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item()
        ) if magnitudes.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        if op_name == "Cutout":
            return self.cutout(img)
        return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
