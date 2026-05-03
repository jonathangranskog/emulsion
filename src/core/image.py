from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ImageData:
    """
    Basically just a class that stores image data and metadata.
    Could potentially be extended in the future to support other things.
    """

    tensor: torch.Tensor  # [C, H, W]
    metadata: Optional[dict] = None

    @property
    def width(self) -> int:
        return self.tensor.shape[2]

    @property
    def height(self) -> int:
        return self.tensor.shape[1]

    @property
    def channels(self) -> int:
        return self.tensor.shape[0]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype
