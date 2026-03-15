from typing import Dict

import torch
from PIL import Image
from torchvision.transforms import functional as F


class Stage1Transform:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, image: Image.Image, mask: Image.Image) -> Dict[str, torch.Tensor]:
        image = F.resize(image, [self.image_size, self.image_size], antialias=True)
        mask = F.resize(mask, [self.image_size, self.image_size], interpolation=Image.NEAREST)

        image_t = F.to_tensor(image)  # [3,H,W], float32 in [0,1]
        mask_t = F.to_tensor(mask)[:1]  # [1,H,W]
        mask_t = (mask_t > 0.5).float()
        return {"image": image_t, "mask": mask_t}
