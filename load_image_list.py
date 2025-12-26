import os
import torch
from PIL import Image
import numpy as np


def load_image(path: str) -> torch.Tensor:
    """
    Load image from disk and convert to ComfyUI IMAGE tensor
    Shape: [1, H, W, 3], float32 0..1
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


class BatchLoadImageList:
    CATEGORY = "FrameIO"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "paths": ("STRING_LIST",),
            },
            "optional": {
                "start": ("INT", {"default": 0, "min": 0}),
                "end": ("INT", {"default": -1, "min": -1}),
                "step": ("INT", {"default": 1, "min": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    FUNCTION = "execute"

    def execute(
        self,
        paths,
        start=0,
        end=-1,
        step=1,
    ):
        if not paths:
            raise RuntimeError("STRING_LIST is empty")

        total = len(paths)

        # Auto-detect end
        if end < 0 or end > total:
            end = total

        if start >= end:
            raise ValueError("Invalid frame range")

        selected = paths[start:end:step]

        if not selected:
            raise RuntimeError("No frames selected after slicing")

        imgs = []
        for path in selected:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image does not exist: {path}")
            imgs.append(load_image(path))

        images = torch.cat(imgs, dim=0)
        return (images, len(selected))


NODE_CLASS_MAPPINGS = {
    "BatchLoadImageList": BatchLoadImageList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchLoadImageList": "Batch Load Image List (Advanced)"
}
