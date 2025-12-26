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
    # Optimize: Direct to tensor -> float conversion (avoids intermediate float64 numpy array)
    return torch.from_numpy(np.array(img)).float().div_(255.0)[None,]


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

        from concurrent.futures import ThreadPoolExecutor
        from comfy.utils import ProgressBar

        pbar = ProgressBar(len(selected))
        imgs = []
        
        # Helper to check existence before loading to keep logic intact
        def load_worker(path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image does not exist: {path}")
            return load_image(path)

        with ThreadPoolExecutor() as executor:
            for img in executor.map(load_worker, selected):
                imgs.append(img)
                pbar.update(1)

        images = torch.cat(imgs, dim=0)
        return (images, len(selected))


NODE_CLASS_MAPPINGS = {
    "BatchLoadImageList": BatchLoadImageList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchLoadImageList": "Batch Load Image List (Advanced)"
}
