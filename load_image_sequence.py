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


class BatchLoadImageSequence:
    CATEGORY = "FrameIO"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path_pattern": (
                    "STRING",
                    {"default": "./frame{:06d}.webp", "multiline": False},
                ),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "frame_count": ("INT", {"default": 16, "min": 1, "step": 1}),
                "ignore_missing_images": (("false", "true"), {"default": "false"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        path_pattern: str,
        start_index: int,
        frame_count: int,
        ignore_missing_images: str,
    ):
        ignore_missing_images = ignore_missing_images == "true"

        image_paths = []
        for i in range(start_index, start_index + frame_count):
            try:
                image_paths.append(path_pattern.format(i))
            except KeyError:
                image_paths.append(path_pattern.format(i=i))

        if ignore_missing_images:
            image_paths = [p for p in image_paths if os.path.exists(p)]
        else:
            for path in image_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Image does not exist: {path}")

        if len(image_paths) == 0:
            raise RuntimeError("Image sequence empty - no images to load")

        from concurrent.futures import ThreadPoolExecutor
        from comfy.utils import ProgressBar
        
        pbar = ProgressBar(len(image_paths))
        imgs = []
        
        with ThreadPoolExecutor() as executor:
            for img in executor.map(load_image, image_paths):
                imgs.append(img)
                pbar.update(1)
            
        return (torch.cat(imgs, dim=0),)


NODE_CLASS_MAPPINGS = {
    "BatchLoadImageSequence": BatchLoadImageSequence
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchLoadImageSequence": "Batch Load Image Sequence"
}
