import os
import torch
import numpy as np
import hashlib
from PIL import Image

from comfy.utils import ProgressBar

def image_tensor_hash(img: torch.Tensor) -> str:
    img = img.clamp(0, 1)
    arr = img.cpu().numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()


class BatchSaveImageSequenceWebP:
    CATEGORY = "FrameIO"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "path_pattern": (
                    "STRING",
                    {"default": "./frame{:06d}.webp", "multiline": False},
                ),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "lossless": (("false", "true"), {"default": "false"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "skip_identical": (("false", "true"), {"default": "true"}),
                "overwrite": (("true", "false"), {"default": "true"}),
            }
        }

    RETURN_TYPES = ("STRING_LIST",)
    RETURN_NAMES = ("paths",)
    FUNCTION = "execute"

    def execute(
        self,
        images,
        path_pattern,
        start_index,
        lossless,
        quality,
        skip_identical,
        overwrite,
    ):
        overwrite = overwrite == "true"
        lossless = lossless == "true"
        skip_identical = skip_identical == "true"

        batch_size = images.shape[0]
        saved_paths = []
        seen_hashes = {}

        pbar = ProgressBar(batch_size)

        for i in range(batch_size):
            index = start_index + i

            try:
                path = path_pattern.format(index)
            except KeyError:
                path = path_pattern.format(i=index)

            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

            img = images[i]

            if skip_identical:
                h = image_tensor_hash(img)
                if h in seen_hashes:
                    saved_paths.append(seen_hashes[h])
                    pbar.update(1)
                    continue

            if os.path.exists(path) and not overwrite:
                saved_paths.append(path)
                pbar.update(1)
                continue

            img = img.clamp(0, 1)
            img = (img.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img, mode="RGB")

            kwargs = {"format": "WEBP", "method": 6}
            if lossless:
                kwargs["lossless"] = True
            else:
                kwargs["lossless"] = False
                kwargs["quality"] = quality

            pil_img.save(path, **kwargs)

            saved_paths.append(path)
            if skip_identical:
                seen_hashes[h] = path

            pbar.update(1)

        return (saved_paths,)


NODE_CLASS_MAPPINGS = {
    "BatchSaveImageSequenceWebP": BatchSaveImageSequenceWebP
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchSaveImageSequenceWebP": "Batch Save Image Sequence (WebP)"
}