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
                "frame_count": ("INT", {"default": 16, "min": 0, "step": 1}),
                "ignore_missing_images": (("false", "true"), {"default": "false"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    FUNCTION = "execute"
    DESCRIPTION = """
Loads a sequence of images from disk using a pattern.
- Supports high-performance async loading.
- Ideal for loading rendered sequences or datasets.
- Use pattern like `frame{:06d}.webp` to match filenames.
- Set `frame_count` to 0 to auto-detect all available frames (contiguous).
"""

    def execute(
        self,
        path_pattern: str,
        start_index: int,
        frame_count: int,
        ignore_missing_images: str,
    ):
        ignore_missing_images = ignore_missing_images == "true"
        image_paths = []

        if frame_count == 0:
            # Auto-detect mode: load all contiguous frames starting from start_index
            i = start_index
            while True:
                try:
                    p = path_pattern.format(i)
                except KeyError:
                    p = path_pattern.format(i=i)
                
                if not os.path.exists(p):
                    break
                    
                image_paths.append(p)
                i += 1
                
                # Safety break to avoid infinite loops if something weird happens (e.g. symlink loops? unlikely but good practice)
                if len(image_paths) > 100000:
                    break
        else:
            # Standard Fixed Count Mode
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
            
        return (torch.cat(imgs, dim=0), len(imgs))



class BatchLoadImageSequenceWithTrigger(BatchLoadImageSequence):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = BatchLoadImageSequence.INPUT_TYPES()
        # Move to optional but with forceInput to allow any connection
        inputs["optional"] = {
            "trigger": ("*", {"forceInput": True}),  # Any type trigger
        }
        return inputs

    RETURN_TYPES = ("IMAGE", "INT")
    FUNCTION = "execute_with_trigger"
    INPUT_IS_LIST = True
    
    DESCRIPTION = """
Same as BatchLoadImageSequence but with an optional trigger input.
- Connect any output to `trigger` to force this node to wait.
- Useful for ensuring a process finishes before loading starts.
- This node will NOT unroll (run multiple times) if the trigger is a list.
"""

    def execute_with_trigger(self, path_pattern, start_index, frame_count, ignore_missing_images, trigger=None):
        # When INPUT_IS_LIST is True, all inputs are passed as lists.
        # We extract the single values from the standard inputs.
        return self.execute(
            path_pattern[0], 
            start_index[0], 
            frame_count[0], 
            ignore_missing_images[0]
        )


NODE_CLASS_MAPPINGS = {
    "BatchLoadImageSequence": BatchLoadImageSequence,
    "BatchLoadImageSequenceWithTrigger": BatchLoadImageSequenceWithTrigger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchLoadImageSequence": "Batch Load Image Sequence",
    "BatchLoadImageSequenceWithTrigger": "Batch Load Image Sequence (Trigger)",
}
