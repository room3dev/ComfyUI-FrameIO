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
                "lossless": (("false", "true"), {"default": "true"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "skip_identical": (("false", "true"), {"default": "false"}),
                "overwrite": (("true", "false"), {"default": "true"}),
            }
        }

    RETURN_TYPES = ("STRING_LIST", "INT")
    RETURN_NAMES = ("paths", "count")
    FUNCTION = "execute"
    DESCRIPTION = """
Saves a batch of images as a WebP sequence.
- Optimized for long sequences with async saving.
- Supports lossy and lossless compression.
- Can skip identical frames to save space.
- Restricted to ComfyUI output directory for security.
"""

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

        # Prepare executor for async saving
        from concurrent.futures import ThreadPoolExecutor
        import folder_paths

        def save_worker(img_tensor, file_path, lossless_mode, quality_val, file_format):
            # Convert tensor to numpy/PIL
            img_np = img_tensor.clamp(0, 1).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np, mode="RGB")
            
            # Save arguments
            save_args = {"format": file_format, "method": 6}
            if lossless_mode:
                save_args["lossless"] = True
            else:
                save_args["lossless"] = False
                save_args["quality"] = quality_val
                
            pil_image.save(file_path, **save_args)

        # Using max_workers=None lets python decide based on cpu count, 
        # but we might want to limit it to avoid too much memory usage/disk contention. 
        # For now, let's stick to default or a reasonable number like 4-8? 
        # Default is usually cpu_count + 4. Let's start with default.
        with ThreadPoolExecutor() as executor:
            futures = []
            
            # Security check: Get base output path
            base_output_dir = os.path.abspath(folder_paths.get_output_directory())

            for i in range(batch_size):
                index = start_index + i

                try:
                    path = path_pattern.format(index)
                except KeyError:
                    path = path_pattern.format(i=index)
                
                # Resolve absolute path
                abs_path = os.path.abspath(path)

                # Check if path is within output directory
                # commonpath raises ValueError if paths are on different drives (Windows)
                try:
                    common = os.path.commonpath([base_output_dir, abs_path])
                    if common != base_output_dir:
                         raise PermissionError(f"Security: Cannot write to {abs_path}. Must be inside {base_output_dir}")
                except ValueError:
                     # Different drives on Windows will assume violation
                     raise PermissionError(f"Security: Cannot write to {abs_path}. Must be inside {base_output_dir}")

                os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)

                img = images[i]

                if skip_identical:
                    h = image_tensor_hash(img)
                    if h in seen_hashes:
                        saved_paths.append(seen_hashes[h])
                        pbar.update(1)
                        continue

                # Check overwrite before doing work
                if os.path.exists(abs_path) and not overwrite:
                    saved_paths.append(abs_path)
                    pbar.update(1)
                    continue

                # Submit save task
                future = executor.submit(
                    save_worker, 
                    img, 
                    abs_path, 
                    lossless, 
                    quality, 
                    "WEBP"
                )
                futures.append(future)

                saved_paths.append(path)
                if skip_identical:
                    seen_hashes[h] = path
                
            # Wait for all to finish ensuring files are properly closed
            # Update pbar as tasks complete
            from concurrent.futures import as_completed
            for f in as_completed(futures):
                pbar.update(1)
                # Check for exceptions
                f.result()

        return (saved_paths, len(saved_paths))


NODE_CLASS_MAPPINGS = {
    "BatchSaveImageSequenceWebP": BatchSaveImageSequenceWebP
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchSaveImageSequenceWebP": "Batch Save Image Sequence (WebP)"
}