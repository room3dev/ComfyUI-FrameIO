import os
import folder_paths

class ListVideoFiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subfolder": ("STRING", {"default": ""}),
                "filter": ("STRING", {"default": "*"}),
                "load_cap": ("INT", {"default": 100, "min": 0, "max": 10000, "step": 1}),
                "deep_search": (("false", "true"), {"default": "false"}),
            }
        }

    RETURN_TYPES = ("STRING_LIST", "INT")
    RETURN_NAMES = ("paths", "count")
    FUNCTION = "execute"
    CATEGORY = "FrameIO"
    DESCRIPTION = """
Lists video file paths from the ComfyUI input directory.
- Filters by common video extensions (mp4, mkv, mov, avi, webm, etc.).
- `load_cap`: Maximum number of files to return (0 for no limit).
- `deep_search`: If true, searches all subdirectories recursively.
- Returns a list of strings and the total count.
"""

    def execute(self, subfolder, filter, load_cap, deep_search):
        deep_search = deep_search == "true"
        input_dir = folder_paths.get_input_directory()
        target_dir = os.path.abspath(os.path.join(input_dir, subfolder))

        if not os.path.exists(target_dir):
            return ([], 0)

        video_extensions = {
            ".mp4", ".mkv", ".mov", ".avi", ".webm", 
            ".mpg", ".mpeg", ".m4v", ".flv", ".wmv",
            ".3gp", ".ts", ".m2ts"
        }

        files = []
        if deep_search:
            for root, _, filenames in os.walk(target_dir):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in video_extensions:
                        if filter == "*" or filter.lower() in filename.lower():
                            files.append(os.path.abspath(os.path.join(root, filename)))
        else:
            for filename in os.listdir(target_dir):
                full_path = os.path.join(target_dir, filename)
                if os.path.isfile(full_path):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in video_extensions:
                        if filter == "*" or filter.lower() in filename.lower():
                            files.append(os.path.abspath(full_path))

        # Sort files naturally
        files.sort()

        # Apply load cap
        if load_cap > 0:
            files = files[:load_cap]
        
        return (files, len(files))

NODE_CLASS_MAPPINGS = {
    "ListVideoFiles": ListVideoFiles
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ListVideoFiles": "List Video Files (Input Dir)"
}
