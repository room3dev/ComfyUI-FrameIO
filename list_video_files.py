import os
import folder_paths

class ListVideoFiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subfolder": ("STRING", {"default": ""}),
                "filter": ("STRING", {"default": "*"}),
            }
        }

    RETURN_TYPES = ("STRING_LIST", "INT")
    RETURN_NAMES = ("paths", "count")
    FUNCTION = "execute"
    CATEGORY = "FrameIO"
    DESCRIPTION = """
Lists video file paths from the ComfyUI input directory.
- Filters by common video extensions (mp4, mkv, mov, avi, webm, etc.).
- Returns a list of strings and the total count.
- Useful for batch processing multiple videos.
"""

    def execute(self, subfolder, filter):
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
        for root, _, filenames in os.walk(target_dir):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in video_extensions:
                    if filter == "*" or filter.lower() in filename.lower():
                        files.append(os.path.abspath(os.path.join(root, filename)))

        # Sort files naturally
        files.sort()
        
        return (files, len(files))

NODE_CLASS_MAPPINGS = {
    "ListVideoFiles": ListVideoFiles
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ListVideoFiles": "List Video Files (Input Dir)"
}
