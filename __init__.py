from .onix_project import OnixProject
from .onix_server import preload  

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "OnixProject": OnixProject,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnixProject": "Onix Project Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "preload"]