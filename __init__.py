from .onix_project import OnixProject, OnixProjectSaver, OnixVideoPrefix
from .onix_server import preload  

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "OnixProject": OnixProject,
    "OnixProjectSaver": OnixProjectSaver,
    "OnixVideoPrefix": OnixVideoPrefix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnixProject": "Onix Project Loader",
    "OnixProjectSaver": "Onix Project Saver",
    "OnixVideoPrefix": "Onix Video Prefix",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "preload"]