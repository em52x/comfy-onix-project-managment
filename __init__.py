from .onix_project import OnixProject, OnixProjectSaver, OnixVideoSaver
from .onix_server import preload  

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "OnixProject": OnixProject,
    "OnixProjectSaver": OnixProjectSaver,
    "OnixVideoSaver": OnixVideoSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnixProject": "Onix Project Loader",
    "OnixProjectSaver": "Onix Project Saver",
    "OnixVideoSaver": "Onix Video Saver",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "preload"]