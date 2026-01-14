from .onix_project import OnixProject, OnixProjectSaver, OnixVideoPrefix, OnixAudioSlicer
from .onix_server import preload  

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "OnixProject": OnixProject,
    "OnixProjectSaver": OnixProjectSaver,
    "OnixVideoPrefix": OnixVideoPrefix,
    "OnixAudioSlicer": OnixAudioSlicer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnixProject": "Onix Project Loader",
    "OnixProjectSaver": "Onix Project Saver",
    "OnixVideoPrefix": "Onix Video Prefix",
    "OnixAudioSlicer": "Onix Audio Slicer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "preload"]
