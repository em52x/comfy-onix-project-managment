import uuid
import os, json, time
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any
import re
import shutil

try:
    from aiohttp import web as aiohttp_web
except Exception:
    aiohttp_web = None

# —— Resolve Comfy root and onix dir ——
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
for _ in range(6):
    if os.path.isdir(os.path.join(_root, "web")):
        break
    _root = os.path.dirname(_root)

COMFY_ROOT = _root
ONIX_DIR = os.path.join(COMFY_ROOT, "Onix-Proyect-Managment")
os.makedirs(ONIX_DIR, exist_ok=True)

def _log(msg: str):
    try:
        print(msg, flush=True)
    except Exception:
        pass

_log("[Onix] node module loaded")
_log(f"[Onix] COMFY_ROOT={COMFY_ROOT}, ONIX_DIR={ONIX_DIR}")

def _last_index_plus_one(base_dir: str, scene_num: int) -> int:
    try:
        if not os.path.isdir(base_dir):
            return 0
        max_idx = -1
        # Pattern: shot_{scene}_{index}.png -> e.g. shot_1_0005.png
        # We escape the curly braces for regex repetition manually if needed, but here we construct string
        pattern_str = f"shot_{scene_num}_(\d+)\.png"
        pattern = re.compile(pattern_str)
        
        for name in os.listdir(base_dir):
            match = pattern.match(name)
            if match:
                try:
                    idx = int(match.group(1))
                    if idx > max_idx:
                        max_idx = idx
                except Exception:
                    pass
        # Logic fix: if max_idx is 0 (scene_0000.png), we are at prompt 0. 
        # If max_idx is 1 (scene_0001.png), we are ready for prompt 1.
        # So return max_idx directly. If no files (-1), return 0.
        return max_idx if max_idx >= 0 else 0
    except Exception:
        return 0

def _list_projects():
    try:
        items = []
        if os.path.isdir(ONIX_DIR):
            for name in os.listdir(ONIX_DIR):
                if name.lower().endswith(".json"):
                    items.append(name)
        items.sort(key=lambda s: s.lower())
        return items
    except Exception as e:
        _log(f"[Onix] _list_projects error: {e}")
        return []

# —— HTTP API: preview and list ——
async def onix_preview(request):
    _log("[Onix] preview hit")
    if aiohttp_web is None:
        return {"error": "aiohttp not available"}

    fname = request.query.get("file", "")
    fname = "".join(c for c in (fname or "") if c.isalnum() or c in "._-")
    _log(f"[Onix] preview file param: {fname}")

    # We need scene number to calculate start prompt correctly
    scene_param = request.query.get("scene", "1")
    try:
        scene_num = int(scene_param)
    except:
        scene_num = 1

    if not fname or not fname.lower().endswith(".json"):
        return aiohttp_web.json_response({"error": "bad file"}, status=400)

    path = os.path.join(ONIX_DIR, fname)
    if not os.path.isfile(path):
        return aiohttp_web.json_response({"error": "not found"}, status=404)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return aiohttp_web.json_response({"error": str(e)}, status=500)

    plines = data.get("prompt_lines")
    if isinstance(plines, list):
        ui_prompt = "\n".join(str(x) for x in plines)
    else:
        ui_prompt = data.get("prompt") or ""

    proj_id = data.get("id") or uuid.uuid4().hex
    proj_dir = os.path.join(ONIX_DIR, proj_id)
    
    # Calculate start prompt based on existing images for THIS scene
    start_from_files = _last_index_plus_one(proj_dir, scene_num)
    
    resp = {
        "id": proj_id,
        "name": data.get("name") or os.path.splitext(fname)[0],
        "prompt": ui_prompt,
        "prompt_lines": plines if isinstance(plines, list) else None,
        "existing": True,
        "start_prompt": start_from_files,
        "scene_number": scene_num,
        "file": fname,
    }
    return aiohttp_web.json_response(resp, status=200)

async def onix_list_projects(request):
    try:
        files = [f for f in _list_projects() if f.lower().endswith(".json")]
        return aiohttp_web.json_response({"files": files}, status=200)
    except Exception as e:
        _log(f"[Onix] list_projects error: {e}")
        return aiohttp_web.json_response({"files": []}, status=200)

def setup_routes(app):
    if aiohttp_web is None:
        return
    try:
        app.router.add_get("/onix/preview", onix_preview)
        app.router.add_get("/onix/projects", onix_list_projects)
    except Exception:
        pass

# Fallback registration
try:
    from server import PromptServer as _PS
    setup_routes(_PS.instance.app)
except Exception:
    pass

class OnixProject:
    @classmethod
    def INPUT_TYPES(cls):
        files = _list_projects()
        enum = ["none"] + files
        return {
            "required": {
                "enable_image_management": ("BOOLEAN", {"default": True}),
                "project_name": ("STRING", {"default": "", "multiline": False}),
                "scene_number": ("INT", {"default": 1, "min": 0}),
                "positive_text": ("STRING", {"default": "", "multiline": True}),
                "start_prompt": ("INT", {"default": 0}),
            },
            "optional": {
                "initial_image": ("IMAGE",),
                "project_list": (enum, {"default": "none"}),
                "existing_project": ("BOOLEAN", {"default": False}),
                "project_id": ("STRING", {"default": ""}),
            },
        }
        
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("actual_prompt", "image", "project_id", "current_index", "scene_number")
    FUNCTION = "apply"
    CATEGORY = "Onix Management"
    
    def apply(
        self,
        enable_image_management: bool,
        project_name: str,
        scene_number: int,
        positive_text: str,
        start_prompt: int,
        initial_image: torch.Tensor = None,
        project_list: str = "none",
        existing_project: bool = False,
        project_id: str = "",
    ):
        sel = (project_list or "").strip()
        is_file = bool(sel and sel.lower().endswith(".json") and sel != "none")

        file_id = ""
        if is_file:
            try:
                selected_path = os.path.join(ONIX_DIR, sel)
                if os.path.isfile(selected_path):
                    with open(selected_path, "r", encoding="utf-8") as f:
                        old = json.load(f)
                    file_id = (old.get("id") or "").strip()
            except Exception:
                pass

        if existing_project:
            pid = (project_id or "").strip() or file_id or uuid.uuid4().hex
        else:
            pid = uuid.uuid4().hex

        proj_dir = os.path.join(ONIX_DIR, pid)
        os.makedirs(proj_dir, exist_ok=True)
        
        # Determine target filename for JSON
        if is_file and existing_project:
            target_file = sel
        else:
            def safe_n(s): return "".join(c for c in (s or "") if c.isalnum() or c in "._-")
            base_name = safe_n(project_name) or f"project_{pid[:8]}"
            if not base_name.lower().endswith(".json"):
                base_name = f"{base_name}.json"
            candidate = base_name
            if os.path.isfile(os.path.join(ONIX_DIR, candidate)):
                stem, ext = os.path.splitext(base_name)
                suffix = 1
                while os.path.isfile(os.path.join(ONIX_DIR, f"{stem}_{suffix}{ext}")):
                    suffix += 1
                candidate = f"{stem}_{suffix}{ext}"
            target_file = candidate

        path = os.path.join(ONIX_DIR, target_file)

        # Process Prompts
        prompt_val = positive_text or ""
        lines = [s.strip() for s in prompt_val.replace("\r\n", "\n").replace("\r", "\n").split("\n") if s.strip()]

        data = {
            "id": pid,
            "name": project_name or f"project_{pid[:8]}",
            "prompt": "\n".join(lines),
            "prompt_lines": lines,
            "existing": True,
            "file": target_file,
            "start_prompt": start_prompt,
            "scene_number": scene_number,
            "ts": time.time(),
        }

        # Save JSON
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Handle Image Logic (Strict Mode)
        out_image = None
        
        if enable_image_management:
            if start_prompt == 0:
                # Require initial_image
                if initial_image is None:
                    raise ValueError(f"[Onix] Start Prompt 0 in Scene {scene_number} requires an 'initial_image' input.")
                
                # Save shot_{scene}_0000.png
                filename = f"shot_{scene_number}_0000.png"
                img_path = os.path.join(proj_dir, filename)
                
                i = 255. * initial_image[0].cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                img.save(img_path)
                _log(f"[Onix] Saved initial frame: {filename}")
                out_image = initial_image
            
            else:
                # Resume: Load shot_{scene}_{start_prompt}.png (the output of the previous step serves as input for this one)
                # If start_prompt is 1, we want the image saved by step 0 (which is saved as 0001).
                # Wait... Saver saves as current_index + 1.
                # Step 0 -> Saves 0001.
                # Step 1 -> Needs 0001.
                # So yes, we need file index = start_prompt.
                filename = f"shot_{scene_number}_{start_prompt:04d}.png"
                prev_img_path = os.path.join(proj_dir, filename)
                
                if not os.path.isfile(prev_img_path):
                    raise FileNotFoundError(f"[Onix] Missing previous frame for resume: {filename} in {proj_dir}")
                
                try:
                    img = Image.open(prev_img_path).convert("RGB")
                    img_np = np.array(img).astype(np.float32) / 255.0
                    out_image = torch.from_numpy(img_np)[None,]
                    _log(f"[Onix] Loaded frame: {filename}")
                except Exception as e:
                    raise RuntimeError(f"[Onix] Failed to load {filename}: {e}")
        
        # Fallback
        if out_image is None:
            out_image = torch.zeros((1, 64, 64, 3))

        # Determine prompt
        actual_prompt = ""
        if 0 <= start_prompt < len(lines):
            actual_prompt = lines[start_prompt]

        ui_payload = {"project": [{
            "id": pid, "name": data["name"], "prompt": data["prompt"],
            "prompt_lines": lines, "existing": True, "file": target_file,
            "start_prompt": start_prompt, "scene_number": scene_number, "ts": data["ts"]
        }]}

        return {"ui": ui_payload, "result": (actual_prompt, out_image, pid, start_prompt, scene_number)}


class OnixProjectSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "project_id": ("STRING", {"default": ""}),
                "current_index": ("INT", {"default": 0}),
                "scene_number": ("INT", {"default": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("saved_frame",)
    OUTPUT_NODE = True
    FUNCTION = "save_last_frame"
    CATEGORY = "Onix Management"

    def save_last_frame(self, images, project_id, current_index, scene_number):
        empty_tensor = torch.zeros((1, 64, 64, 3))
        
        if not project_id:
            _log("[Onix Saver] No project_id provided.")
            return {"ui": {}, "result": (empty_tensor,)}

        proj_dir = os.path.join(ONIX_DIR, project_id)
        if not os.path.isdir(proj_dir):
            _log(f"[Onix Saver] Dir not found: {proj_dir}")
            return {"ui": {}, "result": (empty_tensor,)}

        # Save NEXT index: shot_{scene}_{current+1}.png
        next_idx = current_index + 1
        filename = f"shot_{scene_number}_{next_idx:04d}.png"
        save_path = os.path.join(proj_dir, filename)

        last_image_tensor = images[-1]
        
        try:
            i = 255. * last_image_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(save_path)
            _log(f"[Onix Saver] Saved: {filename}")
        except Exception as e:
             _log(f"[Onix Saver] Error saving: {e}")

class OnixVideoSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"forceInput": True}),
                "project_id": ("STRING", {"default": ""}),
                "current_index": ("INT", {"default": 0}),
                "scene_number": ("INT", {"default": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_video_path",)
    OUTPUT_NODE = True
    FUNCTION = "save_video_shot"
    CATEGORY = "Onix Management"

    def save_video_shot(self, video_path, project_id, current_index, scene_number):
        if not project_id:
            _log("[Onix Video] No project_id provided.")
            return ("",)
            
        # Handle input variations (list of paths vs single string)
        if isinstance(video_path, list) and len(video_path) > 0:
            source_path = str(video_path[0])
        else:
            source_path = str(video_path)

        # Basic cleanup of source path (remove quotes if any)
        source_path = source_path.strip('"').strip("'")
            
        if not os.path.isfile(source_path):
             _log(f"[Onix Video] Source file not found or invalid: {source_path}")
             return ("",)

        proj_dir = os.path.join(ONIX_DIR, project_id)
        if not os.path.isdir(proj_dir):
            _log(f"[Onix Video] Project dir not found: {proj_dir}")
            return ("",)
            
        video_out_dir = os.path.join(proj_dir, "Video_Shots_Output")
        os.makedirs(video_out_dir, exist_ok=True)

        # Naming: videoShot_{scene}_{next_idx}.mp4
        # Using current_index + 1 to align with the 'finished shot' logic
        next_idx = current_index + 1
        filename = f"videoShot_{scene_number}_{next_idx:04d}.mp4"
        dest_path = os.path.join(video_out_dir, filename)

        try:
            shutil.copy2(source_path, dest_path)
            _log(f"[Onix Video] Saved video to: {dest_path}")
            return (dest_path,)
        except Exception as e:
            _log(f"[Onix Video] Error saving video: {e}")
            return ("",)
