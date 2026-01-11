import uuid
import os, json, time
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any

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

def _last_index_plus_one(base_dir: str) -> int:
    try:
        if not os.path.isdir(base_dir):
            return 0
        max_idx = -1
        import re
        # Look for files like scene_0000.png
        pattern = re.compile(r"scene_(\d+)\.png")
        for name in os.listdir(base_dir):
            match = pattern.match(name)
            if match:
                try:
                    idx = int(match.group(1))
                    if idx > max_idx:
                        max_idx = idx
                except Exception:
                    pass
        return max_idx + 1 if max_idx >= 0 else 0
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
    
    # Calculate start prompt based on existing images
    start_from_files = _last_index_plus_one(proj_dir)
    
    resp = {
        "id": proj_id,
        "name": data.get("name") or os.path.splitext(fname)[0],
        "prompt": ui_prompt,
        "prompt_lines": plines if isinstance(plines, list) else None,
        "existing": True,
        "start_prompt": start_from_files,
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
        
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("actual_prompt", "image", "project_id", "current_index")
    FUNCTION = "apply"
    CATEGORY = "Onix Management"
    
    def apply(
        self,
        enable_image_management: bool,
        project_name: str,
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
                    raise ValueError(f"[Onix] Image Management is ON, start_prompt is 0, but NO 'initial_image' was connected. Please connect an image to start the project.")
                
                # Save scene_0000.png
                img_path = os.path.join(proj_dir, "scene_0000.png")
                # Handle batch dimension -> use first frame
                i = 255. * initial_image[0].cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                img.save(img_path)
                _log(f"[Onix] Saved initial frame to {img_path}")
                out_image = initial_image
            
            else:
                # start_prompt > 0. We MUST load previous frame.
                prev_idx = start_prompt - 1
                prev_img_path = os.path.join(proj_dir, f"scene_{prev_idx:04d}.png")
                
                if not os.path.isfile(prev_img_path):
                    raise FileNotFoundError(f"[Onix] Image Management is ON. Expected previous frame at '{prev_img_path}' for prompt index {start_prompt} (previous={prev_idx}), but it DOES NOT EXIST. Cannot resume.")
                
                # Load it
                try:
                    img = Image.open(prev_img_path).convert("RGB")
                    img_np = np.array(img).astype(np.float32) / 255.0
                    out_image = torch.from_numpy(img_np)[None,]
                    _log(f"[Onix] Loaded frame from {prev_img_path}")
                except Exception as e:
                    raise RuntimeError(f"[Onix] Failed to load previous frame '{prev_img_path}': {e}")
        
        # Fallback / Management OFF
        if out_image is None:
            # Return empty black tensor (1,64,64,3) to satisfy ComfyUI types if connected
            out_image = torch.zeros((1, 64, 64, 3))

        # Determine prompt for current index
        actual_prompt = ""
        if 0 <= start_prompt < len(lines):
            actual_prompt = lines[start_prompt]

        ui_payload = {"project": [{
            "id": pid, "name": data["name"], "prompt": data["prompt"],
            "prompt_lines": lines, "existing": True, "file": target_file,
            "start_prompt": start_prompt, "ts": data["ts"]
        }]}

        return {"ui": ui_payload, "result": (actual_prompt, out_image, pid, start_prompt)}


class OnixProjectSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "project_id": ("STRING", {"default": ""}),
                "current_index": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("saved_frame",)
    OUTPUT_NODE = True
    FUNCTION = "save_last_frame"
    CATEGORY = "Onix Management"

    def save_last_frame(self, images, project_id, current_index):
        # Default empty return for safety
        empty_tensor = torch.zeros((1, 64, 64, 3))
        
        if not project_id:
            _log("[Onix Saver] No project_id provided. Skipping save.")
            return {"ui": {}, "result": (empty_tensor,)}

        proj_dir = os.path.join(ONIX_DIR, project_id)
        if not os.path.isdir(proj_dir):
            _log(f"[Onix Saver] Project directory not found: {proj_dir}")
            return {"ui": {}, "result": (empty_tensor,)}

        # Logic: Save for the NEXT index
        next_idx = current_index + 1
        filename = f"scene_{next_idx:04d}.png"
        save_path = os.path.join(proj_dir, filename)

        # Get the LAST image from the batch
        # images shape is (B, H, W, C)
        last_image_tensor = images[-1]
        
        try:
            i = 255. * last_image_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(save_path)
            _log(f"[Onix Saver] Saved next frame start: {save_path}")
        except Exception as e:
             _log(f"[Onix Saver] Failed to save frame: {e}")

        # Return the saved image (adding batch dimension back)
        # last_image_tensor is (H, W, C), need (1, H, W, C)
        out_tensor = last_image_tensor.unsqueeze(0)
        
        return {"ui": {}, "result": (out_tensor,)}