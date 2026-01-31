import uuid
import os, json, time
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any
import re
import shutil
import folder_paths

try:
    import torchaudio
except ImportError:
    torchaudio = None

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
                "force_duration": ("STRING", {"forceInput": True, "default": ""}),
            },
        }
        
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "INT", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("actual_prompt", "image", "project_id", "current_index", "scene_number", "project_name", "past_prompts_context", "prompt_batch")
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
        force_duration: str = "",
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

        # 1. actual_prompt: Always Single Prompt based on start_prompt
        actual_prompt = ""
        if 0 <= start_prompt < len(lines):
            actual_prompt = lines[start_prompt]

        # 2. prompt_batch: Concatenation based on force_duration or fallback to actual_prompt
        prompt_batch = actual_prompt
        offsets = []
        if force_duration and isinstance(force_duration, str):
            parts = [p.strip() for p in force_duration.split(",") if p.strip()]
            for p in parts:
                try:
                    offsets.append(int(p))
                except:
                    pass
        
        if len(offsets) > 0:
            batch_lines = []
            for offset in offsets:
                idx = start_prompt + offset
                if 0 <= idx < len(lines):
                    batch_lines.append(lines[idx])
            if batch_lines:
                prompt_batch = "\n".join(batch_lines)

        # Generate Past Prompts Context
        past_prompts_context = ""
        if start_prompt > 0:
            limit = min(start_prompt, len(lines))
            history = lines[:limit]
            if history:
                header = "Here are the prompts used for the previous shots in this scene. Use them to maintain visual and narrative continuity:\n"
                body = "\n".join([f"- {h}" for h in history])
                past_prompts_context = header + body

        ui_payload = {"project": [{
            "id": pid, "name": data["name"], "prompt": data["prompt"],
            "prompt_lines": lines, "existing": True, "file": target_file,
            "start_prompt": start_prompt, "scene_number": scene_number, "ts": data["ts"]
        }]}

        return {"ui": ui_payload, "result": (actual_prompt, out_image, pid, start_prompt, scene_number, data["name"], past_prompts_context, prompt_batch)}


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
            "optional": {
                "force_duration": ("STRING", {"forceInput": True, "default": ""}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("saved_frame",)
    OUTPUT_NODE = True
    FUNCTION = "save_last_frame"
    CATEGORY = "Onix Management"

    def save_last_frame(self, images, project_id, current_index, scene_number, force_duration="", fps=24):
        empty_tensor = torch.zeros((1, 64, 64, 3))
        
        if not project_id:
            _log("[Onix Saver] No project_id provided.")
            return {"ui": {}, "result": (empty_tensor,)}

        proj_dir = os.path.join(ONIX_DIR, project_id)
        if not os.path.isdir(proj_dir):
            _log(f"[Onix Saver] Dir not found: {proj_dir}")
            return {"ui": {}, "result": (empty_tensor,)}

        # Parse offsets
        offsets = []
        if force_duration:
            parts = [p.strip() for p in force_duration.split(",") if p.strip()]
            for p in parts:
                try:
                    offsets.append(int(p))
                except:
                    pass
        
        # If no offsets provided, treat as single standard save (equivalent to offset 0 relative to end)
        if not offsets:
            # Legacy behavior: just save the very last frame of the batch as current_index + 1
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
            
            return {"ui": {}, "result": (last_image_tensor.unsqueeze(0),)}

        # Multi-frame save logic based on 5s chunks (aligned with OnixAudioSlicer)
        frames_per_unit = int(5.0 * fps)
        last_saved_tensor = empty_tensor
        
        # Determine total frames available
        total_frames = len(images)

        for i, offset_val in enumerate(offsets):
            # Calculate logical end frame for this chunk
            # If we have offsets 0, 1... 
            # i=0 (offset 0) -> target 5s mark -> frame index (1 * frames_per_unit) - 1
            # i=1 (offset 1) -> target 10s mark -> frame index (2 * frames_per_unit) - 1
            
            target_idx = ((i + 1) * frames_per_unit) - 1
            
            # Safety clamp: if generation was shorter/longer, ensure we stay within bounds
            if target_idx >= total_frames:
                target_idx = total_frames - 1
            
            # Extract frame
            img_tensor = images[target_idx]
            
            # Calculate filename index: start_prompt + offset + 1
            # e.g. start=3, offset=0 -> save 4. offset=1 -> save 5.
            save_index = current_index + offset_val + 1
            
            filename = f"shot_{scene_number}_{save_index:04d}.png"
            save_path = os.path.join(proj_dir, filename)
            
            try:
                arr = 255. * img_tensor.cpu().numpy()
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
                img.save(save_path)
                _log(f"[Onix Saver] Saved multi-frame: {filename} (source frame {target_idx})")
            except Exception as e:
                _log(f"[Onix Saver] Error saving {filename}: {e}")
            
            last_saved_tensor = img_tensor

        return {"ui": {}, "result": (last_saved_tensor.unsqueeze(0),)}


class OnixVideoPrefix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"forceInput": True}),
                "scene_number": ("INT", {"default": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename_prefix",)
    OUTPUT_NODE = False
    FUNCTION = "gen_prefix"
    CATEGORY = "Onix Management"

    def gen_prefix(self, project_name, scene_number):
        # Sanitizar nombre para evitar problemas de ruta
        safe_name = "".join(c for c in project_name if c.isalnum() or c in "._- ")
        if not safe_name:
            safe_name = "Untitled_Project"
            
        # Prefix: Project/Scene/Scene
        # Result example: MyTravel/1/1_0001.mp4 (Comfy adds _0001)
        # Note: ComfyUI Save Video nodes usually create subfolders if slashes are present.
        prefix = f"{safe_name}/{scene_number}/{scene_number}"
        
        return (prefix,)


class OnixAudioSlicer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "current_index": ("INT", {"default": 0}),
                "duration_per_prompt": ("FLOAT", {"default": 5.0, "min": 5.0, "step": 5.0}),
                "enable_preview": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "INT",)
    RETURN_NAMES = ("sliced_audio", "force_duration", "frame_count",)
    FUNCTION = "slice_audio"
    CATEGORY = "Onix Management"

    def slice_audio(self, audio, current_index, duration_per_prompt, enable_preview, fps=24):
        # 1. Determine offsets (force_duration) based on duration chunks of 5s
        # 5s -> [0] -> "0"
        # 10s -> [0, 1] -> "0,1"
        # 15s -> [0, 1, 2] -> "0,1,2"
        num_chunks = int(duration_per_prompt / 5.0)
        chunks = list(range(num_chunks)) if num_chunks > 0 else [0]
        force_duration_str = ",".join(str(c) for c in chunks)
        
        # 2. Slice Audio Logic
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        start_time = current_index * 5.0 
        end_time = start_time + duration_per_prompt
        
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        total_samples = waveform.shape[-1]
        
        if start_sample >= total_samples:
             sliced_waveform = torch.zeros_like(waveform)[..., :100]
             _log(f"[Onix Audio] Index {current_index} start time {start_time}s out of bounds.")
        else:
            end_sample = min(end_sample, total_samples)
            if waveform.ndim == 3:
                sliced_waveform = waveform[:, :, start_sample:end_sample]
            elif waveform.ndim == 2:
                sliced_waveform = waveform[:, start_sample:end_sample]
            else:
                 sliced_waveform = waveform[..., start_sample:end_sample]
                 
            _log(f"[Onix Audio] Sliced {duration_per_prompt}s from {start_time:.2f}s")
            
        result_audio = {"waveform": sliced_waveform, "sample_rate": sample_rate}
        
        # 3. Calculate Frames
        # Formula: (seconds * fps) + 1
        frame_count = int(duration_per_prompt * fps) + 1
        
        # 4. Preview Logic
        ui_payload = {}
        if enable_preview and torchaudio is not None:
            try:
                rand_id = uuid.uuid4().hex[:8]
                filename = f"onix_slice_preview_{rand_id}.wav"
                temp_dir = folder_paths.get_temp_directory()
                full_path = os.path.join(temp_dir, filename)
                
                save_wave = sliced_waveform
                if save_wave.ndim == 3:
                    save_wave = save_wave[0]
                save_wave = save_wave.cpu()
                
                torchaudio.save(full_path, save_wave, sample_rate)
                
                ui_payload = {
                    "ui": {
                        "audio": [{
                            "filename": filename,
                            "type": "temp",
                            "subfolder": ""
                        }]
                    }
                }
            except Exception as e:
                _log(f"[Onix Audio] Failed to save preview: {e}")

        return {"ui": ui_payload, "result": (result_audio, force_duration_str, frame_count,)}

class OnixExecutionTimer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*",),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "passthrough"
    CATEGORY = "Onix Management"

    def passthrough(self, **kwargs):
        # Simply return the first value found in kwargs or None
        val = next(iter(kwargs.values())) if kwargs else None
        return (val,)
